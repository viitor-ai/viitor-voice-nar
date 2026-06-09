from __future__ import annotations

import hashlib
import os
import threading
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from viitorvoice import paths

try:
    from dualcodec.infer.dualcodec.inference_with_semantic import _build_semantic_model
except Exception:  # pragma: no cover - optional package internals
    _build_semantic_model = None

try:
    from dualcodec.infer.dualcodec.get_model import get_model as _get_dualcodec_model
except Exception:  # pragma: no cover - optional package internals
    _get_dualcodec_model = None


DEFAULT_DUALCODEC_PATH = paths.dualcodec_ckpt_dir()
DEFAULT_W2V_PATH = paths.dualcodec_w2v_path()
DEFAULT_ONNX_PATH = DEFAULT_DUALCODEC_PATH / "dualcodec_encode_core_30s.onnx"
DEFAULT_TRT_CACHE_ROOT = paths.dualcodec_encoder_trt_cache_root()
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_W2V_SAMPLE_RATE = 16000
DEFAULT_FRAME_RATE = 25
DEFAULT_NUM_CODEBOOKS = 12
DEFAULT_ACOUSTIC_CODEBOOKS = DEFAULT_NUM_CODEBOOKS - 1
DEFAULT_SEMANTIC_DOWNSAMPLE_FACTOR = 2


def _clear_proxies() -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        os.environ.pop(key, None)


@dataclass
class DualCodecEncoderConfig:
    """Configuration for DualCodec W2V + ONNX encode."""

    w2v_path: Path = DEFAULT_W2V_PATH
    onnx_path: Path = DEFAULT_ONNX_PATH
    ep_config: Mapping[str, Any] = field(
        default_factory=lambda: {
            "backend": "trt",
            "device_id": 0,
            "trt_fp16_enable": True,
        }
    )
    dualcodec_path: Path | None = None
    trt_cache_root: Path = DEFAULT_TRT_CACHE_ROOT
    max_samples: int = DEFAULT_SAMPLE_RATE * 30
    max_frames: int = DEFAULT_FRAME_RATE * 30
    min_samples: int = 1
    min_frames: int = 1
    semantic_downsample_factor: int = DEFAULT_SEMANTIC_DOWNSAMPLE_FACTOR


class W2VSemanticExtractor:
    """W2V-only semantic extractor; does not load the DualCodec torch model."""

    def __init__(
        self,
        dualcodec_path: Path,
        w2v_path: Path,
        device: torch.device,
    ) -> None:
        if _build_semantic_model is None:
            raise RuntimeError("dualcodec semantic builder is unavailable.")
        self.device = device
        semantic_cfg = _build_semantic_model(
            dualcodec_path=str(dualcodec_path),
            semantic_model_path=str(w2v_path),
            device=str(device),
        )
        for key, value in list(semantic_cfg.items()):
            if isinstance(value, (torch.nn.Module, torch.Tensor)):
                semantic_cfg[key] = value.to(device)
        self.semantic_cfg = semantic_cfg

    @torch.no_grad()
    def extract(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        semantic_model = self.semantic_cfg["semantic_model"]
        vq_emb = semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[self.semantic_cfg["output_idx"]]
        if not (
            hasattr(self.semantic_cfg, "skip_semantic_normalize")
            and self.semantic_cfg.skip_semantic_normalize
        ):
            feat = (feat - self.semantic_cfg["mean"]) / self.semantic_cfg["std"]
        return feat


class DualCodecOnnxEncoder:
    """DualCodec encoder using W2V semantic extraction plus encode-core ONNX."""

    def __init__(
        self,
        w2v_path: str | Path | None = None,
        onnx_path: str | Path | None = None,
        ep_config: Mapping[str, Any] | None = None,
        dualcodec_path: str | Path | None = None,
        trt_cache_root: str | Path | None = None,
        max_samples: int = DEFAULT_SAMPLE_RATE * 30,
        max_frames: int = DEFAULT_FRAME_RATE * 30,
        min_samples: int = 1,
        min_frames: int = 1,
        semantic_downsample_factor: int = DEFAULT_SEMANTIC_DOWNSAMPLE_FACTOR,
    ) -> None:
        _clear_proxies()
        config = DualCodecEncoderConfig(
            w2v_path=Path(w2v_path) if w2v_path is not None else DEFAULT_W2V_PATH,
            onnx_path=Path(onnx_path) if onnx_path is not None else DEFAULT_ONNX_PATH,
            ep_config=dict(ep_config or {"backend": "trt", "device_id": 0}),
            dualcodec_path=Path(dualcodec_path) if dualcodec_path is not None else None,
            trt_cache_root=(
                Path(trt_cache_root)
                if trt_cache_root is not None
                else DEFAULT_TRT_CACHE_ROOT
            ),
            max_samples=int(max_samples),
            max_frames=int(max_frames),
            min_samples=int(min_samples),
            min_frames=int(min_frames),
            semantic_downsample_factor=int(semantic_downsample_factor),
        )
        self.config = self._resolve_config(config)
        self.backend = str(self.config.ep_config.get("backend", "trt")).lower()
        if self.backend in {"ort_trt", "onnx_trt", "tensorrt"}:
            self.backend = "trt"
        if self.backend in {"ort_cuda", "onnx_cuda"}:
            self.backend = "cuda"
        if self.backend not in {"trt", "cuda"}:
            raise ValueError(
                f"Unsupported DualCodec encoder backend {self.backend!r}; "
                "expected 'trt' or 'cuda'."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("DualCodec ONNX encoder requires CUDA.")

        self.device_id = int(self.config.ep_config.get("device_id", 0))
        self.device = torch.device(f"cuda:{self.device_id}")
        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.compute_stream_ptr = str(int(self.compute_stream.cuda_stream))
        self._io_binding_lock = threading.Lock()
        self.semantic_extractor = W2VSemanticExtractor(
            dualcodec_path=self.config.dualcodec_path,
            w2v_path=self.config.w2v_path,
            device=self.device,
        )
        self.session = self._create_session()
        self.active_providers = list(self.session.get_providers())
        self.io_binding = self.session.io_binding()

    @classmethod
    def from_config(cls, config: DualCodecEncoderConfig) -> "DualCodecOnnxEncoder":
        return cls(
            w2v_path=config.w2v_path,
            onnx_path=config.onnx_path,
            ep_config=config.ep_config,
            dualcodec_path=config.dualcodec_path,
            trt_cache_root=config.trt_cache_root,
            max_samples=config.max_samples,
            max_frames=config.max_frames,
            min_samples=config.min_samples,
            min_frames=config.min_frames,
            semantic_downsample_factor=config.semantic_downsample_factor,
        )

    def __call__(self, audio_numpy: np.ndarray) -> np.ndarray:
        return self.encode(audio_numpy)

    def encode(self, audio_numpy: np.ndarray) -> np.ndarray:
        """Encode ``float32`` 24k audio shaped ``(1, 1, T)`` into ``int32 [12, t]``."""
        audio = self._normalize_audio(audio_numpy)
        with torch.inference_mode():
            semantic_repr = self._extract_semantic_repr(audio)
            semantic_codes, acoustic_codes = self._encode_core(audio, semantic_repr)
            tokens = torch.cat([semantic_codes, acoustic_codes], dim=1)
        return tokens.squeeze(0).to(torch.int32).cpu().numpy()

    def _resolve_config(self, config: DualCodecEncoderConfig) -> DualCodecEncoderConfig:
        config.w2v_path = config.w2v_path.expanduser().resolve()
        config.onnx_path = config.onnx_path.expanduser().resolve()
        if config.dualcodec_path is None:
            config.dualcodec_path = config.onnx_path.parent
        config.dualcodec_path = config.dualcodec_path.expanduser().resolve()
        config.trt_cache_root = config.trt_cache_root.expanduser().resolve()
        required = [
            config.w2v_path / "config.json",
            config.w2v_path / "model.safetensors",
            config.w2v_path / "preprocessor_config.json",
            config.dualcodec_path / "w2vbert2_mean_var_stats_emilia.pt",
            config.onnx_path,
        ]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Required DualCodec encoder asset not found: {path}")
        if config.max_samples <= 0 or config.max_frames <= 0:
            raise ValueError(
                f"max_samples/max_frames should be positive, got "
                f"{config.max_samples}/{config.max_frames}."
            )
        if config.min_samples <= 0 or config.min_frames <= 0:
            raise ValueError(
                f"min_samples/min_frames should be positive, got "
                f"{config.min_samples}/{config.min_frames}."
            )
        if config.min_samples > config.max_samples:
            raise ValueError("min_samples cannot be greater than max_samples.")
        if config.min_frames > config.max_frames:
            raise ValueError("min_frames cannot be greater than max_frames.")
        if config.semantic_downsample_factor <= 0:
            raise ValueError(
                "semantic_downsample_factor should be positive, got "
                f"{config.semantic_downsample_factor}."
            )
        return config

    def _create_session(self):
        import onnxruntime as ort

        self.config.trt_cache_root.mkdir(parents=True, exist_ok=True)
        cuda_providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": self.device_id,
                    "user_compute_stream": self.compute_stream_ptr,
                },
            ),
            ("CPUExecutionProvider", {}),
        ]
        providers = cuda_providers
        if self.backend == "trt":
            cache_dir = self._trt_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            profile_min = (
                f"audio:1x1x{self.config.min_samples},"
                f"semantic_repr:1x1024x{self.config.min_frames}"
            )
            profile_opt = (
                f"audio:1x1x{self.config.max_samples},"
                f"semantic_repr:1x1024x{self.config.max_frames}"
            )
            profile_max = profile_opt
            providers = [
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": self.device_id,
                        "user_compute_stream": self.compute_stream_ptr,
                        "trt_fp16_enable": bool(
                            self.config.ep_config.get("trt_fp16_enable", True)
                        ),
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": str(cache_dir),
                        "trt_profile_min_shapes": profile_min,
                        "trt_profile_opt_shapes": profile_opt,
                        "trt_profile_max_shapes": profile_max,
                        "trt_layer_norm_fp32_fallback": bool(
                            self.config.ep_config.get(
                                "trt_layer_norm_fp32_fallback", True
                            )
                        ),
                    },
                ),
            ] + cuda_providers

        session = ort.InferenceSession(str(self.config.onnx_path), providers=providers)
        active = session.get_providers()
        if self.backend == "trt" and "TensorrtExecutionProvider" not in active:
            self.backend = "cuda"
            session = ort.InferenceSession(
                str(self.config.onnx_path),
                providers=cuda_providers,
            )
            active = session.get_providers()
        if (
            "CUDAExecutionProvider" not in active
            and "TensorrtExecutionProvider" not in active
        ):
            raise RuntimeError(
                f"Neither TRT nor CUDA EP is active for DualCodec encoder: {active}"
            )
        return session

    def _trt_cache_dir(self) -> Path:
        st = os.stat(self.config.onnx_path)
        cache_key = hashlib.md5(
            (
                str(self.config.onnx_path)
                + f"|sig{st.st_size}:{st.st_mtime_ns}"
                + f"|encode|max_samples{self.config.max_samples}"
                + f"|max_frames{self.config.max_frames}"
                + f"|fp16{bool(self.config.ep_config.get('trt_fp16_enable', True))}"
            ).encode("utf-8")
        ).hexdigest()
        return self.config.trt_cache_root / f"trt_cache_{cache_key}"

    def _normalize_audio(self, audio_numpy: np.ndarray) -> torch.Tensor:
        if not isinstance(audio_numpy, np.ndarray):
            raise TypeError(f"audio_numpy should be np.ndarray, got {type(audio_numpy)!r}.")
        if audio_numpy.shape[0:2] != (1, 1) or audio_numpy.ndim != 3:
            raise ValueError(
                "DualCodec encoder expects audio_numpy shape (1, 1, T), "
                f"got {audio_numpy.shape}."
            )
        if audio_numpy.dtype != np.float32:
            raise ValueError(
                f"DualCodec encoder expects float32 audio, got {audio_numpy.dtype}."
            )
        if audio_numpy.shape[-1] > self.config.max_samples:
            raise ValueError(
                f"audio has {audio_numpy.shape[-1]} samples, exceeding "
                f"max_samples={self.config.max_samples}."
            )
        audio = torch.from_numpy(np.ascontiguousarray(audio_numpy)).to(self.device)
        return audio.to(torch.float32)

    def _semantic_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _extract_semantic_repr(self, audio: torch.Tensor) -> torch.Tensor:
        audio_16k = torchaudio.functional.resample(
            audio,
            DEFAULT_SAMPLE_RATE,
            DEFAULT_W2V_SAMPLE_RATE,
        )
        feature_extractor = self.semantic_extractor.semantic_cfg["feature_extractor"]
        inputs = feature_extractor(
            audio_16k.cpu(),
            sampling_rate=DEFAULT_W2V_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs["input_features"][0].unsqueeze(0).to(self.device)
        attention_mask = inputs["attention_mask"][0].unsqueeze(0).to(self.device)
        with self._semantic_context():
            feat = self.semantic_extractor.extract(
                input_features,
                attention_mask,
            ).transpose(1, 2)
            feat = F.avg_pool1d(
                feat,
                self.config.semantic_downsample_factor,
                self.config.semantic_downsample_factor,
            )
        if feat.shape[-1] > self.config.max_frames:
            raise ValueError(
                f"semantic representation has {feat.shape[-1]} frames, exceeding "
                f"max_frames={self.config.max_frames}."
            )
        return feat.to(dtype=torch.float32).contiguous()

    def _encode_core(
        self,
        audio: torch.Tensor,
        semantic_repr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio = audio.contiguous()
        semantic_repr = semantic_repr.contiguous()
        batch = int(audio.shape[0])
        seq_len = int(semantic_repr.shape[-1])
        semantic_codes = torch.empty(
            (batch, 1, seq_len),
            dtype=torch.int64,
            device=self.device,
        )
        acoustic_codes = torch.empty(
            (batch, DEFAULT_ACOUSTIC_CODEBOOKS, seq_len),
            dtype=torch.int64,
            device=self.device,
        )
        caller_stream = torch.cuda.current_stream(device=self.device)
        with self._io_binding_lock:
            with torch.cuda.stream(self.compute_stream):
                self.io_binding.clear_binding_inputs()
                self.io_binding.clear_binding_outputs()
                self.io_binding.bind_input(
                    name="audio",
                    device_type="cuda",
                    device_id=self.device_id,
                    element_type=np.float32,
                    shape=tuple(audio.shape),
                    buffer_ptr=audio.data_ptr(),
                )
                self.io_binding.bind_input(
                    name="semantic_repr",
                    device_type="cuda",
                    device_id=self.device_id,
                    element_type=np.float32,
                    shape=tuple(semantic_repr.shape),
                    buffer_ptr=semantic_repr.data_ptr(),
                )
                self.io_binding.bind_output(
                    name="semantic_codes",
                    device_type="cuda",
                    device_id=self.device_id,
                    element_type=np.int64,
                    shape=tuple(semantic_codes.shape),
                    buffer_ptr=semantic_codes.data_ptr(),
                )
                self.io_binding.bind_output(
                    name="acoustic_codes",
                    device_type="cuda",
                    device_id=self.device_id,
                    element_type=np.int64,
                    shape=tuple(acoustic_codes.shape),
                    buffer_ptr=acoustic_codes.data_ptr(),
                )
                if hasattr(self.io_binding, "synchronize_inputs"):
                    self.io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self.io_binding)
                if hasattr(self.io_binding, "synchronize_outputs"):
                    self.io_binding.synchronize_outputs()
                else:
                    self.compute_stream.synchronize()
            caller_stream.wait_stream(self.compute_stream)
        return semantic_codes, acoustic_codes


class DualCodecTorchEncoder:
    """DualCodec encoder using W2V semantic extraction plus the torch encode core."""

    def __init__(
        self,
        w2v_path: str | Path | None = None,
        dualcodec_path: str | Path | None = None,
        model_id: str = "25hz_v1",
        device_id: int = 0,
        dtype: str | torch.dtype = "bfloat16",
        max_samples: int = DEFAULT_SAMPLE_RATE * 30,
        max_frames: int = DEFAULT_FRAME_RATE * 30,
        semantic_downsample_factor: int = DEFAULT_SEMANTIC_DOWNSAMPLE_FACTOR,
    ) -> None:
        _clear_proxies()
        if _get_dualcodec_model is None:
            raise RuntimeError("dualcodec torch model loader is unavailable.")
        if not torch.cuda.is_available():
            raise RuntimeError("DualCodec torch encoder requires CUDA.")
        self.w2v_path = Path(w2v_path) if w2v_path is not None else DEFAULT_W2V_PATH
        self.dualcodec_path = Path(dualcodec_path) if dualcodec_path is not None else DEFAULT_DUALCODEC_PATH
        self.model_id = model_id
        self.device_id = int(device_id)
        self.device = torch.device(f"cuda:{self.device_id}")
        self.dtype = _torch_dtype(dtype)
        self.max_samples = int(max_samples)
        self.max_frames = int(max_frames)
        self.semantic_downsample_factor = int(semantic_downsample_factor)
        self._resolve_paths()
        self.semantic_extractor = W2VSemanticExtractor(
            dualcodec_path=self.dualcodec_path,
            w2v_path=self.w2v_path,
            device=self.device,
        )
        self.model = _get_dualcodec_model(self.model_id, str(self.dualcodec_path))
        self.model.to(self.device)
        self.model.eval()
        self.active_providers = [f"torch:{self.device}:{str(self.dtype).replace('torch.', '')}"]

    def __call__(self, audio_numpy: np.ndarray) -> np.ndarray:
        return self.encode(audio_numpy)

    def encode(self, audio_numpy: np.ndarray) -> np.ndarray:
        """Encode ``float32`` 24k audio shaped ``(1, 1, T)`` into ``int32 [12, t]``."""
        audio = self._normalize_audio(audio_numpy)
        with torch.inference_mode():
            semantic_repr = self._extract_semantic_repr(audio)
            with self._torch_context():
                semantic_codes, acoustic_codes = self.model.encode(
                    audio,
                    num_quantizers=DEFAULT_NUM_CODEBOOKS,
                    sample_rate=DEFAULT_SAMPLE_RATE,
                    semantic_repr=semantic_repr,
                )
            tokens = torch.cat([semantic_codes, acoustic_codes], dim=1)
        return tokens.squeeze(0).to(torch.int32).cpu().numpy()

    def _resolve_paths(self) -> None:
        self.w2v_path = self.w2v_path.expanduser().resolve()
        self.dualcodec_path = self.dualcodec_path.expanduser().resolve()
        model_file = {
            "12hz_v1": "dualcodec_12hz_16384_4096.safetensors",
            "25hz_v1": "dualcodec_25hz_16384_1024.safetensors",
        }.get(self.model_id, f"{self.model_id}.safetensors")
        required = [
            self.w2v_path / "config.json",
            self.w2v_path / "model.safetensors",
            self.w2v_path / "preprocessor_config.json",
            self.dualcodec_path / "w2vbert2_mean_var_stats_emilia.pt",
            self.dualcodec_path / model_file,
        ]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Required DualCodec torch encoder asset not found: {path}")
        if self.max_samples <= 0 or self.max_frames <= 0:
            raise ValueError(f"max_samples/max_frames should be positive, got {self.max_samples}/{self.max_frames}.")
        if self.semantic_downsample_factor <= 0:
            raise ValueError(
                "semantic_downsample_factor should be positive, got "
                f"{self.semantic_downsample_factor}."
            )

    def _normalize_audio(self, audio_numpy: np.ndarray) -> torch.Tensor:
        if not isinstance(audio_numpy, np.ndarray):
            raise TypeError(f"audio_numpy should be np.ndarray, got {type(audio_numpy)!r}.")
        if audio_numpy.shape[0:2] != (1, 1) or audio_numpy.ndim != 3:
            raise ValueError(
                "DualCodec encoder expects audio_numpy shape (1, 1, T), "
                f"got {audio_numpy.shape}."
            )
        if audio_numpy.dtype != np.float32:
            raise ValueError(f"DualCodec encoder expects float32 audio, got {audio_numpy.dtype}.")
        if audio_numpy.shape[-1] > self.max_samples:
            raise ValueError(
                f"audio has {audio_numpy.shape[-1]} samples, exceeding "
                f"max_samples={self.max_samples}."
            )
        return torch.from_numpy(np.ascontiguousarray(audio_numpy)).to(self.device, dtype=torch.float32)

    def _torch_context(self):
        if self.device.type == "cuda" and self.dtype in {torch.bfloat16, torch.float16}:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    def _extract_semantic_repr(self, audio: torch.Tensor) -> torch.Tensor:
        audio_16k = torchaudio.functional.resample(
            audio,
            DEFAULT_SAMPLE_RATE,
            DEFAULT_W2V_SAMPLE_RATE,
        )
        feature_extractor = self.semantic_extractor.semantic_cfg["feature_extractor"]
        inputs = feature_extractor(
            audio_16k.cpu(),
            sampling_rate=DEFAULT_W2V_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs["input_features"][0].unsqueeze(0).to(self.device)
        attention_mask = inputs["attention_mask"][0].unsqueeze(0).to(self.device)
        with self._torch_context():
            feat = self.semantic_extractor.extract(
                input_features,
                attention_mask,
            ).transpose(1, 2)
            feat = F.avg_pool1d(
                feat,
                self.semantic_downsample_factor,
                self.semantic_downsample_factor,
            )
        if feat.shape[-1] > self.max_frames:
            raise ValueError(
                f"semantic representation has {feat.shape[-1]} frames, exceeding "
                f"max_frames={self.max_frames}."
            )
        return feat.contiguous()


def _torch_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    text = str(value or "bf16").strip().lower()
    if text in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16", "half", "torch.float16"}:
        return torch.float16
    if text in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype {value!r}.")
