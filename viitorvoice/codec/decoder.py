from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from viitorvoice import paths


DEFAULT_ONNX_PATH = paths.dualcodec_decoder_onnx_path()
DEFAULT_SILENCE_CODEC_PATH = paths.silence_codec_path()
DEFAULT_TRT_CACHE_ROOT = paths.dualcodec_decoder_trt_cache_root()
DEFAULT_NUM_CODEBOOKS = 12
DEFAULT_ACOUSTIC_CODEBOOKS = DEFAULT_NUM_CODEBOOKS - 1
DEFAULT_HOP_LENGTH = 960
DEFAULT_SILENCE_FRAMES = 25
DEFAULT_TAIL_TRIM_SAMPLES = 4


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


def _np_type(dtype: torch.dtype) -> type[np.generic]:
    if dtype == torch.int64:
        return np.int64
    if dtype == torch.float32:
        return np.float32
    raise ValueError(f"Unsupported tensor dtype for ORT IOBinding: {dtype}.")


@dataclass
class DualCodecDecoderConfig:
    """Configuration for packed DualCodec ONNX decode."""

    onnx_path: Path = DEFAULT_ONNX_PATH
    ep_config: Mapping[str, Any] = field(
        default_factory=lambda: {
            "backend": "trt",
            "device_id": 0,
            "trt_fp16_enable": True,
        }
    )
    max_frames: int = 2048
    silence_frames: int = DEFAULT_SILENCE_FRAMES
    silence_codec_path: Path = DEFAULT_SILENCE_CODEC_PATH
    trt_cache_root: Path = DEFAULT_TRT_CACHE_ROOT
    hop_length: int = DEFAULT_HOP_LENGTH
    tail_trim_samples: int = DEFAULT_TAIL_TRIM_SAMPLES


class DualCodecPackedDecoder:
    """DualCodec decoder with padding-free packed inference.

    The decoder receives a list of token tensors shaped ``[12, T]``. It packs
    them along the time axis with silence-token context at the beginning, end,
    and every inter-segment boundary, then decodes each packed chunk with
    ONNX Runtime and trims the real segments back out.
    """

    def __init__(
        self,
        onnx_path: str | Path | None = None,
        ep_config: Mapping[str, Any] | None = None,
        max_frames: int = 2048,
        silence_frames: int = DEFAULT_SILENCE_FRAMES,
        silence_codec_path: str | Path | None = None,
        trt_cache_root: str | Path | None = None,
        hop_length: int = DEFAULT_HOP_LENGTH,
        tail_trim_samples: int = DEFAULT_TAIL_TRIM_SAMPLES,
    ) -> None:
        _clear_proxies()
        config = DualCodecDecoderConfig(
            onnx_path=Path(onnx_path) if onnx_path is not None else DEFAULT_ONNX_PATH,
            ep_config=dict(ep_config or {"backend": "trt", "device_id": 0}),
            max_frames=int(max_frames),
            silence_frames=int(silence_frames),
            silence_codec_path=(
                Path(silence_codec_path)
                if silence_codec_path is not None
                else DEFAULT_SILENCE_CODEC_PATH
            ),
            trt_cache_root=(
                Path(trt_cache_root)
                if trt_cache_root is not None
                else DEFAULT_TRT_CACHE_ROOT
            ),
            hop_length=int(hop_length),
            tail_trim_samples=int(tail_trim_samples),
        )
        self.config = self._resolve_config(config)
        self.backend = str(self.config.ep_config.get("backend", "trt")).lower()
        if self.backend in {"ort_trt", "onnx_trt", "tensorrt"}:
            self.backend = "trt"
        if self.backend in {"ort_cuda", "onnx_cuda"}:
            self.backend = "cuda"
        if self.backend not in {"trt", "cuda"}:
            raise ValueError(
                f"Unsupported DualCodec decoder backend {self.backend!r}; "
                "expected 'trt' or 'cuda'."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("DualCodec packed decoder requires CUDA.")

        self.device_id = int(self.config.ep_config.get("device_id", 0))
        self.device = torch.device(f"cuda:{self.device_id}")
        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.compute_stream_ptr = str(int(self.compute_stream.cuda_stream))
        self._io_binding_lock = threading.Lock()
        self.silence_tokens = self._load_silence_tokens(self.config.silence_codec_path)
        self.session = self._create_session()
        self.active_providers = list(self.session.get_providers())
        self.io_binding = self.session.io_binding()

    @classmethod
    def from_config(
        cls,
        config: DualCodecDecoderConfig,
    ) -> "DualCodecPackedDecoder":
        return cls(
            onnx_path=config.onnx_path,
            ep_config=config.ep_config,
            max_frames=config.max_frames,
            silence_frames=config.silence_frames,
            silence_codec_path=config.silence_codec_path,
            trt_cache_root=config.trt_cache_root,
            hop_length=config.hop_length,
            tail_trim_samples=config.tail_trim_samples,
        )

    def __call__(
        self,
        tokens_list: Sequence[np.ndarray | torch.Tensor],
    ) -> list[np.ndarray]:
        return self.decode(tokens_list)

    def decode(
        self,
        tokens_list: Sequence[np.ndarray | torch.Tensor],
    ) -> list[np.ndarray]:
        """Decode a list of ``[12, T]`` tokens into ``[1, 1, samples]`` arrays."""
        if not tokens_list:
            return []

        tokens = [self._normalize_tokens(t, i) for i, t in enumerate(tokens_list)]
        outputs: list[np.ndarray | None] = [None] * len(tokens)
        for group in self._build_groups(tokens):
            packed, spans = self._pack_group(tokens, group)
            audio = self._decode_packed(packed)
            for output_idx, start_frame, token_frames in spans:
                start = start_frame * self.config.hop_length
                length = token_frames * self.config.hop_length
                if self.config.tail_trim_samples > 0:
                    length = max(0, length - self.config.tail_trim_samples)
                outputs[output_idx] = audio[:, :, start : start + length].copy()

        return [self._require_output(out, i) for i, out in enumerate(outputs)]

    def _resolve_config(self, config: DualCodecDecoderConfig) -> DualCodecDecoderConfig:
        config.onnx_path = config.onnx_path.expanduser().resolve()
        config.silence_codec_path = config.silence_codec_path.expanduser().resolve()
        config.trt_cache_root = config.trt_cache_root.expanduser().resolve()
        if not config.onnx_path.is_file():
            raise FileNotFoundError(f"DualCodec decoder ONNX not found: {config.onnx_path}")
        if not config.silence_codec_path.is_file():
            raise FileNotFoundError(
                f"DualCodec silence codec not found: {config.silence_codec_path}"
            )
        if config.max_frames <= 0:
            raise ValueError(f"max_frames should be positive, got {config.max_frames}.")
        if config.silence_frames < 0:
            raise ValueError(
                f"silence_frames should be non-negative, got {config.silence_frames}."
            )
        if config.hop_length <= 0:
            raise ValueError(f"hop_length should be positive, got {config.hop_length}.")
        if config.tail_trim_samples < 0:
            raise ValueError(
                f"tail_trim_samples should be non-negative, got {config.tail_trim_samples}."
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
            min_frames = 1
            profile_min = (
                f"semantic_codes:1x1x{min_frames},"
                f"acoustic_codes:1x{DEFAULT_ACOUSTIC_CODEBOOKS}x{min_frames}"
            )
            profile_opt = (
                f"semantic_codes:1x1x{self.config.max_frames},"
                f"acoustic_codes:1x{DEFAULT_ACOUSTIC_CODEBOOKS}x{self.config.max_frames}"
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
                f"Neither TRT nor CUDA EP is active for DualCodec decoder: {active}"
            )
        return session

    def _trt_cache_dir(self) -> Path:
        st = os.stat(self.config.onnx_path)
        cache_key = hashlib.md5(
            (
                str(self.config.onnx_path)
                + f"|sig{st.st_size}:{st.st_mtime_ns}"
                + f"|packed|max{self.config.max_frames}"
                + f"|silence{self.config.silence_frames}"
                + f"|fp16{bool(self.config.ep_config.get('trt_fp16_enable', True))}"
            ).encode("utf-8")
        ).hexdigest()
        return self.config.trt_cache_root / f"trt_cache_{cache_key}"

    def _load_silence_tokens(self, path: Path) -> torch.Tensor:
        payload = torch.load(path, map_location="cpu")
        tokens = payload["tokens"] if isinstance(payload, Mapping) else payload
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        if tokens.ndim == 3:
            if tokens.shape[0] != 1:
                raise ValueError(
                    f"Silence codec batch size should be 1, got {tuple(tokens.shape)}."
                )
            tokens = tokens.squeeze(0)
        if tokens.ndim != 2 or tokens.shape[0] != DEFAULT_NUM_CODEBOOKS:
            raise ValueError(
                "Silence codec should have shape "
                f"[{DEFAULT_NUM_CODEBOOKS}, T], got {tuple(tokens.shape)}."
            )
        return tokens.contiguous()

    def _silence_gap(self) -> torch.Tensor | None:
        frames = self.config.silence_frames
        if frames == 0:
            return None
        repeat = (
            frames + self.silence_tokens.shape[-1] - 1
        ) // self.silence_tokens.shape[-1]
        return self.silence_tokens.repeat(1, repeat)[:, :frames].contiguous()

    def _normalize_tokens(
        self,
        tokens: np.ndarray | torch.Tensor,
        index: int,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(tokens, dtype=torch.long, device="cpu")
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2:
            raise ValueError(
                f"tokens_list[{index}] should have shape [12, T], got {tuple(tensor.shape)}."
            )
        if tensor.shape[0] != DEFAULT_NUM_CODEBOOKS:
            raise ValueError(
                f"tokens_list[{index}] should have {DEFAULT_NUM_CODEBOOKS} codebooks, "
                f"got {tensor.shape[0]}."
            )
        if tensor.shape[1] <= 0:
            raise ValueError(f"tokens_list[{index}] has empty time dimension.")
        return tensor.contiguous()

    def _packed_frame_count(self, token_frames: Sequence[int]) -> int:
        total = int(sum(token_frames))
        if self.config.silence_frames > 0:
            total += self.config.silence_frames * (len(token_frames) + 1)
        return total

    def _build_groups(self, tokens: Sequence[torch.Tensor]) -> list[list[int]]:
        groups: list[list[int]] = []
        current: list[int] = []
        current_lengths: list[int] = []
        for idx, token in enumerate(tokens):
            token_frames = int(token.shape[-1])
            single_frames = self._packed_frame_count([token_frames])
            if single_frames > self.config.max_frames:
                raise ValueError(
                    f"tokens_list[{idx}] needs {single_frames} packed frames, "
                    f"which exceeds max_frames={self.config.max_frames}."
                )
            candidate_lengths = current_lengths + [token_frames]
            if (
                current
                and self._packed_frame_count(candidate_lengths) > self.config.max_frames
            ):
                groups.append(current)
                current = [idx]
                current_lengths = [token_frames]
            else:
                current.append(idx)
                current_lengths = candidate_lengths
        if current:
            groups.append(current)
        return groups

    def _pack_group(
        self,
        tokens: Sequence[torch.Tensor],
        group: Sequence[int],
    ) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        gap = self._silence_gap()
        parts: list[torch.Tensor] = []
        spans: list[tuple[int, int, int]] = []
        frame_cursor = 0
        if gap is not None:
            parts.append(gap)
            frame_cursor += int(gap.shape[-1])
        for pos, idx in enumerate(group):
            token = tokens[idx]
            token_frames = int(token.shape[-1])
            spans.append((idx, frame_cursor, token_frames))
            parts.append(token)
            frame_cursor += token_frames
            if gap is not None:
                parts.append(gap)
                frame_cursor += int(gap.shape[-1])
            elif pos + 1 < len(group):
                continue
        packed = torch.cat(parts, dim=-1).unsqueeze(0)
        if packed.shape[-1] > self.config.max_frames:
            raise RuntimeError(
                f"Packed group has {packed.shape[-1]} frames, exceeding "
                f"max_frames={self.config.max_frames}."
            )
        return packed, spans

    def _decode_packed(self, packed_tokens: torch.Tensor) -> np.ndarray:
        semantic_codes = packed_tokens[:, :1, :].to(self.device, non_blocking=True)
        acoustic_codes = packed_tokens[:, 1:, :].to(self.device, non_blocking=True)
        semantic_codes = semantic_codes.to(dtype=torch.int64).contiguous()
        acoustic_codes = acoustic_codes.to(dtype=torch.int64).contiguous()
        caller_stream = torch.cuda.current_stream(device=self.device)
        with self._io_binding_lock:
            with torch.cuda.stream(self.compute_stream):
                self.io_binding.clear_binding_inputs()
                self.io_binding.clear_binding_outputs()
                self.io_binding.bind_input(
                    name="semantic_codes",
                    device_type="cuda",
                    device_id=self.device_id,
                    element_type=_np_type(semantic_codes.dtype),
                    shape=tuple(semantic_codes.shape),
                    buffer_ptr=semantic_codes.data_ptr(),
                )
                self.io_binding.bind_input(
                    name="acoustic_codes",
                    device_type="cuda",
                    device_id=self.device_id,
                    element_type=_np_type(acoustic_codes.dtype),
                    shape=tuple(acoustic_codes.shape),
                    buffer_ptr=acoustic_codes.data_ptr(),
                )
                self.io_binding.bind_output(
                    name="audio",
                    device_type="cuda",
                    device_id=self.device_id,
                    element_type=np.float32,
                )
                if hasattr(self.io_binding, "synchronize_inputs"):
                    self.io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self.io_binding)
                if hasattr(self.io_binding, "synchronize_outputs"):
                    self.io_binding.synchronize_outputs()
                else:
                    self.compute_stream.synchronize()
                audio = self.io_binding.copy_outputs_to_cpu()[0]
            caller_stream.wait_stream(self.compute_stream)
        return audio.astype(np.float32, copy=False)

    @staticmethod
    def _require_output(output: np.ndarray | None, index: int) -> np.ndarray:
        if output is None:
            raise RuntimeError(f"Missing decoded output for tokens_list[{index}].")
        return output
