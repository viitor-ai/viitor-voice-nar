from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from onnx import TensorProto

from viitorvoice import paths


DEFAULT_CHECKPOINT_DIR = paths.llm_checkpoint_dir()
DEFAULT_ONNX_PATH = paths.llm_onnx_path(DEFAULT_CHECKPOINT_DIR)
DEFAULT_TRT_CACHE_ROOT = paths.llm_trt_cache_root(DEFAULT_CHECKPOINT_DIR)


def clear_proxies() -> None:
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


def normalize_backend(backend: str) -> str:
    value = (backend or "trt").strip().lower().replace("-", "_")
    if value in {"ort_trt", "onnx_trt", "tensorrt", "trt"}:
        return "trt"
    if value in {"ort_cuda", "onnx_cuda", "cuda"}:
        return "cuda"
    raise ValueError(f"Unsupported LLM backend {backend!r}; expected onnx-cuda or onnx-trt.")


def torch_dtype_from_precision(precision: str) -> torch.dtype:
    value = (precision or "fp16").strip().lower()
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported precision {precision!r}; expected fp16, bf16, or fp32.")


def _np_type(dtype: torch.dtype) -> type[np.generic]:
    if dtype == torch.float16:
        return np.float16
    if dtype == torch.float32:
        return np.float32
    if dtype == torch.bfloat16:
        return TensorProto.BFLOAT16
    if dtype == torch.bool:
        return np.bool_
    raise ValueError(f"Unsupported tensor dtype for ORT IOBinding: {dtype}.")


def _torch_dtype_from_ort_type(ort_type: str) -> torch.dtype:
    if ort_type == "tensor(float16)":
        return torch.float16
    if ort_type == "tensor(float)":
        return torch.float32
    if ort_type == "tensor(bfloat16)":
        return torch.bfloat16
    if ort_type == "tensor(bool)":
        return torch.bool
    raise ValueError(f"Unsupported ONNX tensor type for ORT IOBinding: {ort_type}.")


@dataclass
class LLMOnnxConfig:
    onnx_path: Path = DEFAULT_ONNX_PATH
    backend: str = "trt"
    precision: str = "fp16"
    device_id: int = 0
    hidden_size: int = 1024
    trt_cache_root: Path = DEFAULT_TRT_CACHE_ROOT
    batch_min: int = 1
    batch_opt: int = 2
    batch_max: int = 4
    seq_min: int = 16
    seq_opt: int = 512
    seq_max: int = 2048
    ep_options: Mapping[str, Any] = field(default_factory=dict)
    strict_trt: bool = False


class LLMOnnxStepRunner:
    """Run one full LLM backbone step with ONNX Runtime CUDA/TRT providers."""

    def __init__(self, config: LLMOnnxConfig) -> None:
        clear_proxies()
        self.config = self._resolve_config(config)
        self.backend = normalize_backend(self.config.backend)
        self.precision = self.config.precision.strip().lower()
        if self.precision in {"float16", "half"}:
            self.precision = "fp16"
        if self.precision in {"bfloat16"}:
            self.precision = "bf16"
        if self.precision in {"float32"}:
            self.precision = "fp32"
        if self.precision not in {"fp16", "bf16", "fp32"}:
            raise ValueError(f"Unsupported precision {self.config.precision!r}.")
        if not torch.cuda.is_available():
            raise RuntimeError("LLM ONNX runtime requires CUDA.")

        self.device_id = int(self.config.device_id)
        self.device = torch.device(f"cuda:{self.device_id}")
        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.compute_stream_ptr = str(int(self.compute_stream.cuda_stream))
        self._io_binding_lock = threading.Lock()
        self.session = self._create_session()
        self.active_providers = list(self.session.get_providers())
        self.input_dtypes = {
            item.name: _torch_dtype_from_ort_type(item.type)
            for item in self.session.get_inputs()
        }
        self.output_dtypes = {
            item.name: _torch_dtype_from_ort_type(item.type)
            for item in self.session.get_outputs()
        }
        self.io_binding = self.session.io_binding()

    def run_step(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if inputs_embeds.device.type != "cuda":
            raise ValueError("inputs_embeds must be a CUDA tensor.")
        if inputs_embeds.device.index not in {None, self.device_id}:
            raise ValueError(
                f"inputs_embeds is on {inputs_embeds.device}, expected cuda:{self.device_id}."
            )
        if attention_mask.device.type != "cuda":
            attention_mask = attention_mask.to(inputs_embeds.device, non_blocking=True)
        input_dtype = self.input_dtypes.get("inputs_embeds", inputs_embeds.dtype)
        mask_dtype = self.input_dtypes.get("attention_mask", input_dtype)
        output_dtype = self.output_dtypes.get("hidden_states", input_dtype)
        if inputs_embeds.dtype != input_dtype:
            inputs_embeds = inputs_embeds.to(dtype=input_dtype)
        if attention_mask.dtype == torch.bool:
            mask_value = (
                torch.finfo(mask_dtype).min
                if self.backend == "trt" and mask_dtype in {torch.float16, torch.bfloat16, torch.float32}
                else torch.finfo(mask_dtype).min
            )
            additive_mask = torch.zeros(
                attention_mask.shape,
                device=attention_mask.device,
                dtype=mask_dtype,
            )
            attention_mask = additive_mask.masked_fill(
                ~attention_mask,
                mask_value,
            )
        elif attention_mask.dtype != mask_dtype:
            attention_mask = attention_mask.to(dtype=mask_dtype)

        caller_stream = torch.cuda.current_stream(device=self.device)
        with self._io_binding_lock:
            with torch.cuda.stream(self.compute_stream):
                self.io_binding.clear_binding_inputs()
                self.io_binding.clear_binding_outputs()

                inputs_embeds = inputs_embeds.contiguous()
                attention_mask = attention_mask.contiguous()
                hidden_states = torch.empty(
                    inputs_embeds.shape,
                    device=inputs_embeds.device,
                    dtype=output_dtype,
                )

                self.io_binding.bind_input(
                    name="inputs_embeds",
                    device_type="cuda",
                    device_id=self.device_id,
                    element_type=_np_type(inputs_embeds.dtype),
                    shape=tuple(inputs_embeds.shape),
                    buffer_ptr=inputs_embeds.data_ptr(),
                )
                self.io_binding.bind_input(
                    name="attention_mask",
                    device_type="cuda",
                    device_id=self.device_id,
                    element_type=_np_type(attention_mask.dtype),
                    shape=tuple(attention_mask.shape),
                    buffer_ptr=attention_mask.data_ptr(),
                )
                self.io_binding.bind_output(
                    name="hidden_states",
                    device_type="cuda",
                    device_id=self.device_id,
                    element_type=_np_type(hidden_states.dtype),
                    shape=tuple(hidden_states.shape),
                    buffer_ptr=hidden_states.data_ptr(),
                )

                if hasattr(self.io_binding, "synchronize_inputs"):
                    self.io_binding.synchronize_inputs()
                self.session.run_with_iobinding(self.io_binding)
                if hasattr(self.io_binding, "synchronize_outputs"):
                    self.io_binding.synchronize_outputs()
                else:
                    self.compute_stream.synchronize()
            caller_stream.wait_stream(self.compute_stream)
        return hidden_states

    __call__ = run_step

    def _resolve_config(self, config: LLMOnnxConfig) -> LLMOnnxConfig:
        config.onnx_path = config.onnx_path.expanduser().resolve()
        config.trt_cache_root = config.trt_cache_root.expanduser().resolve()
        if not config.onnx_path.is_file():
            raise FileNotFoundError(f"LLM ONNX model not found: {config.onnx_path}")
        if config.hidden_size <= 0:
            raise ValueError(f"hidden_size should be positive, got {config.hidden_size}.")
        if config.batch_min <= 0 or config.batch_opt <= 0 or config.batch_max <= 0:
            raise ValueError("batch min/opt/max should all be positive.")
        if config.seq_min <= 0 or config.seq_opt <= 0 or config.seq_max <= 0:
            raise ValueError("seq min/opt/max should all be positive.")
        if not (config.batch_min <= config.batch_opt <= config.batch_max):
            raise ValueError("Expected batch_min <= batch_opt <= batch_max.")
        if not (config.seq_min <= config.seq_opt <= config.seq_max):
            raise ValueError("Expected seq_min <= seq_opt <= seq_max.")
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
                f"inputs_embeds:{self.config.batch_min}x{self.config.seq_min}x{self.config.hidden_size},"
                f"attention_mask:{self.config.batch_min}x1x{self.config.seq_min}x{self.config.seq_min}"
            )
            profile_opt = (
                f"inputs_embeds:{self.config.batch_opt}x{self.config.seq_opt}x{self.config.hidden_size},"
                f"attention_mask:{self.config.batch_opt}x1x{self.config.seq_opt}x{self.config.seq_opt}"
            )
            profile_max = (
                f"inputs_embeds:{self.config.batch_max}x{self.config.seq_max}x{self.config.hidden_size},"
                f"attention_mask:{self.config.batch_max}x1x{self.config.seq_max}x{self.config.seq_max}"
            )
            trt_options = {
                "device_id": self.device_id,
                "user_compute_stream": self.compute_stream_ptr,
                "trt_fp16_enable": self.precision == "fp16",
                "trt_bf16_enable": self.precision == "bf16",
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_dir),
                "trt_profile_min_shapes": profile_min,
                "trt_profile_opt_shapes": profile_opt,
                "trt_profile_max_shapes": profile_max,
                "trt_layer_norm_fp32_fallback": bool(
                    self.config.ep_options.get("trt_layer_norm_fp32_fallback", True)
                ),
            }
            trt_options.update(dict(self.config.ep_options))
            providers = [("TensorrtExecutionProvider", trt_options)] + cuda_providers

        session = ort.InferenceSession(str(self.config.onnx_path), providers=providers)
        active = session.get_providers()
        if self.backend == "trt" and "TensorrtExecutionProvider" not in active:
            if self.config.strict_trt:
                raise RuntimeError(f"TensorRT EP is not active for LLM ONNX: {active}")
            self.backend = "cuda"
            session = ort.InferenceSession(str(self.config.onnx_path), providers=cuda_providers)
            active = session.get_providers()
        if (
            "CUDAExecutionProvider" not in active
            and "TensorrtExecutionProvider" not in active
        ):
            raise RuntimeError(f"Neither TRT nor CUDA EP is active for LLM ONNX: {active}")
        return session

    def _trt_cache_dir(self) -> Path:
        st = os.stat(self.config.onnx_path)
        cache_key = hashlib.md5(
            (
                str(self.config.onnx_path)
                + f"|sig{st.st_size}:{st.st_mtime_ns}"
                + f"|h{self.config.hidden_size}"
                + f"|b{self.config.batch_min}-{self.config.batch_opt}-{self.config.batch_max}"
                + f"|s{self.config.seq_min}-{self.config.seq_opt}-{self.config.seq_max}"
                + f"|precision{self.precision}"
                + f"|ep_options{dict(self.config.ep_options)}"
                + "|additive_mask_v1"
            ).encode("utf-8")
        ).hexdigest()
        return self.config.trt_cache_root / f"trt_cache_{cache_key}"
