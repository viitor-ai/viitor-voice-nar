from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from viitorvoice import paths


PROJECT_ROOT = paths.repo_root()
INFERENCE_ROOT = PROJECT_ROOT
LOCAL_MODELS_ROOT = paths.local_models_root()
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_FRAME_RATE = 25
DEFAULT_HOP_LENGTH = DEFAULT_SAMPLE_RATE // DEFAULT_FRAME_RATE
DEFAULT_LLM_MODEL_NAME = paths.DEFAULT_LLM_MODEL_NAME


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


def _env_value(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value not in {None, ""}:
        return value
    return default


def _env_path(name: str, default: Path) -> Path:
    value = _env_value(name)
    return Path(value).expanduser() if value else default


def _env_int(name: str, default: int) -> int:
    value = _env_value(name)
    return default if value in {None, ""} else int(value)


def _env_float(name: str, default: float) -> float:
    value = _env_value(name)
    return default if value in {None, ""} else float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = _env_value(name)
    if value in {None, ""}:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class LLMRuntimeConfig:
    checkpoint_dir: Path = paths.llm_checkpoint_dir(DEFAULT_LLM_MODEL_NAME)
    onnx_path: Path = paths.llm_onnx_path(checkpoint_dir)
    backend: str = "onnx-trt"
    precision: str = "fp32"
    trt_cache_root: Path = paths.llm_trt_cache_root(checkpoint_dir)
    batch_min: int = 1
    batch_opt: int = 2
    batch_max: int = 4
    seq_min: int = 16
    seq_opt: int = 512
    seq_max: int = 2048
    strict_trt: bool = False


@dataclass(frozen=True)
class EncoderRuntimeConfig:
    onnx_path: Path = paths.dualcodec_encoder_onnx_path()
    w2v_path: Path = paths.dualcodec_w2v_path()
    dualcodec_path: Path = paths.dualcodec_ckpt_dir()
    model_id: str = "25hz_v1"
    backend: str = "torch"
    precision: str = "bf16"
    trt_cache_root: Path = paths.dualcodec_encoder_trt_cache_root()
    max_seconds: int = 30


@dataclass(frozen=True)
class DecoderRuntimeConfig:
    onnx_path: Path = paths.dualcodec_decoder_onnx_path()
    backend: str = "onnx-trt"
    precision: str = "fp16"
    trt_cache_root: Path = paths.dualcodec_decoder_trt_cache_root()
    silence_codec_path: Path = paths.silence_codec_path()
    max_frames: int = 2048
    silence_frames: int = 25


@dataclass(frozen=True)
class AlignerRuntimeConfig:
    model_path: Path = paths.aligner_model_path()
    device: str = "cuda:0"
    language: str = "zh"
    dtype: str = "auto"
    attn_implementation: str = ""


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 50051
    device_id: int = 0
    max_queue_size: int = 256
    request_timeout_sec: float = 120.0
    warmup_on_start: bool = True
    debug_dump_dir: Path | None = None
    llm: LLMRuntimeConfig = LLMRuntimeConfig()
    encoder: EncoderRuntimeConfig = EncoderRuntimeConfig()
    decoder: DecoderRuntimeConfig = DecoderRuntimeConfig()
    aligner: AlignerRuntimeConfig = AlignerRuntimeConfig()

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        local_models = _env_path("VIITORVOICE_LOCAL_MODELS", LOCAL_MODELS_ROOT)
        checkpoint_dir = _env_path("VIITORVOICE_LLM_CHECKPOINT", local_models / "llm" / DEFAULT_LLM_MODEL_NAME)
        device_id = _env_int("VIITORVOICE_DEVICE_ID", 0)
        llm = LLMRuntimeConfig(
            checkpoint_dir=checkpoint_dir,
            onnx_path=_env_path(
                "VIITORVOICE_LLM_ONNX",
                checkpoint_dir / ".cache" / "onnx_backbone_fp32" / "llm_backbone_dynamic.onnx",
            ),
            backend=_env_value("VIITORVOICE_LLM_BACKEND", "onnx-trt") or "onnx-trt",
            precision=_env_value("VIITORVOICE_LLM_PRECISION", "fp32") or "fp32",
            trt_cache_root=_env_path("VIITORVOICE_LLM_TRT_CACHE", checkpoint_dir / ".cache" / "trt_cache"),
            batch_min=_env_int("VIITORVOICE_LLM_BATCH_MIN", 1),
            batch_opt=_env_int("VIITORVOICE_LLM_BATCH_OPT", 2),
            batch_max=_env_int("VIITORVOICE_LLM_BATCH_MAX", 4),
            seq_min=_env_int("VIITORVOICE_LLM_SEQ_MIN", 16),
            seq_opt=_env_int("VIITORVOICE_LLM_SEQ_OPT", 512),
            seq_max=_env_int("VIITORVOICE_LLM_SEQ_MAX", 2048),
            strict_trt=_env_bool("VIITORVOICE_LLM_STRICT_TRT", False),
        )
        dualcodec_ckpt = local_models / "dualcodec" / "dualcodec_ckpts"
        encoder = EncoderRuntimeConfig(
            onnx_path=_env_path(
                "VIITORVOICE_CODEC_ENCODER_ONNX",
                dualcodec_ckpt / "dualcodec_encode_core_30s.onnx",
            ),
            w2v_path=_env_path(
                "VIITORVOICE_CODEC_W2V",
                local_models / "dualcodec" / "w2v-bert-2.0",
            ),
            dualcodec_path=_env_path("VIITORVOICE_CODEC_DUALCODEC_PATH", dualcodec_ckpt),
            model_id=_env_value("VIITORVOICE_CODEC_ENCODER_MODEL_ID", "25hz_v1") or "25hz_v1",
            backend=_env_value("VIITORVOICE_CODEC_ENCODER_BACKEND", "torch") or "torch",
            precision=_env_value("VIITORVOICE_CODEC_ENCODER_PRECISION", "bf16") or "bf16",
            trt_cache_root=_env_path(
                "VIITORVOICE_CODEC_ENCODER_TRT_CACHE",
                dualcodec_ckpt / "trt_cache" / "packed_encode",
            ),
            max_seconds=_env_int("VIITORVOICE_CODEC_ENCODER_MAX_SECONDS", 30),
        )
        decoder = DecoderRuntimeConfig(
            onnx_path=_env_path("VIITORVOICE_CODEC_DECODER_ONNX", dualcodec_ckpt / "dualcodec_decoder.onnx"),
            backend=_env_value("VIITORVOICE_CODEC_DECODER_BACKEND", "onnx-trt") or "onnx-trt",
            trt_cache_root=_env_path(
                "VIITORVOICE_CODEC_DECODER_TRT_CACHE",
                dualcodec_ckpt / "trt_cache" / "packed_decode",
            ),
            silence_codec_path=_env_path(
                "VIITORVOICE_CODEC_SILENCE_TOKENS",
                local_models / "assets" / "dualcodec_silence_2s.pt",
            ),
            max_frames=_env_int("VIITORVOICE_CODEC_DECODER_MAX_FRAMES", 2048),
            silence_frames=_env_int("VIITORVOICE_CODEC_DECODER_SILENCE_FRAMES", 25),
        )
        debug_value = (_env_value("VIITORVOICE_DEBUG_DUMP_DIR", "") or "").strip()
        return cls(
            host=_env_value("VIITORVOICE_GRPC_HOST", "0.0.0.0") or "0.0.0.0",
            port=_env_int("VIITORVOICE_GRPC_PORT", 50051),
            device_id=device_id,
            max_queue_size=_env_int("VIITORVOICE_MAX_QUEUE_SIZE", 256),
            request_timeout_sec=_env_float("VIITORVOICE_REQUEST_TIMEOUT_SEC", 120.0),
            warmup_on_start=_env_bool("VIITORVOICE_WARMUP_ON_START", True),
            debug_dump_dir=Path(debug_value).expanduser() if debug_value else None,
            llm=llm,
            encoder=encoder,
            decoder=decoder,
            aligner=AlignerRuntimeConfig(
                model_path=_env_path(
                    "VIITORVOICE_ALIGNER_MODEL",
                    AlignerRuntimeConfig.model_path,
                ),
                device=_env_value("VIITORVOICE_ALIGNER_DEVICE", f"cuda:{device_id}") or f"cuda:{device_id}",
                language=_env_value("VIITORVOICE_ALIGNER_LANGUAGE", "zh") or "zh",
                dtype=_env_value("VIITORVOICE_ALIGNER_DTYPE", "auto") or "auto",
                attn_implementation=_env_value("VIITORVOICE_ALIGNER_ATTN_IMPLEMENTATION", "") or "",
            ),
        )


def norm_backend(value: str, default: str = "onnx-trt") -> str:
    text = (value or default).strip().lower().replace("_", "-")
    if text in {"trt", "onnx-trt", "ort-trt"}:
        return "onnx-trt"
    if text in {"cuda", "onnx-cuda", "ort-cuda"}:
        return "onnx-cuda"
    raise ValueError(f"Unsupported backend {value!r}.")


def codec_backend(value: str) -> str:
    return "trt" if norm_backend(value) == "onnx-trt" else "cuda"


@dataclass(frozen=True)
class OrchestratorTargets:
    encoder_target: str = "127.0.0.1:51051"
    llm_target: str = "127.0.0.1:51052"
    decoder_target: str = "127.0.0.1:51053"

    @classmethod
    def from_env(cls) -> "OrchestratorTargets":
        return cls(
            encoder_target=_env_value("VIITORVOICE_V2_ENCODER_TARGET", "127.0.0.1:51051") or "127.0.0.1:51051",
            llm_target=_env_value("VIITORVOICE_V2_LLM_TARGET", "127.0.0.1:51052") or "127.0.0.1:51052",
            decoder_target=_env_value("VIITORVOICE_V2_DECODER_TARGET", "127.0.0.1:51053") or "127.0.0.1:51053",
        )


@dataclass(frozen=True)
class V2RuntimeConfig:
    service: ServiceConfig
    targets: OrchestratorTargets = OrchestratorTargets()
    log_json: bool = True

    @classmethod
    def from_env(cls, *, default_port: int) -> "V2RuntimeConfig":
        service = ServiceConfig.from_env()
        port_env = _env_value("VIITORVOICE_GRPC_PORT")
        if default_port == 50051:
            port_env = _env_value("VIITORVOICE_V2_ORCH_PORT", port_env)
        service = ServiceConfig(
            host=_env_value("VIITORVOICE_GRPC_HOST", service.host) or service.host,
            port=int(port_env) if port_env not in {None, ""} else default_port,
            device_id=service.device_id,
            max_queue_size=service.max_queue_size,
            request_timeout_sec=float(
                _env_value(
                    "VIITORVOICE_V2_REQUEST_TIMEOUT_SEC",
                    _env_value("VIITORVOICE_REQUEST_TIMEOUT_SEC", str(service.request_timeout_sec)),
                )
            ),
            warmup_on_start=service.warmup_on_start,
            debug_dump_dir=service.debug_dump_dir,
            llm=service.llm,
            encoder=service.encoder,
            decoder=service.decoder,
            aligner=service.aligner,
        )
        return cls(
            service=service,
            targets=OrchestratorTargets.from_env(),
            log_json=(_env_value("VIITORVOICE_V2_LOG_JSON", "true") or "true").strip().lower()
            not in {"0", "false", "no", "off"},
        )


__all__ = [
    "DEFAULT_FRAME_RATE",
    "DEFAULT_HOP_LENGTH",
    "DEFAULT_SAMPLE_RATE",
    "OrchestratorTargets",
    "ServiceConfig",
    "V2RuntimeConfig",
    "clear_proxies",
    "codec_backend",
    "norm_backend",
]
