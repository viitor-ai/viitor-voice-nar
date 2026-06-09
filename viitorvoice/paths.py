from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLM_MODEL_NAME = "1p7_nvv"


def repo_root() -> Path:
    return REPO_ROOT


def env_value(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value not in {None, ""}:
        return value
    return default


def env_path(name: str, default: Path) -> Path:
    value = env_value(name)
    return Path(value).expanduser() if value else default


def local_models_root() -> Path:
    return env_path("VIITORVOICE_LOCAL_MODELS", REPO_ROOT / "local_models")


def llm_checkpoint_dir(model_name: str = DEFAULT_LLM_MODEL_NAME) -> Path:
    return local_models_root() / "llm" / model_name


def llm_onnx_path(checkpoint_dir: Path | None = None) -> Path:
    base = checkpoint_dir or llm_checkpoint_dir()
    return base / ".cache" / "onnx_backbone_fp32" / "llm_backbone_dynamic.onnx"


def llm_trt_cache_root(checkpoint_dir: Path | None = None) -> Path:
    base = checkpoint_dir or llm_checkpoint_dir()
    return base / ".cache" / "trt_cache"


def dualcodec_root() -> Path:
    return local_models_root() / "dualcodec"


def dualcodec_ckpt_dir() -> Path:
    return dualcodec_root() / "dualcodec_ckpts"


def dualcodec_encoder_onnx_path() -> Path:
    return dualcodec_ckpt_dir() / "dualcodec_encode_core_30s.onnx"


def dualcodec_decoder_onnx_path() -> Path:
    return dualcodec_ckpt_dir() / "dualcodec_decoder.onnx"


def dualcodec_w2v_path() -> Path:
    return dualcodec_root() / "w2v-bert-2.0"


def dualcodec_encoder_trt_cache_root() -> Path:
    return dualcodec_ckpt_dir() / "trt_cache" / "packed_encode"


def dualcodec_decoder_trt_cache_root() -> Path:
    return dualcodec_ckpt_dir() / "trt_cache" / "packed_decode"


def silence_codec_path() -> Path:
    return local_models_root() / "assets" / "dualcodec_silence_2s.pt"


def aligner_model_path() -> Path:
    return local_models_root() / "aligner" / "Qwen3-ForcedAligner-0.6B"


def resolve_model_asset_path(value: str | Path, fallback: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        REPO_ROOT / path,
    ]
    if fallback is not None:
        candidates.append(fallback)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return fallback or candidates[-1]
