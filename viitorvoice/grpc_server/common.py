from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from viitorvoice.grpc_server.audio_io import encode_wav_bytes
from viitorvoice.grpc_server.config import DEFAULT_SAMPLE_RATE
from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2
from viitorvoice.llm import LLMGenerationConfig
from viitorvoice.local_edit import AlignmentItem


LOGGER = logging.getLogger("viitorvoice.inference.grpc_server")


def ensure_context(
    context: common_pb2.RequestContext,
    *,
    caller: str = "",
    parent_span_id: str = "",
) -> common_pb2.RequestContext:
    out = common_pb2.RequestContext()
    out.CopyFrom(context)
    if not out.trace_id:
        out.trace_id = uuid.uuid4().hex
    if not out.request_id:
        out.request_id = uuid.uuid4().hex
    if parent_span_id and not out.parent_span_id:
        out.parent_span_id = parent_span_id
    if caller and not out.caller:
        out.caller = caller
    return out


def child_context(
    parent: common_pb2.RequestContext,
    *,
    parent_span_id: str,
    caller: str,
) -> common_pb2.RequestContext:
    out = common_pb2.RequestContext(
        trace_id=parent.trace_id,
        request_id=parent.request_id,
        parent_span_id=parent_span_id,
        span_id=uuid.uuid4().hex,
        caller=caller,
        deadline_ms=parent.deadline_ms,
    )
    out.tags.update(parent.tags)
    return out


def new_span(context: common_pb2.RequestContext) -> common_pb2.RequestContext:
    out = common_pb2.RequestContext()
    out.CopyFrom(context)
    out.parent_span_id = context.span_id or context.parent_span_id
    out.span_id = uuid.uuid4().hex
    return out


def response_context(
    context: common_pb2.RequestContext,
    *,
    service: str,
    status: str = "ok",
    metrics: Sequence[common_pb2.StageMetric] | None = None,
) -> common_pb2.ResponseContext:
    out = common_pb2.ResponseContext(
        trace_id=context.trace_id,
        request_id=context.request_id,
        span_id=context.span_id,
        service=service,
        status=status,
    )
    if metrics:
        out.metrics.extend(metrics)
    return out


def log_event(
    event: str,
    context: common_pb2.RequestContext,
    *,
    service: str,
    rpc: str = "",
    stage: str = "",
    status: str = "ok",
    duration_ms: float | None = None,
    input: Mapping[str, Any] | None = None,
    output: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
    grpc_status_code: str = "",
) -> None:
    record: dict[str, Any] = {
        "event": event,
        "trace_id": context.trace_id,
        "request_id": context.request_id,
        "span_id": context.span_id,
        "parent_span_id": context.parent_span_id,
        "service": service,
        "rpc": rpc,
        "stage": stage,
        "status": status,
    }
    if duration_ms is not None:
        record["duration_ms"] = round(float(duration_ms), 3)
    if input is not None:
        record["input"] = dict(input)
    if output is not None:
        record["output"] = dict(output)
    if error is not None:
        record["error_type"] = type(error).__name__
        record["error_message"] = str(error)
    if grpc_status_code:
        record["grpc_status_code"] = grpc_status_code
    LOGGER.info(json.dumps(record, ensure_ascii=False, sort_keys=True))


class StageTimer(AbstractContextManager["StageTimer"]):
    def __init__(
        self,
        context: common_pb2.RequestContext,
        *,
        service: str,
        stage: str,
        rpc: str = "",
        input: Mapping[str, Any] | None = None,
        target: str = "",
    ) -> None:
        self.context = context
        self.service = service
        self.stage = stage
        self.rpc = rpc
        self.input = dict(input or {})
        if target:
            self.input["target"] = target
        self.output: dict[str, Any] = {}
        self.start_time = 0.0
        self.start_ms = 0
        self.metric: common_pb2.StageMetric | None = None

    def __enter__(self) -> "StageTimer":
        self.start_time = time.perf_counter()
        self.start_ms = int(time.time() * 1000)
        log_event(
            "stage_start",
            self.context,
            service=self.service,
            rpc=self.rpc,
            stage=self.stage,
            input=self.input,
        )
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        duration = (time.perf_counter() - self.start_time) * 1000.0
        status = "error" if exc is not None else "ok"
        log_event(
            "stage_end",
            self.context,
            service=self.service,
            rpc=self.rpc,
            stage=self.stage,
            status=status,
            duration_ms=duration,
            input=self.input,
            output=self.output,
            error=exc,
        )
        metric = common_pb2.StageMetric(
            service=self.service,
            stage=self.stage,
            start_time_unix_ms=self.start_ms,
            duration_ms=duration,
        )
        metric.input.update(_stringify_map(self.input))
        metric.output.update(_stringify_map(self.output))
        self.metric = metric
        return False


def summarize_audio_input(audio: common_pb2.AudioInput) -> dict[str, Any]:
    if audio.audio_path:
        return {"audio_path": str(Path(audio.audio_path).expanduser()), "sample_rate": int(audio.sample_rate)}
    if audio.audio_bytes:
        return {"audio_bytes": f"{len(audio.audio_bytes)} bytes", "sample_rate": int(audio.sample_rate)}
    return {"audio": "empty"}


def summarize_audio_result(audio: common_pb2.AudioResult) -> dict[str, Any]:
    return {
        "audio_bytes": f"{len(audio.audio_bytes)} bytes",
        "sample_rate": int(audio.sample_rate),
        "channels": int(audio.channels),
        "duration_sec": round(float(audio.duration_sec), 4),
    }


def summarize_tensor(tensor: common_pb2.Int64Tensor | torch.Tensor | np.ndarray | None) -> dict[str, Any]:
    if tensor is None:
        return {"shape": [], "numel": 0, "dtype": "int64", "bytes_estimate": 0}
    if isinstance(tensor, common_pb2.Int64Tensor):
        shape = [int(x) for x in tensor.shape]
        numel = len(tensor.values)
    else:
        arr = torch.as_tensor(tensor)
        shape = [int(x) for x in arr.shape]
        numel = int(arr.numel())
    return {"shape": shape, "numel": numel, "dtype": "int64", "bytes_estimate": numel * 8}


def summarize_text(condition: common_pb2.TextCondition) -> dict[str, Any]:
    return {
        "text": condition.text,
        "text_chars": len(condition.text),
        "language": condition.language,
        "ref_text": condition.ref_text,
        "has_ref_text": bool(condition.ref_text),
        "instruct": condition.instruct,
        "instruct_chars": len(condition.instruct),
        "allow_missing_ref_text": bool(condition.allow_missing_ref_text),
        "ref_text_mask_len": int(condition.ref_text_mask_len),
    }


def summarize_generation_config(generation: common_pb2.GenerationConfig) -> dict[str, Any]:
    fields = (
        "max_new_tokens",
        "num_steps",
        "temperature",
        "top_p",
        "top_k",
        "cfg_scale",
        "seed",
        "debug",
        "debug_request_id",
        "request_timeout_sec",
        "t_shift",
        "layer_penalty_factor",
        "position_temperature",
        "class_temperature",
        "denoise",
        "preprocess_prompt",
        "postprocess_output",
        "audio_chunk_duration",
        "audio_chunk_threshold",
        "duration",
        "speed",
        "emotion_guidance_scale",
        "nvv_guidance_scale",
    )
    return {field: getattr(generation, field) for field in fields if _has_field(generation, field)}


def tensor_from_proto(
    tensor: common_pb2.Int64Tensor,
    *,
    name: str,
    required: bool = False,
) -> torch.Tensor | None:
    if not tensor.values and not tensor.shape:
        if required:
            raise ValueError(f"{name} is required.")
        return None
    shape = [int(x) for x in tensor.shape]
    values = [int(x) for x in tensor.values]
    if not shape:
        raise ValueError(f"{name}.shape is required when values are provided.")
    expected = int(np.prod(shape, dtype=np.int64))
    if expected != len(values):
        raise ValueError(f"{name} has {len(values)} values but shape {shape} expects {expected}.")
    return torch.as_tensor(values, dtype=torch.long).reshape(shape).contiguous()


def tensor_to_proto(tensor: torch.Tensor | np.ndarray | Sequence[int] | None) -> common_pb2.Int64Tensor:
    message = common_pb2.Int64Tensor()
    if tensor is None:
        return message
    arr = torch.as_tensor(tensor, dtype=torch.long).detach().cpu().contiguous()
    message.shape.extend(int(x) for x in arr.shape)
    message.values.extend(int(x) for x in arr.reshape(-1).tolist())
    return message


def normalize_audio_codebook(tokens: torch.Tensor | np.ndarray | Sequence[int], *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(tokens, dtype=torch.long)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2 or tensor.shape[0] != 12:
        raise ValueError(f"{name} should have shape [12, T] or [1, 12, T], got {tuple(tensor.shape)}.")
    if tensor.shape[-1] <= 0:
        raise ValueError(f"{name} is empty.")
    return tensor.contiguous()


def normalize_semantic_tokens(tokens: torch.Tensor | np.ndarray | Sequence[int], *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(tokens, dtype=torch.long)
    if tensor.ndim == 2:
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        elif tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
    if tensor.ndim != 1:
        raise ValueError(f"{name} should have shape [T], [1, T], or [T, 1], got {tuple(tensor.shape)}.")
    if tensor.numel() <= 0:
        raise ValueError(f"{name} is empty.")
    return tensor.contiguous()


def alignment_item_from_proto(item: common_pb2.AlignmentItem) -> AlignmentItem:
    return AlignmentItem(
        index=int(item.index),
        text=str(item.text),
        start_time=float(item.start_time),
        end_time=float(item.end_time),
        start_char=int(item.start_char) if item.has_start_char else None,
        end_char=int(item.end_char) if item.has_end_char else None,
        kind=item.kind or ("char" if len(item.text) == 1 else "word"),
    )


def alignment_item_to_proto(item: AlignmentItem) -> common_pb2.AlignmentItem:
    msg = common_pb2.AlignmentItem(
        index=int(item.index),
        text=str(item.text),
        start_time=float(item.start_time),
        end_time=float(item.end_time),
        kind=item.kind or "word",
    )
    if item.start_char is not None:
        msg.start_char = int(item.start_char)
        msg.has_start_char = True
    if item.end_char is not None:
        msg.end_char = int(item.end_char)
        msg.has_end_char = True
    return msg


def alignment_items_from_proto(items: Iterable[common_pb2.AlignmentItem]) -> list[AlignmentItem]:
    return [alignment_item_from_proto(item) for item in items]


def count_remaining_mask_tokens(tokens: torch.Tensor, mask_ids: Sequence[int]) -> int:
    tensor = torch.as_tensor(tokens, dtype=torch.long)
    ids = torch.as_tensor(list(mask_ids), dtype=torch.long).view(-1, 1)
    if tensor.ndim != 2 or tensor.shape[0] != ids.numel():
        return 0
    return int((tensor == ids).sum().item())


def generation_config_from_proto(message: common_pb2.GenerationConfig) -> LLMGenerationConfig:
    defaults = LLMGenerationConfig()
    return LLMGenerationConfig(
        num_step=_optional_int(message, "num_steps", defaults.num_step),
        guidance_scale=_optional_float(message, "cfg_scale", defaults.guidance_scale),
        emotion_guidance_scale=_optional_float(
            message, "emotion_guidance_scale", defaults.emotion_guidance_scale
        ),
        nvv_guidance_scale=_optional_float(message, "nvv_guidance_scale", defaults.nvv_guidance_scale),
        t_shift=_optional_float(message, "t_shift", defaults.t_shift),
        layer_penalty_factor=_optional_float(message, "layer_penalty_factor", defaults.layer_penalty_factor),
        position_temperature=_optional_float(message, "position_temperature", defaults.position_temperature),
        class_temperature=_optional_float(message, "class_temperature", defaults.class_temperature),
        denoise=_optional_bool(message, "denoise", defaults.denoise),
        preprocess_prompt=_optional_bool(message, "preprocess_prompt", defaults.preprocess_prompt),
        postprocess_output=_optional_bool(message, "postprocess_output", defaults.postprocess_output),
        audio_chunk_duration=_optional_float(message, "audio_chunk_duration", defaults.audio_chunk_duration),
        audio_chunk_threshold=_optional_float(message, "audio_chunk_threshold", defaults.audio_chunk_threshold),
    )


def optional_int(message: Any, field: str, default: int | None) -> int | None:
    return _optional_int(message, field, default)


def optional_float(message: Any, field: str, default: float | None) -> float | None:
    return _optional_float(message, field, default)


def optional_bool(message: Any, field: str, default: bool) -> bool:
    return _optional_bool(message, field, default)


def generation_float(message: common_pb2.GenerationConfig, field: str) -> float | None:
    value = _optional_float(message, field, None)
    return None if value is None or value <= 0 else float(value)


def audio_result(wave: np.ndarray) -> common_pb2.AudioResult:
    return common_pb2.AudioResult(
        audio_bytes=encode_wav_bytes(wave),
        sample_rate=DEFAULT_SAMPLE_RATE,
        format=common_pb2.AUDIO_FORMAT_WAV,
        channels=1,
        duration_sec=float(np.asarray(wave).reshape(-1).shape[0]) / DEFAULT_SAMPLE_RATE,
    )


def first_tensor(generated: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(generated, list):
        return generated[0] if generated else torch.empty((12, 0), dtype=torch.long)
    return generated


def strip_structural_audio_frames(tokens: torch.Tensor, llm: Any) -> torch.Tensor:
    strip_ids = []
    if getattr(llm, "audio_pause_ids", None) is not None:
        strip_ids.append(llm.audio_pause_ids)
    if getattr(llm, "audio_separator_ids", None) is not None:
        strip_ids.append(llm.audio_separator_ids)
    if not strip_ids or tokens.numel() == 0:
        return tokens
    keep = torch.ones(tokens.size(-1), dtype=torch.bool, device=tokens.device)
    for ids in strip_ids:
        ids_t = torch.as_tensor(ids, dtype=tokens.dtype, device=tokens.device).view(-1, 1)
        keep &= ~torch.all(tokens == ids_t, dim=0)
    return tokens[:, keep]


def granularity_name(value: int | str) -> str:
    if isinstance(value, str):
        text = value.strip().lower()
        return "char" if text in {"char", "character"} else (text or "auto")
    if value == common_pb2.ALIGNMENT_GRANULARITY_WORD:
        return "word"
    if value == common_pb2.ALIGNMENT_GRANULARITY_CHARACTER:
        return "char"
    return "auto"


def edits_to_selection_text(edits: Sequence[common_pb2.EditSegment]) -> tuple[str, list[str]]:
    if not edits:
        raise ValueError("At least one edit segment is required.")
    selections: list[str] = []
    replacements: list[str] = []
    for edit in edits:
        indices = [int(x) for x in edit.selection.alignment_indices]
        if not indices:
            raise ValueError("Only alignment_indices based edits are currently supported.")
        indices = sorted(set(indices))
        runs: list[str] = []
        start = prev = indices[0]
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            runs.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = idx
        runs.append(f"{start}-{prev}" if start != prev else str(start))
        selections.append(",".join(runs))
        replacements.append(edit.replacement_text)
    return ";".join(selections), replacements


def unlink_temp(path: str, audio: common_pb2.AudioInput) -> None:
    if audio.audio_path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


def _optional_int(message: Any, field: str, default: int | None) -> int | None:
    return int(getattr(message, field)) if message.HasField(field) else default


def _optional_float(message: Any, field: str, default: float | None) -> float | None:
    return float(getattr(message, field)) if message.HasField(field) else default


def _optional_bool(message: Any, field: str, default: bool) -> bool:
    return bool(getattr(message, field)) if message.HasField(field) else default


def _has_field(message: Any, field: str) -> bool:
    try:
        return bool(message.HasField(field))
    except ValueError:
        return False


def _stringify_map(values: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in values.items():
        if isinstance(value, (dict, list, tuple)):
            out[str(key)] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            out[str(key)] = str(value)
    return out
