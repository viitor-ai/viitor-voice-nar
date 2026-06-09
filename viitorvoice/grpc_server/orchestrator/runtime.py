from __future__ import annotations

from pathlib import Path
import re

from viitorvoice.grpc_server import common
from viitorvoice.grpc_server.audio_io import audio_input_to_temp_wav
from viitorvoice.grpc_server.config import DEFAULT_FRAME_RATE, ServiceConfig
from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2
from viitorvoice.grpc_server.proto import viitorvoice_orchestrator_pb2 as orch_pb2
from viitorvoice.local_edit import Qwen3ForcedAlignerWrapper, build_multi_local_edit_token_plan
from viitorvoice.llm.text_utils import normalize_nvv_tags


class OrchestratorRuntime:
    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self._aligners: dict[tuple[str, str, str, str], Qwen3ForcedAlignerWrapper] = {}

    def warmup(self) -> None:
        self._get_aligner(None, None, None)

    def runtime_info(self, context: common_pb2.RequestContext, message: str = "") -> common_pb2.HealthResponse:
        backends = ["orchestrator"]
        if self._aligners:
            backends.extend(
                f"aligner:{model}@{device}:dtype={dtype}:attn={attn or 'default'}"
                for model, device, dtype, attn in self._aligners
            )
        return common_pb2.HealthResponse(
            context=common.response_context(context, service="orchestrator"),
            state=common_pb2.SERVICE_STATE_READY,
            message=message or "ready",
            version="inference-grpc-v2-orchestrator",
            active_backends=backends,
            queued_jobs=0,
        )

    def align_for_edit(
        self,
        request: orch_pb2.AlignForEditRequest,
        context: common_pb2.RequestContext,
    ) -> orch_pb2.AlignForEditResponse:
        response: orch_pb2.AlignForEditResponse
        with common.StageTimer(
            context,
            service="orchestrator",
            rpc="AlignForEdit",
            stage="aligner.align",
            input=common.summarize_audio_input(request.source_audio),
        ) as timer:
            audio_path = audio_input_to_temp_wav(request.source_audio)
            try:
                aligner = self._get_aligner(None, None, request.language or None)
                items = aligner.align(
                    audio=audio_path,
                    text=request.original_text,
                    language=request.language or None,
                    granularity=common.granularity_name(request.granularity),
                )
                response = orch_pb2.AlignForEditResponse(
                    context=common.response_context(context, service="orchestrator"),
                    alignments=[common.alignment_item_to_proto(item) for item in items],
                    sample_rate=24000,
                )
                timer.output.update({"alignment_items": len(items)})
            finally:
                common.unlink_temp(audio_path, request.source_audio)
        if timer.metric is not None:
            response.context.metrics.append(timer.metric)
        return response

    def build_local_edit_plan(self, request: orch_pb2.LocalEditRequest, source_tokens, source_duration: float):
        items = common.alignment_items_from_proto(request.alignments)
        if not items:
            audio_path = audio_input_to_temp_wav(request.source_audio)
            try:
                aligner = self._get_aligner(None, None, request.language or None)
                items = aligner.align(
                    audio=audio_path,
                    text=request.original_text,
                    language=request.language or None,
                    granularity=_resolve_alignment_granularity_name(
                        language=request.language,
                        text=request.original_text,
                        explicit=request.align_granularity,
                    ),
                )
            finally:
                common.unlink_temp(audio_path, request.source_audio)

        selection, replacement_text = common.edits_to_selection_text(request.edits)
        replacement_text = [normalize_nvv_tags(text) for text in replacement_text]
        plan = build_multi_local_edit_token_plan(
            source_tokens=source_tokens,
            alignment_items=items,
            original_text=request.original_text,
            selection=selection,
            replacement_text=replacement_text,
            audio_duration=source_duration,
            padding_ms=common.optional_float(request, "padding_ms", 0.0),
            expand_mask_ratio=common.optional_float(request, "expand_mask_ratio", 1.0),
            frame_rate=DEFAULT_FRAME_RATE,
            length_mode=request.length_mode or "auto",
            manual_seconds=common.optional_float(request, "manual_duration", None),
            manual_frames=common.optional_int(request, "manual_frames", None),
            length_scale=common.optional_float(request, "length_scale", 1.0),
            min_edit_frames=common.optional_int(request, "min_mask_frames", 6),
            reference_context_frames=common.optional_int(request, "edit_ref_context_frames", 300),
        )
        return plan, items

    def _get_aligner(
        self,
        model_path: str | None,
        device: str | None,
        _language: str | None,
    ) -> Qwen3ForcedAlignerWrapper:
        resolved_model = (
            str(Path(model_path).expanduser().resolve()) if model_path else str(self.config.aligner.model_path)
        )
        resolved_device = device or self.config.aligner.device
        resolved_dtype = self.config.aligner.dtype
        resolved_attn = self.config.aligner.attn_implementation
        key = (resolved_model, resolved_device, resolved_dtype, resolved_attn)
        if key not in self._aligners:
            self._aligners[key] = Qwen3ForcedAlignerWrapper(
                model_path=resolved_model,
                device=resolved_device,
                default_language=self.config.aligner.language,
                dtype=resolved_dtype,
                attn_implementation=resolved_attn,
            )
        return self._aligners[key]


_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_JA_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_KO_HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")
_ASCII_WORD_RE = re.compile(r"[A-Za-z]")


def _resolve_alignment_granularity_name(*, language: str | None, text: str, explicit: str | None) -> str:
    explicit_text = common.granularity_name(explicit or "")
    if explicit_text in {"word", "char"}:
        return explicit_text

    lang = _normalize_edit_language(language)
    if lang in {"zh", "yue", "ja", "ko"}:
        return "char"
    if lang == "en":
        return "word"
    if _CJK_CHAR_RE.search(text) or _JA_KANA_RE.search(text) or _KO_HANGUL_RE.search(text):
        return "char"
    if _ASCII_WORD_RE.search(text):
        return "word"
    return "char"


def _normalize_edit_language(language: str | None) -> str | None:
    if language is None:
        return None
    text = str(language).strip().lower().replace("_", "-")
    aliases = {
        "english": "en",
        "eng": "en",
        "en-us": "en",
        "en-gb": "en",
        "chinese": "zh",
        "mandarin": "zh",
        "zho": "zh",
        "cmn": "zh",
        "cn": "zh",
        "zh-cn": "zh",
        "zh-hans": "zh",
        "zh-hant": "zh",
        "japanese": "ja",
        "jpn": "ja",
        "jp": "ja",
        "ja-jp": "ja",
        "korean": "ko",
        "kor": "ko",
        "kr": "ko",
        "ko-kr": "ko",
    }
    return aliases.get(text, text or None)
