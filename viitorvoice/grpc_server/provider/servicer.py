from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

import grpc

from viitorvoice.grpc_server import common as grpc_common
from viitorvoice.grpc_server.config import DEFAULT_SAMPLE_RATE, clear_proxies
from viitorvoice.grpc_server.proto import backend_provider_pb2 as provider_pb2
from viitorvoice.grpc_server.proto import backend_provider_pb2_grpc as provider_pb2_grpc
from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2
from viitorvoice.grpc_server.proto import viitorvoice_encoder_pb2 as encoder_pb2
from viitorvoice.grpc_server.proto import viitorvoice_orchestrator_pb2 as orch_pb2
from viitorvoice.grpc_server.proto import viitorvoice_orchestrator_pb2_grpc as orch_pb2_grpc
from viitorvoice.grpc_server.servicer_utils import run_rpc


MAX_GRPC_MESSAGE_BYTES = 512 * 1024 * 1024
BACKEND_ID = os.environ.get("SPEECH_EDIT_BACKEND_ID", "speech-edit-grpc-v2")
BACKEND_VERSION = os.environ.get("SPEECH_EDIT_BACKEND_VERSION", "viitorvoice.grpc_server")
PROTOCOL_VERSION = os.environ.get("SPEECH_EDIT_BACKEND_PROTOCOL_VERSION", "tts.backend.v1")
FEATURE_SCHEMA = os.environ.get(
    "SPEECH_EDIT_PROMPT_FEATURE_SCHEMA",
    "viitorvoice.inference.v2.Int64Tensor.audio_codebook",
)
FEATURE_VERSION = os.environ.get("SPEECH_EDIT_PROMPT_FEATURE_VERSION", "1")
FEATURE_MODEL_VERSION = os.environ.get(
    "SPEECH_EDIT_PROMPT_FEATURE_MODEL_VERSION",
    os.environ.get("VIITORVOICE_CODEC_ENCODER_MODEL_ID", "25hz_v1"),
)
FEATURE_CONTENT_TYPE = "application/x-protobuf; message=viitorvoice.inference.v2.Int64Tensor"
DEFAULT_ORCHESTRATOR_TARGET = "127.0.0.1:50051"
DEFAULT_TIMEOUT_SEC = float(os.environ.get("SPEECH_EDIT_PROVIDER_REQUEST_TIMEOUT_SEC", "600"))


class BackendProviderServicer(provider_pb2_grpc.BackendProviderServiceServicer):
    def __init__(self, orchestrator_target: str) -> None:
        clear_proxies()
        self.orchestrator_target = orchestrator_target
        self.channel = grpc.aio.insecure_channel(orchestrator_target, options=_grpc_channel_options())
        self.orchestrator = orch_pb2_grpc.ViiTorVoiceOrchestratorServiceStub(self.channel)

    async def close(self) -> None:
        await self.channel.close(grace=5)

    async def Health(
        self,
        request: provider_pb2.HealthRequest,
        context: grpc.aio.ServicerContext,
    ) -> provider_pb2.HealthResponse:
        del context
        try:
            response = await self.orchestrator.Health(
                common_pb2.HealthRequest(context=_to_viitorvoice_context(request.context)),
                timeout=_timeout_from_context(request.context),
            )
        except grpc.aio.AioRpcError as exc:
            return provider_pb2.HealthResponse(
                status="unavailable",
                message=exc.details() or str(exc),
                backend_id=BACKEND_ID,
                backend_version=BACKEND_VERSION,
            )
        return provider_pb2.HealthResponse(
            status=_health_status(response.state),
            message=response.message,
            backend_id=BACKEND_ID,
            backend_version=BACKEND_VERSION,
        )

    async def GetCapabilities(
        self,
        request: provider_pb2.GetCapabilitiesRequest,
        context: grpc.aio.ServicerContext,
    ) -> provider_pb2.CapabilitiesResponse:
        del request, context
        return provider_pb2.CapabilitiesResponse(capabilities=_capabilities())

    async def PreparePrompt(
        self,
        request: provider_pb2.PreparePromptRequest,
        context: grpc.aio.ServicerContext,
    ) -> provider_pb2.PreparePromptResponse:
        async def invoke() -> provider_pb2.PreparePromptResponse:
            req_context = _to_viitorvoice_context(request.context)
            audio = _to_viitorvoice_audio_input(request.audio)
            grpc_common.log_event(
                "provider_request_received",
                req_context,
                service="provider",
                rpc="PreparePrompt",
                input={
                    "prompt_id": request.prompt_id,
                    "prompt_text": request.prompt_text,
                    "prompt_text_chars": len(request.prompt_text),
                    "language": request.lang,
                    "audio": grpc_common.summarize_audio_input(audio),
                    "metadata": dict(request.metadata),
                },
            )
            response = await self.orchestrator.EncodeAudio(
                encoder_pb2.EncodeAudioRequest(
                    context=req_context,
                    audio=audio,
                    preprocess_prompt=_metadata_bool(request.metadata, "preprocess_prompt", True),
                    return_audio=False,
                    can_trim_long_audio=True,
                ),
                timeout=_timeout_from_context(request.context),
            )
            artifact = _artifact_from_codebook(
                prompt_id=request.prompt_id,
                codebook=response.audio_codebook,
                audio=request.audio,
            )
            if request.prompt_text:
                artifact.metadata["prompt_text"] = request.prompt_text
            if request.lang:
                artifact.metadata["lang"] = request.lang
            grpc_common.log_event(
                "provider_request_completed",
                req_context,
                service="provider",
                rpc="PreparePrompt",
                output={
                    "artifacts": len([artifact]),
                    "feature_schema": artifact.feature_schema,
                    "feature_version": artifact.feature_version,
                },
            )
            return provider_pb2.PreparePromptResponse(backend_id=BACKEND_ID, artifacts=[artifact])

        return await run_rpc(context, invoke)

    async def Synthesize(
        self,
        request: provider_pb2.SynthesizeRequest,
        context: grpc.aio.ServicerContext,
    ) -> provider_pb2.SynthesizeResponse:
        async def invoke() -> provider_pb2.SynthesizeResponse:
            req_context = _to_viitorvoice_context(request.context)
            _validate_output_spec(request.output)
            segments = list(request.segments)
            if not segments:
                raise ValueError("SynthesizeRequest.segments is required.")

            ref_codebook = _codebook_from_prompt_features(request.prompt.features)
            ref_audio = _to_viitorvoice_audio_input(request.prompt.prompt_audio) if ref_codebook is None else common_pb2.AudioInput()
            if ref_codebook is None and not _has_audio(request.prompt.prompt_audio):
                raise ValueError("A prompt audio payload or prepared prompt feature is required.")
            generation = _generation_config(request.generation, request.context)
            grpc_common.log_event(
                "provider_request_received",
                req_context,
                service="provider",
                rpc="Synthesize",
                input={
                    "segments": [
                        {
                            "index": int(segment.index),
                            "text": segment.text.strip(),
                            "text_chars": len(segment.text.strip()),
                            "language": segment.lang or request.prompt.lang,
                            "emotion": segment.emotion,
                        }
                        for segment in segments
                    ],
                    "prompt_text": request.prompt.prompt_text,
                    "prompt_language": request.prompt.lang,
                    "has_prompt_feature": ref_codebook is not None,
                    "prompt_audio": grpc_common.summarize_audio_input(ref_audio) if ref_codebook is None else {},
                    "output_format": request.output.format,
                    "generation": grpc_common.summarize_generation_config(generation),
                    "generation_extra": dict(request.generation.extra),
                },
            )

            audio_items: list[provider_pb2.AudioItem] = []
            total_audio_bytes = 0
            for position, segment in enumerate(segments):
                text = segment.text.strip()
                if not text:
                    raise ValueError(f"segments[{position}].text is required.")
                response = await self.orchestrator.Synthesize(
                    orch_pb2.SynthesizeRequest(
                        context=_to_viitorvoice_context(request.context),
                        condition=_text_condition(request, segment),
                        ref_audio=ref_audio,
                        ref_audio_codebook=ref_codebook or common_pb2.Int64Tensor(),
                        generation=generation,
                        return_tokens=False,
                        output_format=_to_viitorvoice_output_format(request.output),
                    ),
                    timeout=_timeout_from_context(request.context),
                )
                audio_payload = _to_provider_audio_payload(response.audio)
                total_audio_bytes += len(audio_payload.audio_bytes)
                audio_items.append(
                    provider_pb2.AudioItem(
                        item_index=segment.index if segment.index else position,
                        text=text,
                        audio=audio_payload,
                    )
                )

            usage = provider_pb2.BackendUsage(
                input_segments=len(segments),
                output_items=len(audio_items),
                audio_bytes=total_audio_bytes,
            )
            usage.metrics["orchestrator_target"] = self.orchestrator_target
            usage.metrics["supports_true_streaming"] = "false"
            grpc_common.log_event(
                "provider_request_completed",
                req_context,
                service="provider",
                rpc="Synthesize",
                output={
                    "input_segments": len(segments),
                    "output_items": len(audio_items),
                    "audio_bytes": total_audio_bytes,
                },
            )
            return provider_pb2.SynthesizeResponse(audio_items=audio_items, usage=usage)

        return await run_rpc(context, invoke)

    async def SynthesizeStream(
        self,
        request: provider_pb2.SynthesizeRequest,
        context: grpc.aio.ServicerContext,
    ):
        del request
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "speech-edit provider does not support true streaming yet.")
        if False:
            yield provider_pb2.SynthesizeStreamEvent()


def orchestrator_target_from_env() -> str:
    return os.environ.get(
        "SPEECH_EDIT_ORCHESTRATOR_TARGET",
        os.environ.get("VIITORVOICE_V2_ORCHESTRATOR_TARGET", DEFAULT_ORCHESTRATOR_TARGET),
    )


def _grpc_channel_options() -> list[tuple[str, int]]:
    return [
        ("grpc.max_send_message_length", MAX_GRPC_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", MAX_GRPC_MESSAGE_BYTES),
    ]


def _capabilities() -> provider_pb2.BackendCapabilities:
    capabilities = provider_pb2.BackendCapabilities(
        backend_id=BACKEND_ID,
        backend_version=BACKEND_VERSION,
        protocol_version=PROTOCOL_VERSION,
        supported_languages=["en", "zh"],
        supported_output_formats=["wav"],
        supported_sample_rates=[DEFAULT_SAMPLE_RATE],
        supports_unary=True,
        supports_true_streaming=False,
        supports_prompt_registration=False,
        supports_prompt_features=True,
        prompt_feature_specs=[
            provider_pb2.PromptFeatureSpec(
                feature_schema=FEATURE_SCHEMA,
                feature_version=FEATURE_VERSION,
                model_version=FEATURE_MODEL_VERSION,
            )
        ],
        supported_generation_params=[
            "duration",
            "temperature",
            "top_p",
            "top_k",
            "num_steps",
            "cfg_scale",
            "emotion_guidance_scale",
            "nvv_guidance_scale",
            "seed",
            "speed",
            "request_timeout_sec",
            "preprocess_prompt",
            "postprocess_output",
        ],
    )
    capabilities.labels.update(
        {
            "adapter": "viitorvoice.grpc_server.provider",
            "orchestrator_target_env": "SPEECH_EDIT_ORCHESTRATOR_TARGET",
            "prompt_feature_content_type": FEATURE_CONTENT_TYPE,
        }
    )
    return capabilities


def _health_status(state: int) -> str:
    if state == common_pb2.SERVICE_STATE_READY:
        return "serving"
    if state == common_pb2.SERVICE_STATE_DEGRADED:
        return "degraded"
    if state == common_pb2.SERVICE_STATE_STARTING:
        return "starting"
    if state == common_pb2.SERVICE_STATE_STOPPING:
        return "stopping"
    return "unknown"


def _to_viitorvoice_context(context: provider_pb2.RequestContext) -> common_pb2.RequestContext:
    request_id = context.request_id or uuid.uuid4().hex
    trace_id = context.trace_id or request_id
    out = common_pb2.RequestContext(
        trace_id=trace_id,
        request_id=request_id,
        span_id=uuid.uuid4().hex,
        caller=context.caller or "tts.backend.v1.provider",
        deadline_ms=context.deadline_ms,
    )
    if context.app_id:
        out.tags["app_id"] = context.app_id
    if context.task_id:
        out.tags["task_id"] = context.task_id
    if context.debug:
        out.tags["debug"] = "true"
    return out


def _timeout_from_context(context: provider_pb2.RequestContext) -> float:
    if context.deadline_ms <= 0:
        return DEFAULT_TIMEOUT_SEC
    if context.deadline_ms > 100_000_000_000:
        return max(0.001, (context.deadline_ms - int(time.time() * 1000)) / 1000.0)
    return max(0.001, context.deadline_ms / 1000.0)


def _to_viitorvoice_audio_input(audio: provider_pb2.AudioPayload) -> common_pb2.AudioInput:
    if audio.audio_bytes:
        return common_pb2.AudioInput(
            audio_bytes=audio.audio_bytes,
            sample_rate=audio.sample_rate or DEFAULT_SAMPLE_RATE,
            format=_audio_payload_format(audio),
        )
    if audio.audio_uri:
        return common_pb2.AudioInput(
            audio_path=_local_path_from_uri(audio.audio_uri),
            sample_rate=audio.sample_rate or DEFAULT_SAMPLE_RATE,
            format=_audio_payload_format(audio),
        )
    raise ValueError("AudioPayload.audio_bytes or audio_uri is required.")


def _audio_payload_format(audio: provider_pb2.AudioPayload) -> int:
    text = (audio.content_type or audio.sample_format or "").strip().lower()
    if not text or text in {"audio/wav", "audio/x-wav", "wav"}:
        return common_pb2.AUDIO_FORMAT_WAV
    if text in {"audio/flac", "flac"}:
        return common_pb2.AUDIO_FORMAT_FLAC
    if text in {"audio/pcm", "audio/l16", "pcm_s16le", "s16le"}:
        return common_pb2.AUDIO_FORMAT_PCM_S16LE
    raise ValueError(f"Unsupported audio content type or sample format: {text}")


def _to_viitorvoice_output_format(output: provider_pb2.OutputSpec) -> int:
    text = (output.format or "wav").strip().lower()
    if text in {"wav", "audio/wav", "audio/x-wav"}:
        return common_pb2.AUDIO_FORMAT_WAV
    if text in {"flac", "audio/flac"}:
        return common_pb2.AUDIO_FORMAT_FLAC
    if text in {"pcm_s16le", "s16le", "audio/pcm", "audio/l16"}:
        return common_pb2.AUDIO_FORMAT_PCM_S16LE
    raise ValueError(f"Unsupported output format: {output.format}")


def _validate_output_spec(output: provider_pb2.OutputSpec) -> None:
    if output.sample_rate and output.sample_rate != DEFAULT_SAMPLE_RATE:
        raise ValueError(f"Unsupported output sample_rate {output.sample_rate}; only {DEFAULT_SAMPLE_RATE} is supported.")
    if output.channels and output.channels != 1:
        raise ValueError(f"Unsupported output channels {output.channels}; only mono output is supported.")
    output_format = (output.format or "wav").strip().lower()
    if output_format not in {"wav", "audio/wav", "audio/x-wav"}:
        raise ValueError("The speech-edit provider currently exposes only WAV output through tts.backend.v1.")


def _to_provider_audio_payload(audio: common_pb2.AudioResult) -> provider_pb2.AudioPayload:
    return provider_pb2.AudioPayload(
        audio_bytes=audio.audio_bytes,
        content_type=_content_type_from_audio_format(audio.format),
        sample_rate=int(audio.sample_rate or DEFAULT_SAMPLE_RATE),
        channels=int(audio.channels or 1),
        sample_format="s16le",
        duration_ms=int(round(float(audio.duration_sec) * 1000.0)) if audio.duration_sec else 0,
    )


def _content_type_from_audio_format(audio_format: int) -> str:
    if audio_format == common_pb2.AUDIO_FORMAT_FLAC:
        return "audio/flac"
    if audio_format == common_pb2.AUDIO_FORMAT_PCM_S16LE:
        return "audio/pcm"
    return "audio/wav"


def _local_path_from_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme in {"", "file"}:
        if parsed.scheme == "file":
            return unquote(parsed.path)
        return str(Path(uri).expanduser())
    raise ValueError(f"Unsupported audio_uri scheme: {parsed.scheme}")


def _has_audio(audio: provider_pb2.AudioPayload) -> bool:
    return bool(audio.audio_bytes or audio.audio_uri)


def _artifact_from_codebook(
    *,
    prompt_id: str,
    codebook: common_pb2.Int64Tensor,
    audio: provider_pb2.AudioPayload,
) -> provider_pb2.PromptFeatureArtifact:
    artifact = provider_pb2.PromptFeatureArtifact(
        prompt_id=prompt_id,
        backend_id=BACKEND_ID,
        feature_schema=FEATURE_SCHEMA,
        feature_version=FEATURE_VERSION,
        model_version=FEATURE_MODEL_VERSION,
        audio_sha256=_audio_sha256(audio),
        payload=codebook.SerializeToString(),
        content_type=FEATURE_CONTENT_TYPE,
    )
    artifact.metadata.update(
        {
            "shape": json.dumps([int(x) for x in codebook.shape], separators=(",", ":")),
            "values": str(len(codebook.values)),
            "payload_message": "viitorvoice.inference.v2.Int64Tensor",
        }
    )
    return artifact


def _audio_sha256(audio: provider_pb2.AudioPayload) -> str:
    if audio.audio_bytes:
        return hashlib.sha256(audio.audio_bytes).hexdigest()
    if audio.audio_uri:
        path = Path(_local_path_from_uri(audio.audio_uri))
        if path.exists() and path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
    return ""


def _codebook_from_prompt_features(
    features: list[provider_pb2.PromptFeatureArtifact],
) -> common_pb2.Int64Tensor | None:
    for feature in features:
        if feature.feature_schema and feature.feature_schema != FEATURE_SCHEMA:
            continue
        if feature.feature_version and feature.feature_version != FEATURE_VERSION:
            continue
        if feature.model_version and feature.model_version != FEATURE_MODEL_VERSION:
            continue
        if not feature.payload:
            continue
        codebook = common_pb2.Int64Tensor()
        try:
            codebook.ParseFromString(feature.payload)
        except Exception as exc:
            raise ValueError("Prompt feature payload is not a serialized Int64Tensor.") from exc
        if not codebook.values:
            raise ValueError("Prompt feature payload contains an empty audio codebook.")
        return codebook
    return None


def _text_condition(
    request: provider_pb2.SynthesizeRequest,
    segment: provider_pb2.TextSegment,
) -> common_pb2.TextCondition:
    ref_text = request.generation.fixed_prompt_text or request.prompt.prompt_text
    condition = common_pb2.TextCondition(
        text=segment.text.strip(),
        language=segment.lang or request.prompt.lang or "en",
        ref_text=ref_text,
        allow_missing_ref_text=True,
    )
    if "ref_text_mask_len" in request.generation.extra:
        condition.ref_text_mask_len = _uint(request.generation.extra["ref_text_mask_len"], "ref_text_mask_len")
    if "instruct" in request.generation.extra:
        condition.instruct = request.generation.extra["instruct"]
    elif segment.emotion:
        condition.instruct = segment.emotion
    return condition


def _generation_config(
    generation: provider_pb2.GenerationSpec,
    context: provider_pb2.RequestContext,
) -> common_pb2.GenerationConfig:
    config = common_pb2.GenerationConfig(
        request_timeout_sec=int(round(_timeout_from_context(context))),
        preprocess_prompt=True,
        postprocess_output=True,
    )
    if generation.HasField("duration"):
        config.duration = float(generation.duration)
    if generation.HasField("temperature"):
        config.temperature = float(generation.temperature)
    if generation.HasField("top_p"):
        config.top_p = float(generation.top_p)
    if generation.HasField("top_k"):
        config.top_k = int(generation.top_k)
    for key, value in generation.extra.items():
        _apply_generation_extra(config, key, value)
    return config


def _apply_generation_extra(config: common_pb2.GenerationConfig, key: str, value: str) -> None:
    if value == "":
        return
    if key in {"max_new_tokens", "num_steps", "top_k", "seed", "request_timeout_sec"}:
        setattr(config, key, _uint(value, key))
    elif key in {
        "cfg_scale",
        "emotion_guidance_scale",
        "nvv_guidance_scale",
        "t_shift",
        "layer_penalty_factor",
        "position_temperature",
        "class_temperature",
        "audio_chunk_duration",
        "audio_chunk_threshold",
        "duration",
        "speed",
    }:
        setattr(config, key, float(value))
    elif key in {"debug", "denoise", "preprocess_prompt", "postprocess_output"}:
        setattr(config, key, _bool(value))
    elif key == "debug_request_id":
        config.debug_request_id = value


def _uint(value: str, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _metadata_bool(metadata: dict[str, str], key: str, default: bool) -> bool:
    value = metadata.get(key)
    return default if value in {None, ""} else _bool(value)
