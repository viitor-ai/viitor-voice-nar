from __future__ import annotations

import argparse
import base64
import json
import logging
import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import grpc
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from viitorvoice.grpc_server import common as grpc_common
from viitorvoice.grpc_server.config import clear_proxies
from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2
from viitorvoice.grpc_server.proto import viitorvoice_orchestrator_pb2 as orch_pb2
from viitorvoice.grpc_server.proto import viitorvoice_orchestrator_pb2_grpc as orch_pb2_grpc


LOGGER = logging.getLogger("viitorvoice.inference.grpc_server.http")
MAX_MESSAGE_BYTES = 512 * 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ViiTorVoice HTTP gateway for gRPC v2.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--grpc-target", default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-warmup", action="store_true", help=argparse.SUPPRESS)
    return parser


def create_app(grpc_target: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        clear_proxies()
        channel = grpc.aio.insecure_channel(
            grpc_target,
            options=[
                ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
            ],
        )
        app.state.grpc_channel = channel
        app.state.orchestrator = orch_pb2_grpc.ViiTorVoiceOrchestratorServiceStub(channel)
        LOGGER.info("ViiTorVoice HTTP gateway connected to orchestrator target %s", grpc_target)
        try:
            yield
        finally:
            await channel.close(grace=5)

    app = FastAPI(title="ViiTorVoice gRPC v2 HTTP Gateway", version="inference-grpc-v2-http", lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        stub = _stub(app)
        try:
            response = await stub.Health(common_pb2.HealthRequest(context=_context()))
        except grpc.aio.AioRpcError as exc:
            raise _grpc_http_error(exc) from exc
        return JSONResponse(
            {
                "state": int(response.state),
                "message": response.message,
                "version": response.version,
                "active_backends": list(response.active_backends),
                "trace_id": response.context.trace_id,
            }
        )

    @app.post("/v1/voice-clone")
    async def voice_clone(
        text: str = Form(...),
        language: str = Form("en"),
        ref_audio: UploadFile | None = File(None),
        ref_audio_path: str | None = Form(None),
        ref_audio_base64: str | None = Form(None),
        ref_audio_codebook: str | None = Form(None),
        ref_audio_codebook_file: UploadFile | None = File(None),
        ref_text: str = Form(""),
        instruct: str = Form(""),
        allow_missing_ref_text: bool = Form(True),
        ref_text_mask_len: int = Form(10),
        sample_rate: int = Form(0),
        input_format: str = Form("wav"),
        output_format: str = Form("wav"),
        num_steps: int | None = Form(32),
        cfg_scale: float | None = Form(2.0),
        emotion_guidance_scale: float | None = Form(0.0),
        nvv_guidance_scale: float | None = Form(0.0),
        position_temperature: float | None = Form(1.0),
        class_temperature: float | None = Form(0.0),
        t_shift: float | None = Form(0.1),
        layer_penalty_factor: float | None = Form(5.0),
        duration: float | None = Form(None),
        speed: float | None = Form(None),
        preprocess_prompt: bool | None = Form(True),
        postprocess_output: bool | None = Form(True),
        timeout_sec: int = Form(600),
    ) -> StreamingResponse:
        codebook = await _parse_codebook(ref_audio_codebook, ref_audio_codebook_file, field_name="ref_audio_codebook")
        if codebook is None:
            audio = await _audio_input(ref_audio, ref_audio_path, ref_audio_base64, sample_rate, input_format, "ref_audio")
        else:
            audio = common_pb2.AudioInput()

        request = orch_pb2.SynthesizeRequest(
            context=_context(),
            condition=common_pb2.TextCondition(
                text=text,
                language=language,
                ref_text=ref_text,
                instruct=instruct,
                allow_missing_ref_text=allow_missing_ref_text,
                ref_text_mask_len=max(0, int(ref_text_mask_len)),
            ),
            ref_audio=audio,
            generation=_generation_config(
                num_steps=num_steps,
                cfg_scale=cfg_scale,
                emotion_guidance_scale=emotion_guidance_scale,
                nvv_guidance_scale=nvv_guidance_scale,
                position_temperature=position_temperature,
                class_temperature=class_temperature,
                t_shift=t_shift,
                layer_penalty_factor=layer_penalty_factor,
                duration=duration,
                speed=speed,
                preprocess_prompt=preprocess_prompt,
                postprocess_output=postprocess_output,
                timeout_sec=timeout_sec,
            ),
            output_format=_audio_format(output_format),
        )
        if codebook is not None:
            request.ref_audio_codebook.CopyFrom(codebook)
        grpc_common.log_event(
            "http_request_received",
            request.context,
            service="http",
            rpc="voice_clone",
            input={
                **grpc_common.summarize_text(request.condition),
                "has_ref_audio_codebook": codebook is not None,
                "ref_audio": grpc_common.summarize_audio_input(audio),
                "output_format": output_format,
                "generation": grpc_common.summarize_generation_config(request.generation),
            },
        )
        try:
            response = await _stub(app).Synthesize(request, timeout=float(timeout_sec))
        except grpc.aio.AioRpcError as exc:
            raise _grpc_http_error(exc) from exc
        grpc_common.log_event(
            "http_request_completed",
            request.context,
            service="http",
            rpc="voice_clone",
            output=grpc_common.summarize_audio_result(response.audio),
        )
        return _audio_response(response.audio, response.context.trace_id)

    @app.post("/v1/text-local-edit")
    async def text_local_edit(
        original_text: str = Form(...),
        edited_text: str = Form(...),
        language: str = Form("en"),
        source_audio: UploadFile | None = File(None),
        source_audio_path: str | None = Form(None),
        source_audio_base64: str | None = Form(None),
        sample_rate: int = Form(0),
        input_format: str = Form("wav"),
        output_format: str = Form("wav"),
        align_granularity: str = Form(""),
        padding_ms: float | None = Form(None),
        expand_mask_ratio: float | None = Form(None),
        length_mode: str | None = Form(None),
        manual_duration: float | None = Form(None),
        manual_frames: int | None = Form(None),
        length_scale: float | None = Form(None),
        min_mask_frames: int | None = Form(6),
        edit_context_frames: int | None = Form(40),
        edit_ref_context_frames: int | None = Form(120),
        preprocess_source_audio: bool | None = Form(None),
        postprocess_output: bool | None = Form(True),
        num_steps: int | None = Form(32),
        cfg_scale: float | None = Form(2.0),
        emotion_guidance_scale: float | None = Form(0.0),
        nvv_guidance_scale: float | None = Form(0.0),
        position_temperature: float | None = Form(1.0),
        class_temperature: float | None = Form(0.0),
        t_shift: float | None = Form(0.1),
        layer_penalty_factor: float | None = Form(5.0),
        timeout_sec: int = Form(900),
    ) -> StreamingResponse:
        request = orch_pb2.TextLocalEditRequest(
            context=_context(),
            source_audio=await _audio_input(
                source_audio,
                source_audio_path,
                source_audio_base64,
                sample_rate,
                input_format,
                "source_audio",
            ),
            original_text=original_text,
            edited_text=edited_text,
            language=language,
            generation=_generation_config(
                num_steps=num_steps,
                cfg_scale=cfg_scale,
                emotion_guidance_scale=emotion_guidance_scale,
                nvv_guidance_scale=nvv_guidance_scale,
                position_temperature=position_temperature,
                class_temperature=class_temperature,
                t_shift=t_shift,
                layer_penalty_factor=layer_penalty_factor,
                postprocess_output=postprocess_output,
                timeout_sec=timeout_sec,
            ),
            output_format=_audio_format(output_format),
            align_granularity=align_granularity,
        )
        _set_optional_float(request, "padding_ms", padding_ms)
        _set_optional_float(request, "expand_mask_ratio", expand_mask_ratio)
        _set_optional_str(request, "length_mode", length_mode)
        _set_optional_float(request, "manual_duration", manual_duration)
        _set_optional_int(request, "manual_frames", manual_frames)
        _set_optional_float(request, "length_scale", length_scale)
        _set_optional_int(request, "min_mask_frames", min_mask_frames)
        _set_optional_int(request, "edit_context_frames", edit_context_frames)
        _set_optional_int(request, "edit_ref_context_frames", edit_ref_context_frames)
        _set_optional_bool(request, "preprocess_source_audio", preprocess_source_audio)
        _set_optional_bool(request, "postprocess_output", postprocess_output)

        grpc_common.log_event(
            "http_request_received",
            request.context,
            service="http",
            rpc="text_local_edit",
            input={
                "original_text": request.original_text,
                "original_chars": len(request.original_text),
                "edited_text": request.edited_text,
                "edited_chars": len(request.edited_text),
                "language": request.language,
                "source_audio": grpc_common.summarize_audio_input(request.source_audio),
                "align_granularity": request.align_granularity,
                "edit_params": _summarize_text_local_edit_params(request),
                "generation": grpc_common.summarize_generation_config(request.generation),
            },
        )
        try:
            response = await _stub(app).TextLocalEdit(request, timeout=float(timeout_sec))
        except grpc.aio.AioRpcError as exc:
            raise _grpc_http_error(exc) from exc
        grpc_common.log_event(
            "http_request_completed",
            request.context,
            service="http",
            rpc="text_local_edit",
            output=grpc_common.summarize_audio_result(response.audio),
        )
        return _audio_response(response.audio, response.context.trace_id)

    return app


def _stub(app: FastAPI) -> orch_pb2_grpc.ViiTorVoiceOrchestratorServiceStub:
    return app.state.orchestrator


def _context() -> common_pb2.RequestContext:
    return grpc_common.new_span(
        grpc_common.ensure_context(common_pb2.RequestContext(caller="viitorvoice_grpc_server_http"), caller="viitorvoice_grpc_server_http")
    )


def _has_field(message: Any, field: str) -> bool:
    try:
        return bool(message.HasField(field))
    except ValueError:
        return False


def _summarize_text_local_edit_params(request: orch_pb2.TextLocalEditRequest) -> dict[str, Any]:
    fields = (
        "padding_ms",
        "expand_mask_ratio",
        "length_mode",
        "manual_duration",
        "manual_frames",
        "length_scale",
        "min_mask_frames",
        "edit_context_frames",
        "edit_ref_context_frames",
        "preprocess_source_audio",
        "postprocess_output",
    )
    return {field: getattr(request, field) for field in fields if _has_field(request, field)}


async def _audio_input(
    upload: UploadFile | None,
    audio_path: str | None,
    audio_base64: str | None,
    sample_rate: int,
    format_name: str,
    field_name: str,
) -> common_pb2.AudioInput:
    provided = sum(1 for value in (upload, audio_path, audio_base64) if value)
    if provided != 1:
        raise HTTPException(
            status_code=400,
            detail=f"Provide exactly one of {field_name}, {field_name}_path, or {field_name}_base64.",
        )
    if upload is not None:
        return common_pb2.AudioInput(
            audio_bytes=await upload.read(),
            sample_rate=max(0, int(sample_rate)),
            format=_audio_format(format_name),
        )
    if audio_base64:
        try:
            data = base64.b64decode(audio_base64, validate=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 for {field_name}_base64.") from exc
        return common_pb2.AudioInput(
            audio_bytes=data,
            sample_rate=max(0, int(sample_rate)),
            format=_audio_format(format_name),
        )
    path = str(Path(str(audio_path)).expanduser().resolve())
    return common_pb2.AudioInput(audio_path=path, sample_rate=max(0, int(sample_rate)), format=_audio_format(format_name))


async def _parse_codebook(
    codebook_json: str | None,
    codebook_file: UploadFile | None,
    *,
    field_name: str,
) -> common_pb2.Int64Tensor | None:
    if codebook_json and codebook_file is not None:
        raise HTTPException(status_code=400, detail=f"Provide only one of {field_name} or {field_name}_file.")
    if codebook_file is not None:
        codebook_json = (await codebook_file.read()).decode("utf-8")
    if not codebook_json:
        return None
    try:
        payload = json.loads(codebook_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} should be JSON.") from exc
    if isinstance(payload, dict) and "audio_codebook" in payload:
        payload = payload["audio_codebook"]
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail=f"{field_name} should contain values and shape.")
    values = payload.get("values")
    shape = payload.get("shape")
    if not isinstance(values, list) or not isinstance(shape, list):
        raise HTTPException(status_code=400, detail=f"{field_name} should contain list fields: values and shape.")
    return common_pb2.Int64Tensor(values=[int(item) for item in values], shape=[int(item) for item in shape])


def _generation_config(**kwargs: Any) -> common_pb2.GenerationConfig:
    config = common_pb2.GenerationConfig()
    mapping = {
        "num_steps": ("num_steps", int),
        "cfg_scale": ("cfg_scale", float),
        "emotion_guidance_scale": ("emotion_guidance_scale", float),
        "nvv_guidance_scale": ("nvv_guidance_scale", float),
        "position_temperature": ("position_temperature", float),
        "class_temperature": ("class_temperature", float),
        "t_shift": ("t_shift", float),
        "layer_penalty_factor": ("layer_penalty_factor", float),
        "duration": ("duration", float),
        "speed": ("speed", float),
        "preprocess_prompt": ("preprocess_prompt", bool),
        "postprocess_output": ("postprocess_output", bool),
        "timeout_sec": ("request_timeout_sec", int),
    }
    for input_name, (field_name, caster) in mapping.items():
        value = kwargs.get(input_name)
        if value is not None:
            setattr(config, field_name, caster(value))
    return config


def _audio_format(value: str) -> int:
    text = (value or "wav").strip().lower()
    if text in {"wav", "audio/wav", "audio/x-wav"}:
        return common_pb2.AUDIO_FORMAT_WAV
    if text in {"pcm", "pcm_s16le", "pcm-s16le", "audio/pcm"}:
        return common_pb2.AUDIO_FORMAT_PCM_S16LE
    if text in {"flac", "audio/flac"}:
        return common_pb2.AUDIO_FORMAT_FLAC
    raise HTTPException(status_code=400, detail=f"Unsupported audio format: {value!r}.")


def _media_type(audio: common_pb2.AudioResult) -> str:
    if audio.format == common_pb2.AUDIO_FORMAT_FLAC:
        return "audio/flac"
    if audio.format == common_pb2.AUDIO_FORMAT_PCM_S16LE:
        return "audio/L16"
    return "audio/wav"


def _audio_response(audio: common_pb2.AudioResult, trace_id: str) -> StreamingResponse:
    headers = {
        "X-ViiTorVoice-Trace-Id": trace_id,
        "X-ViiTorVoice-Sample-Rate": str(int(audio.sample_rate)),
        "X-ViiTorVoice-Duration-Sec": f"{float(audio.duration_sec):.6f}",
    }
    return StreamingResponse(BytesIO(audio.audio_bytes), media_type=_media_type(audio), headers=headers)


def _grpc_http_error(exc: grpc.aio.AioRpcError) -> HTTPException:
    status = 503
    if exc.code() in {grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.FAILED_PRECONDITION}:
        status = 400
    elif exc.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        status = 504
    return HTTPException(status_code=status, detail=f"{exc.code().name}: {exc.details()}")


def _set_optional_int(message: Any, field: str, value: int | None) -> None:
    if value is not None:
        setattr(message, field, int(value))


def _set_optional_float(message: Any, field: str, value: float | None) -> None:
    if value is not None:
        setattr(message, field, float(value))


def _set_optional_bool(message: Any, field: str, value: bool | None) -> None:
    if value is not None:
        setattr(message, field, bool(value))


def _set_optional_str(message: Any, field: str, value: str | None) -> None:
    if value:
        setattr(message, field, value)


def main() -> None:
    args = build_parser().parse_args()
    clear_proxies()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    host = args.host or os.environ.get("VIITORVOICE_HTTP_HOST", os.environ.get("VIITORVOICE_GRPC_HOST", "0.0.0.0"))
    port = int(args.port or os.environ.get("VIITORVOICE_HTTP_PORT", os.environ.get("VIITORVOICE_V2_HTTP_PORT", "51080")))
    grpc_target = args.grpc_target or os.environ.get(
        "VIITORVOICE_V2_HTTP_TARGET",
        os.environ.get("VIITORVOICE_V2_ORCH_TARGET", "127.0.0.1:50051"),
    )
    uvicorn.run(create_app(grpc_target), host=host, port=port, log_level=str(args.log_level).lower())


if __name__ == "__main__":
    main()
