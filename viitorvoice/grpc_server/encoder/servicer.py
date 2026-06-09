from __future__ import annotations

import grpc

from viitorvoice.grpc_server import common
from viitorvoice.grpc_server.encoder.runtime import EncoderRuntime
from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2
from viitorvoice.grpc_server.proto import viitorvoice_encoder_pb2 as encoder_pb2
from viitorvoice.grpc_server.proto import viitorvoice_encoder_pb2_grpc as encoder_pb2_grpc
from viitorvoice.grpc_server.servicer_utils import request_timeout, run_rpc


class ViiTorVoiceEncoderServicer(encoder_pb2_grpc.ViiTorVoiceEncoderServiceServicer):
    def __init__(self, runtime: EncoderRuntime, config) -> None:
        self.runtime = runtime
        self.config = config

    async def Health(
        self,
        request: common_pb2.HealthRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.HealthResponse:
        del context
        req_context = common.new_span(common.ensure_context(request.context, caller="encoder"))
        return self.runtime.runtime_info(req_context, "ready")

    async def EncodeAudio(
        self,
        request: encoder_pb2.EncodeAudioRequest,
        context: grpc.aio.ServicerContext,
    ) -> encoder_pb2.EncodeAudioResponse:
        async def invoke() -> encoder_pb2.EncodeAudioResponse:
            req_context = common.new_span(common.ensure_context(request.context, caller="encoder"))
            common.log_event(
                "module_request_received",
                req_context,
                service="encoder",
                rpc="EncodeAudio",
                input={
                    **common.summarize_audio_input(request.audio),
                    "preprocess_prompt": bool(request.preprocess_prompt),
                    "return_audio": bool(request.return_audio),
                    "can_trim_long_audio": bool(request.can_trim_long_audio),
                },
            )
            response = await self.runtime.submit(
                lambda: self.runtime.encode_audio(request, req_context),
                timeout=request_timeout(request.context, self.config.request_timeout_sec),
            )
            common.log_event("module_request_completed", req_context, service="encoder", rpc="EncodeAudio")
            return response

        return await run_rpc(context, invoke)


