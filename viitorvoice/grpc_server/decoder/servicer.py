from __future__ import annotations

import grpc

from viitorvoice.grpc_server import common
from viitorvoice.grpc_server.decoder.runtime import DecoderRuntime
from viitorvoice.grpc_server.proto import viitorvoice_common_pb2 as common_pb2
from viitorvoice.grpc_server.proto import viitorvoice_decoder_pb2 as decoder_pb2
from viitorvoice.grpc_server.proto import viitorvoice_decoder_pb2_grpc as decoder_pb2_grpc
from viitorvoice.grpc_server.servicer_utils import request_timeout, run_rpc


class ViiTorVoiceDecoderServicer(decoder_pb2_grpc.ViiTorVoiceDecoderServiceServicer):
    def __init__(self, runtime: DecoderRuntime, config) -> None:
        self.runtime = runtime
        self.config = config

    async def Health(
        self,
        request: common_pb2.HealthRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.HealthResponse:
        del context
        req_context = common.new_span(common.ensure_context(request.context, caller="decoder"))
        return self.runtime.runtime_info(req_context, "ready")

    async def DecodeAudio(
        self,
        request: decoder_pb2.DecodeAudioRequest,
        context: grpc.aio.ServicerContext,
    ) -> decoder_pb2.DecodeAudioResponse:
        async def invoke() -> decoder_pb2.DecodeAudioResponse:
            req_context = common.new_span(common.ensure_context(request.context, caller="decoder"))
            common.log_event(
                "module_request_received",
                req_context,
                service="decoder",
                rpc="DecodeAudio",
                input={
                    "chunks": len(request.audio_codebooks),
                    "audio_codebooks": [common.summarize_tensor(item) for item in request.audio_codebooks],
                    "postprocess_output": bool(request.postprocess_output),
                    "output_format": int(request.output_format),
                },
            )
            response = await self.runtime.submit(
                lambda: self.runtime.decode_audio(request, req_context),
                timeout=request_timeout(request.context, self.config.request_timeout_sec),
            )
            common.log_event("module_request_completed", req_context, service="decoder", rpc="DecodeAudio")
            return response

        return await run_rpc(context, invoke)


