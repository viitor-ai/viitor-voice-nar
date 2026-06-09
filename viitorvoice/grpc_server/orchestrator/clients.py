from __future__ import annotations

import grpc

from viitorvoice.grpc_server.config import OrchestratorTargets
from viitorvoice.grpc_server.proto import viitorvoice_decoder_pb2_grpc as decoder_pb2_grpc
from viitorvoice.grpc_server.proto import viitorvoice_encoder_pb2_grpc as encoder_pb2_grpc
from viitorvoice.grpc_server.proto import viitorvoice_llm_pb2_grpc as llm_pb2_grpc


GRPC_OPTIONS = [
    ("grpc.max_send_message_length", 512 * 1024 * 1024),
    ("grpc.max_receive_message_length", 512 * 1024 * 1024),
]


class ModuleClients:
    def __init__(self, targets: OrchestratorTargets) -> None:
        self.targets = targets
        self.encoder_channel = grpc.aio.insecure_channel(targets.encoder_target, options=GRPC_OPTIONS)
        self.llm_channel = grpc.aio.insecure_channel(targets.llm_target, options=GRPC_OPTIONS)
        self.decoder_channel = grpc.aio.insecure_channel(targets.decoder_target, options=GRPC_OPTIONS)
        self.encoder = encoder_pb2_grpc.ViiTorVoiceEncoderServiceStub(self.encoder_channel)
        self.llm = llm_pb2_grpc.ViiTorVoiceLLMServiceStub(self.llm_channel)
        self.decoder = decoder_pb2_grpc.ViiTorVoiceDecoderServiceStub(self.decoder_channel)

    async def close(self) -> None:
        await self.encoder_channel.close()
        await self.llm_channel.close()
        await self.decoder_channel.close()
