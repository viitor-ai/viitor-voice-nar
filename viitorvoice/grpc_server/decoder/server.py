from __future__ import annotations

import argparse
import asyncio
import logging

import grpc

from viitorvoice.grpc_server.config import ServiceConfig, V2RuntimeConfig, clear_proxies
from viitorvoice.grpc_server.decoder.runtime import DecoderRuntime
from viitorvoice.grpc_server.decoder.servicer import ViiTorVoiceDecoderServicer
from viitorvoice.grpc_server.proto import viitorvoice_decoder_pb2_grpc as decoder_pb2_grpc


LOGGER = logging.getLogger("viitorvoice.inference.grpc_server.decoder")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ViiTorVoice decoder gRPC v2 service.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


async def serve(config: V2RuntimeConfig) -> None:
    clear_proxies()
    runtime = DecoderRuntime(config.service)
    await runtime.start()
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", 512 * 1024 * 1024),
            ("grpc.max_receive_message_length", 512 * 1024 * 1024),
        ]
    )
    decoder_pb2_grpc.add_ViiTorVoiceDecoderServiceServicer_to_server(
        ViiTorVoiceDecoderServicer(runtime, config.service),
        server,
    )
    address = f"{config.service.host}:{config.service.port}"
    server.add_insecure_port(address)
    await server.start()
    LOGGER.info("ViiTorVoice decoder gRPC v2 service listening on %s", address)
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=5)
        await runtime.stop()


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    config = V2RuntimeConfig.from_env(default_port=51053)
    if args.host is not None or args.port is not None or args.no_warmup:
        service = config.service
        service = ServiceConfig(
            host=args.host or service.host,
            port=args.port if args.port is not None else service.port,
            device_id=service.device_id,
            max_queue_size=service.max_queue_size,
            request_timeout_sec=service.request_timeout_sec,
            warmup_on_start=False if args.no_warmup else service.warmup_on_start,
            debug_dump_dir=service.debug_dump_dir,
            llm=service.llm,
            encoder=service.encoder,
            decoder=service.decoder,
            aligner=service.aligner,
        )
        config = V2RuntimeConfig(service=service, targets=config.targets, log_json=config.log_json)
    try:
        asyncio.run(serve(config))
    except KeyboardInterrupt:
        LOGGER.info("ViiTorVoice decoder gRPC v2 service stopped")


if __name__ == "__main__":
    main()
