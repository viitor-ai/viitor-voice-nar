from __future__ import annotations

import argparse
import asyncio
import logging

import grpc

from viitorvoice.grpc_server.config import ServiceConfig, V2RuntimeConfig, clear_proxies
from viitorvoice.grpc_server.provider.servicer import (
    MAX_GRPC_MESSAGE_BYTES,
    BackendProviderServicer,
    orchestrator_target_from_env,
)
from viitorvoice.grpc_server.proto import backend_provider_pb2_grpc as provider_pb2_grpc


LOGGER = logging.getLogger("viitorvoice.inference.grpc_server.provider")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Speech Edit standard backend-provider gRPC service.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--orchestrator-target", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


async def serve(config: V2RuntimeConfig, *, orchestrator_target: str) -> None:
    clear_proxies()
    servicer = BackendProviderServicer(orchestrator_target)
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", MAX_GRPC_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", MAX_GRPC_MESSAGE_BYTES),
        ]
    )
    provider_pb2_grpc.add_BackendProviderServiceServicer_to_server(servicer, server)
    address = f"{config.service.host}:{config.service.port}"
    server.add_insecure_port(address)
    await server.start()
    LOGGER.info(
        "Speech Edit backend provider listening on %s, orchestrator target %s",
        address,
        orchestrator_target,
    )
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=5)
        await servicer.close()


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    config = V2RuntimeConfig.from_env(default_port=50062)
    if args.host is not None or args.port is not None:
        service = config.service
        service = ServiceConfig(
            host=args.host or service.host,
            port=args.port if args.port is not None else service.port,
            device_id=service.device_id,
            max_queue_size=service.max_queue_size,
            request_timeout_sec=service.request_timeout_sec,
            warmup_on_start=False,
            debug_dump_dir=service.debug_dump_dir,
            llm=service.llm,
            encoder=service.encoder,
            decoder=service.decoder,
            aligner=service.aligner,
        )
        config = V2RuntimeConfig(service=service, targets=config.targets, log_json=config.log_json)
    target = args.orchestrator_target or orchestrator_target_from_env()
    try:
        asyncio.run(serve(config, orchestrator_target=target))
    except KeyboardInterrupt:
        LOGGER.info("Speech Edit backend provider stopped")


if __name__ == "__main__":
    main()
