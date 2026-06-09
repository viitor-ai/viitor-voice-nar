from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import grpc

from viitorvoice.grpc_server.config import clear_proxies
from viitorvoice.grpc_server.provider.servicer import MAX_GRPC_MESSAGE_BYTES
from viitorvoice.grpc_server.proto import backend_provider_pb2 as provider_pb2
from viitorvoice.grpc_server.proto import backend_provider_pb2_grpc as provider_pb2_grpc


async def main() -> None:
    clear_proxies()
    parser = argparse.ArgumentParser(description="Smoke client for the standard backend-provider service.")
    parser.add_argument("--target", default="127.0.0.1:50062")
    parser.add_argument("--mode", choices=["health", "capabilities"], default="capabilities")
    parser.add_argument("--output-dir", default="test_outputs/viitorvoice_grpc_server_provider_smoke")
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    async with grpc.aio.insecure_channel(
        args.target,
        options=[
            ("grpc.max_send_message_length", MAX_GRPC_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", MAX_GRPC_MESSAGE_BYTES),
        ],
    ) as channel:
        stub = provider_pb2_grpc.BackendProviderServiceStub(channel)
        if args.mode == "health":
            response = await stub.Health(
                provider_pb2.HealthRequest(context=provider_pb2.RequestContext(caller="provider_smoke")),
                timeout=args.timeout_sec,
            )
            summary = {
                "mode": args.mode,
                "status": response.status,
                "message": response.message,
                "backend_id": response.backend_id,
                "backend_version": response.backend_version,
            }
        else:
            response = await stub.GetCapabilities(
                provider_pb2.GetCapabilitiesRequest(context=provider_pb2.RequestContext(caller="provider_smoke")),
                timeout=args.timeout_sec,
            )
            capabilities = response.capabilities
            summary = {
                "mode": args.mode,
                "backend_id": capabilities.backend_id,
                "backend_version": capabilities.backend_version,
                "protocol_version": capabilities.protocol_version,
                "supports_unary": capabilities.supports_unary,
                "supports_true_streaming": capabilities.supports_true_streaming,
                "supports_prompt_features": capabilities.supports_prompt_features,
                "supported_output_formats": list(capabilities.supported_output_formats),
                "prompt_feature_specs": [
                    {
                        "feature_schema": spec.feature_schema,
                        "feature_version": spec.feature_version,
                        "model_version": spec.model_version,
                    }
                    for spec in capabilities.prompt_feature_specs
                ],
            }
    report = output_dir / f"{args.mode}.json"
    report.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "report": str(report)}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
