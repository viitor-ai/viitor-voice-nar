from __future__ import annotations

import asyncio
import traceback
from collections.abc import Awaitable, Callable
from typing import TypeVar

import grpc


T = TypeVar("T")


async def run_rpc(
    context: grpc.aio.ServicerContext,
    fn: Callable[[], Awaitable[T]],
) -> T:
    try:
        return await fn()
    except asyncio.TimeoutError as exc:
        await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, f"Request timed out: {exc}")
    except ValueError as exc:
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
    except FileNotFoundError as exc:
        await context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
    except RuntimeError as exc:
        if "queue is full" in str(exc).lower():
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        await context.abort(grpc.StatusCode.INTERNAL, detail)
    except grpc.aio.AioRpcError as exc:
        await context.abort(exc.code(), exc.details() or str(exc))
    except grpc.RpcError:
        raise
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        await context.abort(grpc.StatusCode.INTERNAL, detail)
    raise AssertionError("context.abort should have raised")


def request_timeout(config: object, default: float) -> float:
    if hasattr(config, "HasField"):
        try:
            if config.HasField("request_timeout_sec") and config.request_timeout_sec > 0:
                return float(config.request_timeout_sec)
        except ValueError:
            return float(default)
    return float(default)
