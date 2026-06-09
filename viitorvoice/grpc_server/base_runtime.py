from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from viitorvoice.grpc_server.config import ServiceConfig, clear_proxies


T = TypeVar("T")


class SingleWorkerRuntime:
    """Single-worker async facade around blocking GPU inference work."""

    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self._queue: asyncio.Queue[tuple[Callable[[], Any], asyncio.Future[Any]]] = asyncio.Queue(
            maxsize=config.max_queue_size
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        clear_proxies()
        self._worker_task = asyncio.create_task(self._worker_loop())
        self._started = True
        if self.config.warmup_on_start:
            await self.submit(self.warmup)

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None
        self._started = False

    async def submit(self, fn: Callable[[], T], timeout: float | None = None) -> T:
        if not self._started:
            await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        try:
            self._queue.put_nowait((fn, future))
        except asyncio.QueueFull as exc:
            raise RuntimeError("Runtime queue is full.") from exc
        return await asyncio.wait_for(
            future,
            timeout=self.config.request_timeout_sec if timeout is None else timeout,
        )

    async def _worker_loop(self) -> None:
        while True:
            fn, future = await self._queue.get()
            if future.cancelled():
                continue
            try:
                result = fn()
            except Exception as exc:
                if not future.cancelled():
                    future.set_exception(exc)
            else:
                if not future.cancelled():
                    future.set_result(result)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def warmup(self) -> None:
        return None
