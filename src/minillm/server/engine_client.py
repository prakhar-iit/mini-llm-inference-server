"""The frontend's handle on the engine process: start it, send requests, route outputs back.

One background task reads every StepOutputs message from the engine and puts each RequestOutput on
the queue of the request it belongs to. `generate()` is an async iterator per request; if its consumer
goes away (client disconnected), the request is aborted so the engine frees its KV cache (R1).
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import tempfile
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import zmq
import zmq.asyncio

from ..engine.process import run_engine_process
from ..engine.protocol import Abort, AddRequest, EngineStats, Ready, RequestOutput, StepOutputs, decode, encode


class EngineDeadError(RuntimeError):
    pass


class EngineClient:
    def __init__(self, model_dir: str, device: str = "auto", max_model_len: int | None = None,
                 step_log_path: str | None = None) -> None:
        self._ipc_dir = tempfile.mkdtemp(prefix="minillm-")
        in_addr, out_addr = f"ipc://{self._ipc_dir}/in", f"ipc://{self._ipc_dir}/out"
        self._ctx = zmq.asyncio.Context()
        self._to_engine = self._ctx.socket(zmq.PUSH)
        self._to_engine.bind(in_addr)
        self._from_engine = self._ctx.socket(zmq.PULL)
        self._from_engine.bind(out_addr)
        # spawn, not fork: a forked child would inherit the parent's torch/MPS state.
        self._proc = mp.get_context("spawn").Process(
            target=run_engine_process, args=(model_dir, device, in_addr, out_addr, max_model_len, step_log_path),
            daemon=True, name="minillm-engine")
        self._proc.start()
        self._queues: dict[str, asyncio.Queue[RequestOutput]] = {}
        # Requests whose consumer left: the engine's final "abort" output still arrives and must be counted.
        self._abandoned: dict[str, Callable[[RequestOutput], None]] = {}
        self._reader: asyncio.Task | None = None
        self.stats = EngineStats(0, 0, 0)
        self.ready: Ready | None = None

    async def start(self, timeout_s: float = 600) -> Ready:
        """Wait for the engine to load the model and say Ready, then start routing outputs."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            if not self._proc.is_alive():
                raise EngineDeadError(f"engine process exited with code {self._proc.exitcode} while loading")
            if await self._from_engine.poll(500):
                msg = decode(await self._from_engine.recv())
                if isinstance(msg, Ready):
                    self.ready = msg
                    break
            if loop.time() > deadline:
                raise TimeoutError("engine did not become ready")
        self._reader = asyncio.create_task(self._read_outputs())
        return self.ready

    def is_alive(self) -> bool:
        return self._proc.is_alive()

    async def _read_outputs(self) -> None:
        while True:
            msg = decode(await self._from_engine.recv())
            if not isinstance(msg, StepOutputs):
                continue
            self.stats = msg.stats
            for out in msg.outputs:
                q = self._queues.get(out.req_id)
                if q is not None:
                    q.put_nowait(out)
                elif out.finished and out.req_id in self._abandoned:
                    self._abandoned.pop(out.req_id)(out)

    async def generate(self, req: AddRequest,
                       on_abandoned: Callable[[RequestOutput], None] | None = None) -> AsyncIterator[RequestOutput]:
        """Yield the request's outputs until it finishes. If the consumer stops early, the request is aborted
        and `on_abandoned` receives the engine's final output when it arrives (for metrics and logs)."""
        if not self._proc.is_alive():
            raise EngineDeadError("engine process is not running")
        q: asyncio.Queue[RequestOutput] = asyncio.Queue()
        self._queues[req.req_id] = q
        finished = False
        try:
            await self._to_engine.send(encode(req))
            while True:
                out = await q.get()
                finished = out.finished
                yield out
                if finished:
                    return
        finally:
            self._queues.pop(req.req_id, None)
            if not finished:
                # Consumer went away mid-request (client disconnect, cancellation). Send the abort without
                # awaiting: we may be inside a cancelled task, where awaiting is not allowed.
                if on_abandoned is not None:
                    self._abandoned[req.req_id] = on_abandoned
                self._to_engine.send(encode(Abort(req.req_id)), flags=zmq.NOBLOCK)

    async def shutdown(self) -> None:
        if self._reader:
            self._reader.cancel()
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=10)
        self._ctx.destroy(linger=0)
        for name in ("in", "out"):
            Path(self._ipc_dir, name).unlink(missing_ok=True)
        Path(self._ipc_dir).rmdir()
