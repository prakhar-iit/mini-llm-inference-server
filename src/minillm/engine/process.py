"""The engine core's own process (decision 1).

Loop: drain incoming messages (non-blocking while there is work, a short blocking poll when idle),
run one engine step, send its outputs. Nothing here touches HTTP or text, so the GPU loop never waits
on the frontend.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import zmq

from ..device import select_device
from ..model.loader import load_model
from .core import EngineCore
from .protocol import Abort, AddRequest, Ready, RequestOutput, StepOutputs, decode, encode

IDLE_POLL_MS = 50


def run_engine_process(
    model_dir: str,
    device: str,
    in_addr: str,
    out_addr: str,
    max_model_len: int | None,
    step_log_path: str | None,
    max_batch_size: int = 1,
    max_num_batched_tokens: int | None = None,
) -> None:
    info = select_device(device)
    model = load_model(model_dir, info.device, info.dtype)
    core = EngineCore(model, max_model_len, step_log=step_log_path is not None, max_batch_size=max_batch_size,
                      max_num_batched_tokens=max_num_batched_tokens)
    # Every slot reserves max_model_len tokens up front (decision 9); say how much, since it can rival the weights.
    kv_bytes = sum(t.numel() * t.element_size() for t in core.cache.k + core.cache.v)
    print(f"minillm engine: KV cache reserved {kv_bytes / 2**30:.2f} GiB = {max_batch_size} slots × "
          f"{core.max_model_len} tokens", file=sys.stderr, flush=True)

    ctx = zmq.Context()
    inbox = ctx.socket(zmq.PULL)
    inbox.connect(in_addr)
    outbox = ctx.socket(zmq.PUSH)
    outbox.connect(out_addr)
    outbox.send(encode(Ready(model=Path(model_dir).name, device=info.name, dtype=str(info.dtype))))

    while True:
        timeout = 0 if core.has_work() else IDLE_POLL_MS
        while inbox.poll(timeout):
            msg = decode(inbox.recv())
            if isinstance(msg, AddRequest):
                try:
                    core.add_request(msg)
                except ValueError as e:  # the frontend validates first; this is a backstop
                    outbox.send(encode(StepOutputs(
                        [RequestOutput(msg.req_id, [], finished=True, finish_reason=f"error: {e}")], core.stats())))
            elif isinstance(msg, Abort):
                core.abort(msg.req_id)
            timeout = 0  # keep draining whatever already arrived, then step

        if core.has_work():
            outbox.send(encode(core.step()))
        elif core.step_log:
            _flush_step_log(core, step_log_path)  # only when idle: file writes stay off the hot path


def _flush_step_log(core: EngineCore, path: str) -> None:
    with open(path, "a") as f:
        for rec in core.step_log:
            f.write(json.dumps(asdict(rec)) + "\n")
    core.step_log.clear()
