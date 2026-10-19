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

import torch
import zmq

from ..attention import get_backend
from ..cache.paged import BLOCK_SIZE, PagedKVCache
from ..device import select_device
from ..model.loader import load_model
from .core import DEFAULT_MAX_NUM_BATCHED_TOKENS, EngineCore
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
    kv_cache_gb: float | None = None,
    attention_backend: str = "torch_ref",
    gpu_memory_utilization: float = 0.9,
) -> None:
    info = select_device(device)
    model = load_model(model_dir, info.device, info.dtype, backend=get_backend(attention_backend))
    kv_bytes = kv_pool_bytes(model, info.device, kv_cache_gb, gpu_memory_utilization,
                             max_num_batched_tokens or DEFAULT_MAX_NUM_BATCHED_TOKENS)
    num_blocks = PagedKVCache.blocks_for_bytes(model.config, info.dtype, kv_bytes)
    core = EngineCore(model, max_model_len, step_log=step_log_path is not None, max_batch_size=max_batch_size,
                      max_num_batched_tokens=max_num_batched_tokens, num_kv_blocks=num_blocks)
    print(f"minillm engine: {attention_backend} attention, KV pool {kv_bytes / 2**30:.2f} GiB = {num_blocks} blocks "
          f"× {BLOCK_SIZE} tokens ({num_blocks * BLOCK_SIZE} tokens)", file=sys.stderr, flush=True)

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


def kv_pool_bytes(model, device, kv_cache_gb: float | None, utilization: float, profile_tokens: int) -> int:
    """The paged pool's size (decision 7). Given explicitly, or (CUDA) what's left of `utilization` × GPU memory
    after the weights and the activations of the largest step, measured with one profiling forward at startup.
    Elsewhere the default is 1 GiB."""
    if kv_cache_gb is not None:
        return int(kv_cache_gb * 2**30)
    if device.type != "cuda":
        return 2**30
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    _, total = torch.cuda.mem_get_info()
    # A throwaway pool just big enough for the profiling step, then a forward over `profile_tokens` tokens.
    probe = EngineCore(model, profile_tokens + 1, max_batch_size=1, num_kv_blocks=-(-(profile_tokens + 1) // BLOCK_SIZE),
                       max_num_batched_tokens=profile_tokens)
    probe.add_request(AddRequest("profile", [1] * profile_tokens, 1, ignore_eos=True))
    probe.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()  # weights + probe pool + activations
    probe_pool = sum(t.numel() * t.element_size() for t in probe.cache.k + probe.cache.v)
    del probe
    torch.cuda.empty_cache()
    free_for_kv = int(total * utilization) - (peak - probe_pool)
    if free_for_kv <= 0:
        raise RuntimeError(f"no GPU memory left for KV: weights + activations peak at {peak / 2**30:.1f} GiB")
    return free_for_kv


def _flush_step_log(core: EngineCore, path: str) -> None:
    with open(path, "a") as f:
        for rec in core.step_log:
            f.write(json.dumps(asdict(rec)) + "\n")
    core.step_log.clear()
