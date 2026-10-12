"""Engine core (M2): a FIFO queue and a step loop that serves ONE request at a time.

Each `step()` does one unit of GPU work and returns the tokens it produced:
- nothing running and a request waiting -> admit it and run its prefill (first token);
- a request running -> one decode step (next token).
Serving one request at a time is deliberate: S2 measures how the queue grows once requests arrive
faster than one can be served, which is what batching (M3) and continuous batching (M4) fix.

The core runs in-process (tests) or inside the engine process (`process.py`); it never touches text,
HTTP or ZMQ.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import torch

from ..model.kv_cache import KVCache
from ..model.llama import LlamaForCausalLM
from .protocol import AddRequest, EngineStats, RequestOutput, StepOutputs


def now_ns() -> int:
    # One monotonic clock for both processes on one machine (decision 6).
    return time.monotonic_ns()


@dataclass
class Request:
    req_id: str
    prompt: list[int]
    max_tokens: int
    stop_ids: frozenset[int]
    timeline: dict[str, int]
    output: list[int] = field(default_factory=list)
    cache: KVCache | None = None


@dataclass
class StepRecord:
    """One row of the engine step log (decision 6)."""

    step: int
    t_start: int
    duration_ns: int
    num_seqs: int
    prefill_tokens: int
    decode_tokens: int
    kv_tokens_used: int
    queue_len: int


class EngineCore:
    def __init__(self, model: LlamaForCausalLM, max_model_len: int | None = None, step_log: bool = False) -> None:
        self.model = model
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.max_model_len = max_model_len or model.config.max_position_embeddings
        self.waiting: deque[Request] = deque()
        self.running: Request | None = None
        self.step_log: list[StepRecord] | None = [] if step_log else None
        self._step = 0
        self._aborted: list[RequestOutput] = []

    # ---------- inputs ----------

    def add_request(self, msg: AddRequest) -> None:
        if not msg.prompt_token_ids:
            raise ValueError("empty prompt")
        if len(msg.prompt_token_ids) + msg.max_tokens > self.max_model_len:
            raise ValueError(f"prompt + max_tokens exceeds max_model_len {self.max_model_len}")
        timeline = dict(msg.timeline)
        timeline["queued"] = now_ns()
        stop = frozenset() if msg.ignore_eos else frozenset(msg.stop_token_ids)
        self.waiting.append(Request(msg.req_id, list(msg.prompt_token_ids), msg.max_tokens, stop, timeline))

    def abort(self, req_id: str) -> None:
        """Drop a request wherever it is; its KV cache is freed. Unknown ids are ignored (already finished)."""
        if self.running is not None and self.running.req_id == req_id:
            req, self.running = self.running, None
        else:
            req = next((r for r in self.waiting if r.req_id == req_id), None)
            if req is None:
                return
            self.waiting.remove(req)
        req.cache = None
        self._aborted.append(self._finish(req, [], "abort"))

    # ---------- the loop ----------

    def has_work(self) -> bool:
        return self.running is not None or bool(self.waiting) or bool(self._aborted)

    def stats(self) -> EngineStats:
        kv = self.running.cache.seq_len if self.running and self.running.cache else 0
        return EngineStats(num_waiting=len(self.waiting), num_running=int(self.running is not None), kv_tokens_used=kv)

    @torch.inference_mode()
    def step(self) -> StepOutputs:
        outputs, self._aborted = self._aborted, []
        t0 = now_ns()
        prefill_tokens = decode_tokens = 0

        if self.running is None and self.waiting:
            req = self.running = self.waiting.popleft()
            req.timeline["scheduled"] = t0
            req.cache = KVCache(self.model.config, 1, len(req.prompt) + req.max_tokens, self.device, self.dtype)
            ids = torch.tensor([req.prompt], device=self.device)
            logits = self.model(ids, kv_cache=req.cache, last_only=True)
            prefill_tokens = len(req.prompt)
        elif self.running is not None:
            req = self.running
            ids = torch.tensor([[req.output[-1]]], device=self.device)
            logits = self.model(ids, kv_cache=req.cache, last_only=True)
            decode_tokens = 1
        else:
            return StepOutputs(outputs, self.stats())

        token = int(logits[0, -1].argmax())  # .item() waits for the GPU: the step's real end
        req.output.append(token)
        if prefill_tokens:
            req.timeline["first_token"] = now_ns()

        if token in req.stop_ids or len(req.output) >= req.max_tokens:
            reason = "stop" if token in req.stop_ids else "length"
            self.running, req.cache = None, None
            outputs.append(self._finish(req, [token], reason))
        else:
            outputs.append(RequestOutput(req.req_id, [token]))

        if self.step_log is not None:
            self.step_log.append(StepRecord(self._step, t0, now_ns() - t0, 1, prefill_tokens, decode_tokens,
                                            self.stats().kv_tokens_used, len(self.waiting)))
        self._step += 1
        return StepOutputs(outputs, self.stats())

    def _finish(self, req: Request, new_tokens: list[int], reason: str) -> RequestOutput:
        req.timeline["finished"] = now_ns()
        return RequestOutput(req.req_id, new_tokens, finished=True, finish_reason=reason, timeline=req.timeline,
                             num_prompt_tokens=len(req.prompt), num_output_tokens=len(req.output))
