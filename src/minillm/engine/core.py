"""Engine core: a FIFO queue and a step loop. M2 served one request at a time; M3 adds static batching.

Each `step()` does one unit of GPU work and returns the tokens it produced:
- no batch running and requests waiting -> admit up to `max_batch_size` of them, left-pad their prompts
  and prefill them together (one first token each);
- a batch running -> one decode step for every row (one token each).
Static: a batch runs until its LAST request finishes, and nothing joins it meanwhile. Finished rows keep
being computed (their tokens are discarded) and keep their KV memory until the whole batch ends. S3
measures that waste; continuous batching (M4) removes it. max_batch_size=1 is exactly M2's behaviour.

The core runs in-process (tests) or inside the engine process (`process.py`); it never touches text,
HTTP or ZMQ.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import torch

from ..generate import pad_left
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


@dataclass
class Batch:
    reqs: list[Request]
    cache: KVCache  # one cache for the whole batch: rows share a length (left-padded)
    active: list[bool]  # False once a row finished or was aborted; it is still computed until the batch ends


@dataclass
class StepRecord:
    """One row of the engine step log (decision 6)."""

    step: int
    t_start: int
    duration_ns: int
    num_seqs: int  # rows computed this step, finished-but-not-released rows included
    prefill_tokens: int  # real prompt tokens prefilled (padding excluded)
    decode_tokens: int  # tokens kept (rows still active); num_seqs - decode_tokens is wasted work
    kv_tokens_used: int
    queue_len: int


class EngineCore:
    def __init__(self, model: LlamaForCausalLM, max_model_len: int | None = None, step_log: bool = False,
                 max_batch_size: int = 1) -> None:
        self.model = model
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.max_model_len = max_model_len or model.config.max_position_embeddings
        self.max_batch_size = max_batch_size
        self.waiting: deque[Request] = deque()
        self.batch: Batch | None = None
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
        """Drop a request wherever it is. Unknown ids are ignored (already finished). In a static batch the row
        stops producing output, but its memory is only released when the whole batch ends."""
        b = self.batch
        row = next((i for i, r in enumerate(b.reqs) if r.req_id == req_id and b.active[i]), None) if b else None
        if row is not None:
            req = b.reqs[row]
            b.active[row] = False
            if not any(b.active):
                self.batch = None
        else:
            req = next((r for r in self.waiting if r.req_id == req_id), None)
            if req is None:
                return
            self.waiting.remove(req)
        self._aborted.append(self._finish(req, [], "abort"))

    # ---------- the loop ----------

    def has_work(self) -> bool:
        return self.batch is not None or bool(self.waiting) or bool(self._aborted)

    def stats(self) -> EngineStats:
        b = self.batch
        if b is None:
            return EngineStats(num_waiting=len(self.waiting), num_running=0, kv_tokens_used=0)
        # Finished rows still hold their KV until the batch ends: count every row's tokens.
        kv = sum(len(r.prompt) + len(r.output) for r in b.reqs)
        return EngineStats(num_waiting=len(self.waiting), num_running=sum(b.active), kv_tokens_used=kv)

    @torch.inference_mode()
    def step(self) -> StepOutputs:
        outputs, self._aborted = self._aborted, []
        t0 = now_ns()
        prefill_tokens = 0

        if self.batch is None and self.waiting:
            reqs = [self.waiting.popleft() for _ in range(min(self.max_batch_size, len(self.waiting)))]
            for r in reqs:
                r.timeline["scheduled"] = t0
            ids, mask = pad_left([r.prompt for r in reqs], self.device)
            cache = KVCache(self.model.config, len(reqs), ids.shape[1] + max(r.max_tokens for r in reqs),
                            self.device, self.dtype)
            logits = self.model(ids, kv_cache=cache, last_only=True, attention_mask=mask)
            b = self.batch = Batch(reqs, cache, [True] * len(reqs))
            prefill_tokens = sum(len(r.prompt) for r in reqs)
        elif self.batch is not None:
            b = self.batch
            # Every row feeds its newest token; rows that already finished are computed and discarded.
            ids = torch.tensor([[r.output[-1]] for r in b.reqs], device=self.device)
            logits = self.model(ids, kv_cache=b.cache, last_only=True)
        else:
            return StepOutputs(outputs, self.stats())

        tokens = logits[:, -1].argmax(dim=-1).tolist()  # .tolist() waits for the GPU: the step's real end
        t_tokens = now_ns()
        kept = 0
        for i, (req, token) in enumerate(zip(b.reqs, tokens)):
            if not b.active[i]:
                continue
            kept += 1
            req.output.append(token)
            if prefill_tokens:
                req.timeline["first_token"] = t_tokens
            if token in req.stop_ids or len(req.output) >= req.max_tokens:
                b.active[i] = False
                outputs.append(self._finish(req, [token], "stop" if token in req.stop_ids else "length"))
            else:
                outputs.append(RequestOutput(req.req_id, [token]))
        num_seqs = len(b.reqs)
        if not any(b.active):
            self.batch = None  # the whole batch is done: its KV memory is released at once

        if self.step_log is not None:
            self.step_log.append(StepRecord(self._step, t0, now_ns() - t0, num_seqs, prefill_tokens,
                                            0 if prefill_tokens else kept, self.stats().kv_tokens_used,
                                            len(self.waiting)))
        self._step += 1
        return StepOutputs(outputs, self.stats())

    def _finish(self, req: Request, new_tokens: list[int], reason: str) -> RequestOutput:
        req.timeline["finished"] = now_ns()
        return RequestOutput(req.req_id, new_tokens, finished=True, finish_reason=reason, timeline=req.timeline,
                             num_prompt_tokens=len(req.prompt), num_output_tokens=len(req.output))
