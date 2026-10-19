"""Engine core: a scheduler, a slot KV cache and a step loop. M2 served one request at a time, M3 batched
statically, M4 batches continuously (iteration-level, Orca).

Each `step()` asks the scheduler (decision 9) what to run, then does one unit of GPU work:
- requests waiting and a slot free -> prefill the newly admitted requests together (right-padded, one first
  token each). Running requests pause for this step;
- otherwise -> one decode step for every running request (one token each).
A request joins as soon as a slot is free and leaves the step it finishes; the request in the last slot moves
into the freed one, so running requests always fill slots [0, n). Every slot reserves max_model_len tokens of
KV, allocated once at startup, so an admitted request can never run out of memory. max_batch_size=1 is
exactly M2's behaviour. Static batching (M3) is at git tag m3, which S4 compares against.

The core runs in-process (tests) or inside the engine process (`process.py`); it never touches text,
HTTP or ZMQ.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from ..model.kv_cache import SlotKVCache
from ..model.llama import LlamaForCausalLM
from .protocol import AddRequest, EngineStats, RequestOutput, StepOutputs
from .scheduler import Request, Scheduler

DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192


def now_ns() -> int:
    # One monotonic clock for both processes on one machine (decision 6).
    return time.monotonic_ns()


@dataclass
class StepRecord:
    """One row of the engine step log (decision 6)."""

    step: int
    t_start: int
    duration_ns: int
    num_seqs: int  # rows computed this step
    prefill_tokens: int  # real prompt tokens prefilled (padding excluded)
    decode_tokens: int  # tokens kept on a decode step; equals num_seqs since M4 (no finished rows linger)
    kv_tokens_used: int
    queue_len: int


class EngineCore:
    def __init__(self, model: LlamaForCausalLM, max_model_len: int | None = None, step_log: bool = False,
                 max_batch_size: int = 1, max_num_batched_tokens: int | None = None) -> None:
        self.model = model
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.max_model_len = max_model_len or model.config.max_position_embeddings
        self.max_batch_size = max_batch_size
        self.scheduler = Scheduler(max_batch_size, max_num_batched_tokens or DEFAULT_MAX_NUM_BATCHED_TOKENS)
        self.cache = SlotKVCache(model.config, max_batch_size, self.max_model_len, self.device, self.dtype)
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
        self.scheduler.add(Request(msg.req_id, list(msg.prompt_token_ids), msg.max_tokens, stop, timeline))

    def abort(self, req_id: str) -> None:
        """Drop a request wherever it is; a running one frees its slot at once. Unknown ids are ignored
        (already finished)."""
        req = next((r for r in self.scheduler.running if r.req_id == req_id), None)
        if req is not None:
            self._release([req])
        else:
            req = self.scheduler.remove_waiting(req_id)
            if req is None:
                return
        self._aborted.append(self._finish(req, [], "abort"))

    # ---------- the loop ----------

    def has_work(self) -> bool:
        s = self.scheduler
        return bool(s.running) or bool(s.waiting) or bool(self._aborted)

    def stats(self) -> EngineStats:
        s = self.scheduler
        kv = sum(self.cache.lens[r.slot] for r in s.running)
        return EngineStats(num_waiting=len(s.waiting), num_running=len(s.running), kv_tokens_used=kv)

    @torch.inference_mode()
    def step(self) -> StepOutputs:
        outputs, self._aborted = self._aborted, []
        t0 = now_ns()
        plan = self.scheduler.schedule()
        prefill_tokens = 0

        if plan.prefill:
            reqs = plan.prefill
            for r in reqs:
                r.timeline["scheduled"] = t0
            lens = [len(r.prompt) for r in reqs]
            # Right-padded: each row's tokens land at [0, len) of its slot, where its decode steps continue.
            ids = torch.zeros(len(reqs), max(lens), dtype=torch.long)
            for i, r in enumerate(reqs):
                ids[i, :lens[i]] = torch.tensor(r.prompt)
            first = reqs[0].slot
            view = self.cache.batch(first, first + len(reqs), lens)
            logits = self.model(ids.to(self.device), kv_cache=view,
                                last_index=torch.tensor(lens, device=self.device) - 1)
            prefill_tokens = sum(lens)
        elif plan.decode:
            reqs = plan.decode
            ids = torch.tensor([[r.output[-1]] for r in reqs], device=self.device)
            logits = self.model(ids, kv_cache=self.cache.batch(0, len(reqs), [1] * len(reqs)), last_only=True)
        else:
            return StepOutputs(outputs, self.stats())

        tokens = logits[:, -1].argmax(dim=-1).tolist()  # .tolist() waits for the GPU: the step's real end
        t_tokens = now_ns()
        done = []
        for req, token in zip(reqs, tokens):
            req.output.append(token)
            if prefill_tokens:
                req.timeline["first_token"] = t_tokens
            if token in req.stop_ids or len(req.output) >= req.max_tokens:
                done.append(req)
                outputs.append(self._finish(req, [token], "stop" if token in req.stop_ids else "length"))
            else:
                outputs.append(RequestOutput(req.req_id, [token]))
        self._release(done)

        if self.step_log is not None:
            self.step_log.append(StepRecord(self._step, t0, now_ns() - t0, len(reqs), prefill_tokens,
                                            0 if prefill_tokens else len(reqs), self.stats().kv_tokens_used,
                                            len(self.scheduler.waiting)))
        self._step += 1
        return StepOutputs(outputs, self.stats())

    def _release(self, reqs: list[Request]) -> None:
        """Free the slots of requests that left, moving tail requests into the holes."""
        if not reqs:
            return
        slots = [r.slot for r in reqs]
        for src, dst in self.scheduler.release(reqs):
            self.cache.move(src, dst)
        for slot in slots:
            if slot >= len(self.scheduler.running):  # not refilled by a move
                self.cache.free(slot)

    def _finish(self, req: Request, new_tokens: list[int], reason: str) -> RequestOutput:
        req.timeline["finished"] = now_ns()
        return RequestOutput(req.req_id, new_tokens, finished=True, finish_reason=reason, timeline=req.timeline,
                             num_prompt_tokens=len(req.prompt), num_output_tokens=len(req.output))
