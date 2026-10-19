"""Engine core: a scheduler, a paged KV pool and a step loop. M2 served one request at a time, M3 batched
statically, M4 continuously, M5a on paged KV.

Each `step()` asks the scheduler (decision 9) what to run, then does one unit of GPU work:
- requests waiting and room for one more -> prefill the newly admitted requests together;
- otherwise -> one decode step for every running request (one token each).
With prefix caching (M6, decision 8), a request whose leading blocks are already in the pool shares them and
prefills only the rest. A step's tokens are ragged (decision 7): every scheduled request's uncached tokens, concatenated into one row,
with `AttentionMetadata` saying which are whose and where their k/v live. A request holds only the blocks its
tokens fill; when the pool runs out, the scheduler preempts the newest request, which is later recomputed.
max_batch_size=1 is exactly M2's behaviour. Contiguous slots (M4) are at git tag m4, which S5 compares against.

The core runs in-process (tests) or inside the engine process (`process.py`); it never touches text,
HTTP or ZMQ.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from ..attention.metadata import AttentionMetadata, SeqStep
from ..model.llama import LlamaForCausalLM
from ..cache.paged import BLOCK_SIZE, PagedKVCache
from ..cache.prefix import PrefixCache, root_hash
from .block_manager import BlockManager
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
    num_seqs: int  # sequences computed this step
    prefill_tokens: int  # tokens prefilled (recomputed tokens of preempted requests included)
    decode_tokens: int  # tokens kept on a decode step; equals num_seqs since M4 (no finished rows linger)
    kv_tokens_used: int
    queue_len: int
    kv_blocks_used: int = 0
    num_preempted: int = 0  # requests preempted this step


class EngineCore:
    def __init__(self, model: LlamaForCausalLM, max_model_len: int | None = None, step_log: bool = False,
                 max_batch_size: int = 1, max_num_batched_tokens: int | None = None,
                 num_kv_blocks: int | None = None, enable_prefix_caching: bool = True,
                 model_id: str = "unknown") -> None:
        """`num_kv_blocks` defaults to the memory M4 reserved: max_batch_size × max_model_len tokens.
        `model_id` (name@revision) salts the prefix-cache hashes, with the KV dtype, layout and block size."""
        self.model = model
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.max_model_len = max_model_len or model.config.max_position_embeddings
        self.max_batch_size = max_batch_size
        if num_kv_blocks is None:
            num_kv_blocks = max_batch_size * -(-self.max_model_len // BLOCK_SIZE)
        self.cache = PagedKVCache(model.config, num_kv_blocks, self.device, self.dtype)
        self.prefix = PrefixCache() if enable_prefix_caching else None
        self.blocks = BlockManager(num_kv_blocks, BLOCK_SIZE, self.prefix)
        self.scheduler = Scheduler(max_batch_size, max_num_batched_tokens or DEFAULT_MAX_NUM_BATCHED_TOKENS,
                                   self.blocks, self.prefix)
        self._hash_keys = {"block_size": BLOCK_SIZE, "kv_dtype": str(self.dtype).removeprefix("torch."),
                           "layout": "NHD", "model": model_id}
        self.step_log: list[StepRecord] | None = [] if step_log else None
        self._step = 0
        self._aborted: list[RequestOutput] = []

    # ---------- inputs ----------

    def add_request(self, msg: AddRequest) -> None:
        if not msg.prompt_token_ids:
            raise ValueError("empty prompt")
        total = len(msg.prompt_token_ids) + msg.max_tokens
        if total > self.max_model_len:
            raise ValueError(f"prompt + max_tokens exceeds max_model_len {self.max_model_len}")
        if self.blocks.blocks_for(total) > self.blocks.num_blocks:
            raise ValueError(f"prompt + max_tokens needs more KV blocks than the pool has ({self.blocks.num_blocks})")
        timeline = dict(msg.timeline)
        timeline["queued"] = now_ns()
        stop = frozenset() if msg.ignore_eos else frozenset(msg.stop_token_ids)
        root = root_hash({**self._hash_keys, "salt": msg.cache_salt}) if self.prefix is not None else b""
        self.scheduler.add(Request(msg.req_id, list(msg.prompt_token_ids), msg.max_tokens, stop, timeline,
                                   root=root))

    def abort(self, req_id: str) -> None:
        """Drop a request wherever it is; a running one frees its blocks at once. Unknown ids are ignored
        (already finished)."""
        req = next((r for r in self.scheduler.running if r.req_id == req_id), None)
        if req is not None:
            self.scheduler.finish([req])
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
        return EngineStats(num_waiting=len(s.waiting), num_running=len(s.running),
                           kv_tokens_used=sum(r.num_cached for r in s.running),
                           kv_blocks_used=self.blocks.num_blocks - self.blocks.num_free,
                           kv_blocks_total=self.blocks.num_blocks, num_preemptions=s.num_preemptions,
                           kv_blocks_cached=self.blocks.num_cached,
                           prefix_cache_queries=self.prefix.queries if self.prefix else 0,
                           prefix_cache_hits=self.prefix.hits if self.prefix else 0)

    def check_blocks(self) -> None:
        """R3: every block is free, cached or held, and held refcounts match the running requests' tables."""
        self.blocks.check([r.block_ids for r in self.scheduler.running])

    def clear_prefix_cache(self) -> None:
        """Drop every cached block nobody holds (R3: an idle engine then has every block free)."""
        if self.prefix is not None:
            self.blocks.clear_cache()

    @torch.inference_mode()
    def step(self) -> StepOutputs:
        outputs, self._aborted = self._aborted, []
        t0 = now_ns()
        plan = self.scheduler.schedule()
        reqs = plan.prefill or plan.decode
        if not reqs:
            return StepOutputs(outputs, self.stats())
        for r in plan.prefill:
            r.timeline.setdefault("scheduled", t0)  # a preempted request keeps its first admission time

        new = [r.uncached() for r in reqs]
        meta = AttentionMetadata.build([SeqStep(r.num_cached, len(n), r.block_ids) for r, n in zip(reqs, new)],
                                       BLOCK_SIZE, self.device)
        ids = torch.tensor([[t for n in new for t in n]], device=self.device)  # [1, N]: ragged, no padding
        logits = self.model(ids, kv_cache=self.cache.step(meta), last_index=meta.logits_index)  # [S, 1, V]
        tokens = logits[:, -1].argmax(dim=-1).tolist()  # .tolist() waits for the GPU: the step's real end
        t_tokens = now_ns()

        done = []
        for req, token in zip(reqs, tokens):
            req.num_cached = req.num_tokens
            req.output.append(token)
            req.timeline.setdefault("first_token", t_tokens)
            if token in req.stop_ids or len(req.output) >= req.max_tokens:
                done.append(req)
                outputs.append(self._finish(req, [token], "stop" if token in req.stop_ids else "length"))
            else:
                outputs.append(RequestOutput(req.req_id, [token]))
        self.scheduler.name_full_blocks(reqs)  # before finishing: freed blocks with a name stay cached
        self.scheduler.finish(done)

        if self.step_log is not None:
            st = self.stats()
            prefill_tokens = sum(len(n) for n in new) if plan.prefill else 0
            self.step_log.append(StepRecord(self._step, t0, now_ns() - t0, len(reqs), prefill_tokens,
                                            0 if plan.prefill else len(reqs), st.kv_tokens_used, st.num_waiting,
                                            st.kv_blocks_used, len(plan.preempted)))
        self._step += 1
        return StepOutputs(outputs, self.stats())

    def _finish(self, req: Request, new_tokens: list[int], reason: str) -> RequestOutput:
        req.timeline["finished"] = now_ns()
        return RequestOutput(req.req_id, new_tokens, finished=True, finish_reason=reason, timeline=req.timeline,
                             num_prompt_tokens=len(req.prompt), num_output_tokens=len(req.output),
                             num_cached_tokens=req.num_cached_tokens or 0)
