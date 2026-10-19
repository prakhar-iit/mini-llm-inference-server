"""Iteration-level scheduler (M4, decision 9) on paged KV (M5a).

Every engine step asks `schedule()` what to run:
- requests waiting and room for one more -> a prefill step: admit FCFS, without skipping, while
  - fewer than `max_num_seqs` run,
  - the step's prompt tokens fit `max_num_batched_tokens` (the head request is always admitted; no padding
    since M5, so real tokens are counted),
  - the pool has blocks for the request's tokens, plus one for every request (running, just admitted, or this
    one) whose next token starts a new block, so admitting it doesn't force a preemption on the next decode step;
- otherwise -> a decode step for every running request. A request whose next token starts a new block gets one;
  when none is free, the newest running request is preempted: its blocks are freed and it goes back to the front
  of the queue, to be recomputed (prompt + output so far) when it is admitted again.
M6, prefix caching (decision 8): at admission a request looks up the hashes of its leading full blocks; the
cached ones are shared (a reference each) instead of recomputed, so only the rest of the prompt is prefilled and
counted against the token budget. After every step, blocks that just filled are named in the prefix cache.
The scheduler only does bookkeeping: block ids, no tensors, no model.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..cache.prefix import PrefixCache, extend_chain
from .block_manager import BlockManager


@dataclass(eq=False)  # compared by identity: two requests with equal fields are still two requests
class Request:
    req_id: str
    prompt: list[int]
    max_tokens: int
    stop_ids: frozenset[int]
    timeline: dict[str, int]
    output: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)  # its block table, while it holds KV
    num_cached: int = 0  # tokens whose k/v are in the pool
    num_preemptions: int = 0
    root: bytes = b""  # M6: the hash chain's salted root (decision 8)
    block_hashes: list[bytes] = field(default_factory=list)  # hashes of its first full blocks, computed so far
    num_named: int = 0  # leading blocks of its table already named in the prefix cache
    num_cached_tokens: int | None = None  # prompt tokens served from the prefix cache at its first admission

    @property
    def num_tokens(self) -> int:
        return len(self.prompt) + len(self.output)

    def uncached(self) -> list[int]:
        """The tokens this step must compute: the whole prompt (and output, after a preemption) on a prefill,
        the newest token on a decode step."""
        tokens = self.prompt + self.output
        return tokens[self.num_cached:]


@dataclass
class SchedulerOutput:
    prefill: list[Request]  # admitted this step, in admission order
    decode: list[Request]  # every running request (empty on a prefill step)
    preempted: list[Request] = field(default_factory=list)  # sent back to the queue this step


class Scheduler:
    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int, blocks: BlockManager,
                 prefix: PrefixCache | None = None) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.blocks = blocks
        self.prefix = prefix
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []  # admission order: the newest is last
        self.num_preemptions = 0

    def add(self, req: Request) -> None:
        self.waiting.append(req)

    def schedule(self) -> SchedulerOutput:
        admitted = self._admit()
        if admitted:
            return SchedulerOutput(prefill=admitted, decode=[])
        preempted = self._grow_running()
        return SchedulerOutput(prefill=[], decode=list(self.running), preempted=preempted)

    def _admit(self) -> list[Request]:
        admitted: list[Request] = []
        tokens = 0
        bs = self.blocks.block_size
        # Blocks the next decode step will need: a running request whose next token starts a new block.
        # (Running requests don't decode during this prefill step, so their next write is still at num_cached.)
        reserve = sum(1 for r in self.running if r.num_cached >= len(r.block_ids) * bs)
        for req in self._admission_order():
            if len(self.running) == self.max_num_seqs:
                break
            n = req.num_tokens
            hits = self._lookup(req)
            new_tokens = n - len(hits) * bs  # what the prefill actually computes
            if admitted and tokens + new_tokens > self.max_num_batched_tokens:
                break
            total = self.blocks.blocks_for(n)
            need = total - len(hits)
            revive = sum(1 for b in hits if self.blocks.is_cached_unheld(b))  # cached blocks nobody holds count as free
            spare = 1 if n == total * bs else 0  # after its prefill, its next token starts a new block
            if need + revive + spare + reserve > self.blocks.num_free:
                break  # FCFS without skipping: nothing behind the head-of-line request jumps it
            self.blocks.acquire(hits)  # before allocating, so these can't be evicted to make room
            req.block_ids = hits + self.blocks.allocate(need)
            req.num_cached, req.num_named = len(hits) * bs, len(hits)
            if self.prefix is not None:
                self.prefix.queries += n
                self.prefix.hits += len(hits) * bs
            if req.num_cached_tokens is None:
                req.num_cached_tokens = len(hits) * bs
            reserve += spare
            tokens += new_tokens
            admitted.append(req)
            self.running.append(req)
        for req in admitted:
            self.waiting.remove(req)
        return admitted

    def _lookup(self, req: Request) -> list[int]:
        """Cached blocks for the request's longest cached prefix. Only blocks entirely within its first
        num_tokens - 1 tokens count: the last token is always computed, so the step has logits to sample from."""
        if self.prefix is None:
            return []
        bs = self.blocks.block_size
        k = (req.num_tokens - 1) // bs
        if k == 0:
            return []
        extend_chain(req.block_hashes, req.root, req.prompt + req.output, k, bs)
        return self.prefix.lookup(req.block_hashes[:k])

    def name_full_blocks(self, reqs: list[Request]) -> None:
        """After a step: name every block that just filled, so later requests can share it."""
        if self.prefix is None:
            return
        bs = self.blocks.block_size
        for req in reqs:
            full = req.num_cached // bs
            if full <= req.num_named:
                continue
            extend_chain(req.block_hashes, req.root, req.prompt + req.output, full, bs)
            for j in range(req.num_named, full):
                self.prefix.insert(req.block_hashes[j], req.block_ids[j])
            req.num_named = full

    def _grow_running(self) -> list[Request]:
        """Give every running request room for its next token, preempting the newest when the pool is empty."""
        preempted: list[Request] = []
        for req in list(self.running):  # oldest first
            if req not in self.running:  # preempted earlier in this loop
                continue
            if req.num_cached < len(req.block_ids) * self.blocks.block_size:
                continue  # its last block still has room
            while self.blocks.num_free == 0:
                victim = self.running[-1]
                self._preempt(victim)
                preempted.append(victim)
                if victim is req:
                    break
            if req in self.running:
                req.block_ids += self.blocks.allocate(1)
        return preempted

    def _preempt(self, req: Request) -> None:
        self.running.remove(req)
        self.blocks.free(req.block_ids)  # named blocks stay cached: a resume may find its own prefix again
        req.num_cached = req.num_named = 0
        req.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(req)  # it arrived before everything still waiting

    def finish(self, reqs: list[Request]) -> None:
        """Remove finished or aborted running requests and free their blocks."""
        for req in reqs:
            self.running.remove(req)
            self.blocks.free(req.block_ids)

    def remove_waiting(self, req_id: str) -> Request | None:
        req = next((r for r in self.waiting if r.req_id == req_id), None)
        if req is not None:
            self.waiting.remove(req)
        return req

    def _admission_order(self) -> Iterable[Request]:
        """The priority hook (decision 9): the order waiting requests are considered in. FCFS today."""
        return list(self.waiting)
