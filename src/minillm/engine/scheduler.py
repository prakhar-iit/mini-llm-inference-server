"""Iteration-level scheduler (M4, decision 9).

Every engine step asks `schedule()` what to run:
- requests waiting and a slot free -> a prefill step: admit FCFS, without skipping, while slots are free and
  the padded prefill fits `max_num_batched_tokens` (the head request is always admitted);
- otherwise -> a decode step for every running request.
Running requests occupy slots [0, n) in `running` order. When some leave, `release` moves requests from the
tail into the holes and returns the copies the KV cache must make, so the slots stay contiguous.
The scheduler only does bookkeeping: no tensors, no model.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class Request:
    req_id: str
    prompt: list[int]
    max_tokens: int
    stop_ids: frozenset[int]
    timeline: dict[str, int]
    output: list[int] = field(default_factory=list)
    slot: int = -1  # index into `running` and the KV cache while running; -1 while waiting


@dataclass
class SchedulerOutput:
    prefill: list[Request]  # admitted this step; their slots are contiguous and come after every running one
    decode: list[Request]  # every running request, in slot order (empty on a prefill step)


class Scheduler:
    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []  # running[i].slot == i

    def add(self, req: Request) -> None:
        self.waiting.append(req)

    def schedule(self) -> SchedulerOutput:
        free = self.max_num_seqs - len(self.running)
        if free and self.waiting:
            admitted: list[Request] = []
            longest = 0
            for req in self._admission_order():
                longest = max(longest, len(req.prompt))
                # Padded tokens: the prefill computes every row at the longest prompt's length.
                if admitted and (len(admitted) + 1) * longest > self.max_num_batched_tokens:
                    break  # FCFS without skipping: nothing behind the head-of-line request jumps it
                admitted.append(req)
                if len(admitted) == free:
                    break
            for req in admitted:
                self.waiting.remove(req)
                req.slot = len(self.running)
                self.running.append(req)
            return SchedulerOutput(prefill=admitted, decode=[])
        return SchedulerOutput(prefill=[], decode=list(self.running))

    def release(self, reqs: list[Request]) -> list[tuple[int, int]]:
        """Remove finished or aborted running requests; return the (src, dst) slot moves that keep slots contiguous."""
        gone = {id(r) for r in reqs}
        n = len(self.running) - len(gone)
        holes = sorted(r.slot for r in reqs if r.slot < n)
        movers = [r for r in self.running[n:] if id(r) not in gone]
        moves = []
        for req, hole in zip(movers, holes):
            moves.append((req.slot, hole))
            self.running[hole] = req
            req.slot = hole
        del self.running[n:]
        for r in reqs:
            r.slot = -1
        return moves

    def remove_waiting(self, req_id: str) -> Request | None:
        req = next((r for r in self.waiting if r.req_id == req_id), None)
        if req is not None:
            self.waiting.remove(req)
        return req

    def _admission_order(self) -> Iterable[Request]:
        """The priority hook (decision 9): the order waiting requests are considered in. FCFS today."""
        return self.waiting
