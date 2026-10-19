"""Block manager (M5a): hands out and takes back the paged pool's blocks.

Only bookkeeping: a free list of block ids. Each request keeps its own block table (`Request.block_ids`); the
scheduler asks for blocks when a request is admitted or crosses into a new block, and gives them back when it
finishes, is aborted or is preempted. R3: free + held = total, with no block held twice.
"""

from __future__ import annotations

import math
from collections import deque


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: deque[int] = deque(range(num_blocks))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def blocks_for(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size)

    def allocate(self, n: int) -> list[int]:
        if n > len(self._free):
            raise RuntimeError(f"asked for {n} blocks, {len(self._free)} free")
        return [self._free.popleft() for _ in range(n)]

    def free(self, blocks: list[int]) -> None:
        self._free.extend(blocks)
        blocks.clear()

    def check(self, held: list[list[int]]) -> None:
        """R3: every block is either free or held by exactly one request."""
        owned = [b for table in held for b in table]
        everything = owned + list(self._free)
        if len(everything) != self.num_blocks or len(set(everything)) != self.num_blocks:
            raise AssertionError(f"block accounting broken: {len(owned)} held + {len(self._free)} free "
                                 f"!= {self.num_blocks} total, or a block is listed twice")
