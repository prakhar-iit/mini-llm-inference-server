"""Block manager (M5a, refcounts and LRU from M6): hands out and takes back the paged pool's blocks.

Every block is in exactly one state:
- free: no KV worth keeping, ready to hand out;
- held: refcount ≥ 1, one per request whose block table lists it (M6: requests sharing a cached prefix share
  its blocks);
- cached: refcount 0 but named in the prefix cache, so a later request may reuse its KV. Kept in LRU order and
  handed out (evicted, its name dropped) only when no free block is left.
Each request keeps its own block table (`Request.block_ids`). R3: these states account for every block, and a
held block's refcount equals the number of tables listing it.
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict, deque

from ..cache.prefix import PrefixCache


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, prefix: PrefixCache | None = None) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.prefix = prefix
        self._free: deque[int] = deque(range(num_blocks))
        self._cached: OrderedDict[int, None] = OrderedDict()  # refcount 0, still named; oldest first
        self._ref = [0] * num_blocks

    @property
    def num_free(self) -> int:
        """Blocks that can be handed out: free, plus cached ones that would be evicted."""
        return len(self._free) + len(self._cached)

    @property
    def num_cached(self) -> int:
        return len(self._cached)

    def blocks_for(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size)

    def is_cached_unheld(self, block: int) -> bool:
        return block in self._cached

    def allocate(self, n: int) -> list[int]:
        if n > self.num_free:
            raise RuntimeError(f"asked for {n} blocks, {self.num_free} free")
        out = []
        for _ in range(n):
            if self._free:
                b = self._free.popleft()
            else:  # evict the least recently used cached block
                b, _ = self._cached.popitem(last=False)
                self.prefix.forget(b)
            self._ref[b] = 1
            out.append(b)
        return out

    def acquire(self, blocks: list[int]) -> None:
        """Take a reference on cached blocks (a prefix-cache hit)."""
        for b in blocks:
            if self._ref[b] == 0:
                del self._cached[b]
            self._ref[b] += 1

    def free(self, blocks: list[int]) -> None:
        """Drop one reference on each block; a block nobody holds becomes cached (if named) or free.

        A block table is freed tail first, so a sequence's last blocks are the first to be evicted: evicting its
        first block would make every later block unreachable, since a lookup stops at the first miss."""
        for b in reversed(blocks):
            self._ref[b] -= 1
            if self._ref[b] == 0:
                if self.prefix is not None and self.prefix.is_cached(b):
                    self._cached[b] = None  # most recently used: evicted last
                else:
                    self._free.append(b)
        blocks.clear()

    def clear_cache(self) -> None:
        """Evict every cached block nobody holds (R3: afterwards, an idle engine has every block free)."""
        while self._cached:
            b, _ = self._cached.popitem(last=False)
            self.prefix.forget(b)
            self._free.append(b)

    def check(self, held: list[list[int]]) -> None:
        """R3: every block is free, cached or held, exactly once, and held refcounts match the block tables."""
        owners = Counter(b for table in held for b in table)
        problems = [b for b in range(self.num_blocks) if self._ref[b] != owners.get(b, 0)]
        if problems:
            raise AssertionError(f"refcounts don't match block tables for blocks {problems[:10]}")
        states = [b for b in owners] + list(self._free) + list(self._cached)
        if len(states) != self.num_blocks or len(set(states)) != self.num_blocks:
            raise AssertionError(f"block accounting broken: {len(owners)} held + {len(self._free)} free + "
                                 f"{len(self._cached)} cached != {self.num_blocks} total, or a block is listed twice")
