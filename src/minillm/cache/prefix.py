"""Prefix cache (M6): which full blocks in the paged pool hold KV for which token prefix (decision 8).

A full block is named by a SHA-256 chain: its hash commits to its 16 tokens and, through its parent's hash, to
every token before it, salted at the root with everything else that shapes the KV (model, dtype, layout, block
size, tenant salt). A new request whose leading blocks hash the same reuses those blocks instead of recomputing
their k/v; only the rest of its prompt is prefilled.

This file is the index: hash -> block id. Who holds a block (refcounts) and which unheld cached blocks to evict
first (LRU) is the block manager's job (`engine/block_manager.py`).
"""

from __future__ import annotations

import hashlib
import json
import struct

HASH_VERSION = "minillm-kv-v1"


def root_hash(extra_keys: dict) -> bytes:
    """SHA-256("minillm-kv-v1" || 0x00 || canonical JSON of the extra keys): the chain's salt."""
    canonical = json.dumps(extra_keys, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(HASH_VERSION.encode() + b"\x00" + canonical.encode()).digest()


def block_hash(parent: bytes, tokens: list[int]) -> bytes:
    """SHA-256(parent || tokens as uint32 little-endian), for one full block."""
    return hashlib.sha256(parent + struct.pack(f"<{len(tokens)}I", *tokens)).digest()


def extend_chain(chain: list[bytes], root: bytes, tokens: list[int], num_blocks: int, block_size: int) -> None:
    """Grow `chain` (the hashes of a sequence's first full blocks) to `num_blocks` entries."""
    for i in range(len(chain), num_blocks):
        chain.append(block_hash(chain[-1] if chain else root, tokens[i * block_size:(i + 1) * block_size]))


class PrefixCache:
    def __init__(self) -> None:
        self._block_of: dict[bytes, int] = {}
        self._hash_of: dict[int, bytes] = {}
        self.queries = 0  # prompt tokens looked up, since start (hit rate = hits / queries)
        self.hits = 0  # of those, tokens served from cached blocks

    def __len__(self) -> int:
        return len(self._block_of)

    def lookup(self, hashes: list[bytes]) -> list[int]:
        """The blocks of the longest cached prefix of `hashes`."""
        blocks = []
        for h in hashes:
            b = self._block_of.get(h)
            if b is None:
                break
            blocks.append(b)
        return blocks

    def insert(self, h: bytes, block: int) -> None:
        """Name a just-filled block. If another block already holds this prefix (two requests computed it at once),
        the first one stays the cached copy and this block remains private to its request."""
        if h not in self._block_of and block not in self._hash_of:
            self._block_of[h] = block
            self._hash_of[block] = h

    def is_cached(self, block: int) -> bool:
        return block in self._hash_of

    def forget(self, block: int) -> None:
        """The block is being reused for something else: drop its name."""
        h = self._hash_of.pop(block, None)
        if h is not None:
            del self._block_of[h]

    def clear(self) -> None:
        self._block_of.clear()
        self._hash_of.clear()
