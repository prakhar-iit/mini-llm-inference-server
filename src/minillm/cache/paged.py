"""Paged KV cache (M5a, decision 7).

The pool: per layer, separate K and V tensors of shape [num_blocks, block_size, Hkv, Dh] (NHD: token-major
inside a block), allocated once. A request owns a list of block ids (its block table, kept by the engine); its
token at position p lives at block_ids[p // block_size], offset p % block_size, in every layer. Blocks are
handed out and taken back by `engine/block_manager.py`; this file only holds the memory.

Compared with M4's slots (`slot.py`), nothing is reserved per request: a request holds
ceil(tokens / block_size) blocks, so the waste is at most one partly filled block per request (S5).

`PagedStep` is the cache as one forward pass sees it: the pool plus that step's `AttentionMetadata`.
"""

from __future__ import annotations

import torch

from ..attention.metadata import AttentionMetadata
from ..model.config import LlamaConfig

BLOCK_SIZE = 16


class PagedKVCache:
    def __init__(self, config: LlamaConfig, num_blocks: int, device: torch.device | str, dtype: torch.dtype,
                 block_size: int = BLOCK_SIZE) -> None:
        shape = (num_blocks, block_size, config.num_key_value_heads, config.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.num_blocks = num_blocks
        self.block_size = block_size

    @staticmethod
    def bytes_per_block(config: LlamaConfig, dtype: torch.dtype, block_size: int = BLOCK_SIZE) -> int:
        """K and V, every layer: 2 × layers × block_size × KV heads × head_dim × bytes per element."""
        elem = torch.tensor([], dtype=dtype).element_size()
        return 2 * config.num_hidden_layers * block_size * config.num_key_value_heads * config.head_dim * elem

    @classmethod
    def blocks_for_bytes(cls, config: LlamaConfig, dtype: torch.dtype, num_bytes: int,
                         block_size: int = BLOCK_SIZE) -> int:
        return num_bytes // cls.bytes_per_block(config, dtype, block_size)

    def step(self, meta: AttentionMetadata) -> PagedStep:
        return PagedStep(self, meta)


class PagedStep:
    """The methods `LlamaModel` calls on a KV cache, for one ragged step. Attention reads and writes the pool
    itself (through the backend's `forward`), so there is nothing to mark or advance: the engine tracks how
    many tokens each request has cached."""

    ragged = True

    def __init__(self, cache: PagedKVCache, meta: AttentionMetadata) -> None:
        self.cache, self.meta = cache, meta

    def positions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.meta.positions[None, :]  # input_ids is one row [1, N]

    def mark(self, attention_mask: torch.Tensor | None, num_tokens: int) -> None:
        if attention_mask is not None:
            raise ValueError("a ragged step has no padding, so no attention mask")

    def key_valid(self, num_new: int) -> None:
        return None

    def advance(self, num_tokens: int) -> None:
        pass
