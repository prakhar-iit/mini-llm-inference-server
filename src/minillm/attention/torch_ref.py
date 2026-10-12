"""Plain-PyTorch reference attention. Works on MPS, CUDA and CPU.

To build:
- M0/M1: `prefill` with torch.nn.functional.scaled_dot_product_attention(is_causal=True).
- M5: `decode` that gathers KV blocks through the block table, then runs SDPA.
"""

from __future__ import annotations

import torch.nn.functional as F

from .base import AttentionBackend


class TorchRefBackend(AttentionBackend):
    name = "torch_ref"

    def prefill(self, q, k, v, **kwargs):
        # B, T, V - Batch Size, Sequence Token Length, and Vocab
        # q [B, H, T, Dh], k/v [B, Hkv, T, Dh] -> [B, H, T, Dh]. Scale defaults to 1/sqrt(Dh).
        # is_causal: token t attends only to tokens 0..t. Its mask is aligned top-left, which is only
        #   right while q and k have the same length (true in M0; not once M1 adds cached KV).
        # enable_gqa: each KV head is shared by H // Hkv query heads, without copying k and v.
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    def decode(self, q, kv_cache, block_tables, context_lens, **kwargs):
        raise NotImplementedError("M5")
