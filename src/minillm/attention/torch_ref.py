"""Plain-PyTorch reference attention. Works on MPS, CUDA and CPU.

To build:
- M0/M1: `prefill` with torch.nn.functional.scaled_dot_product_attention(is_causal=True).
- M5: `decode` that gathers KV blocks through the block table, then runs SDPA.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AttentionBackend


class TorchRefBackend(AttentionBackend):
    name = "torch_ref"

    def prefill(self, q, k, v, **kwargs):
        # B, T, V - Batch Size, Sequence Token Length, and Vocab
        # q [B, H, Tq, Dh], k/v [B, Hkv, Tk, Dh] -> [B, H, Tq, Dh]. Scale defaults to 1/sqrt(Dh).
        # enable_gqa: each KV head is shared by H // Hkv query heads, without copying k and v.
        tq, tk = q.shape[2], k.shape[2]
        if tq == tk:
            # Fresh prompt. is_causal aligns its mask top-left, which is right only when Tq == Tk.
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        if tq == 1:
            # Decode step: the one new token is the newest position, so it may see every key. No mask.
            return F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        # Prompt continued on top of cached tokens: causal mask aligned bottom-right.
        # True = may attend. Query i (position tk - tq + i) sees keys 0 .. tk - tq + i.
        mask = torch.ones(tq, tk, dtype=torch.bool, device=q.device).tril(diagonal=tk - tq)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)

    def decode(self, q, kv_cache, block_tables, context_lens, **kwargs):
        raise NotImplementedError("M5")
