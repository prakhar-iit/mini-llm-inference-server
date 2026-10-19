"""Plain-PyTorch reference attention. Works on MPS, CUDA and CPU.

- M0/M1: `prefill` with scaled_dot_product_attention: causal, and on top of a KV cache.
- M3: a padding mask for left-padded batches (`key_valid`); M4 reuses it for rows of different lengths.
- M5a: `forward` on paged KV, in `paged_torch_ref.py`: gathers blocks through the block table, then SDPA.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AttentionBackend
from .paged_torch_ref import paged_attention, write_kv


class TorchRefBackend(AttentionBackend):
    name = "torch_ref"

    def prefill(self, q, k, v, key_valid=None, **kwargs):
        # B, T, V - Batch Size, Sequence Token Length, and Vocab
        # q [B, H, Tq, Dh], k/v [B, Hkv, Tk, Dh] -> [B, H, Tq, Dh]. Scale defaults to 1/sqrt(Dh).
        # enable_gqa: each KV head is shared by H // Hkv query heads, without copying k and v.
        tq, tk = q.shape[2], k.shape[2]
        if key_valid is not None:
            # Rows with padding (M3) or of different lengths (M4): causal (bottom-right) AND key is real.
            causal = torch.ones(tq, tk, dtype=torch.bool, device=q.device).tril(diagonal=tk - tq)
            mask = causal & key_valid[:, None, None, :]  # [B, 1, Tq, Tk], broadcast over heads
            # A query that sees no real key (a left-padding position) may see its own slot instead. Only then:
            # in an M4 decode step the last column is a stale slot for every row shorter than the longest.
            own = torch.zeros(tq, tk, dtype=torch.bool, device=q.device)
            own[torch.arange(tq), torch.arange(tq) + tk - tq] = True
            mask = mask | (own & ~mask.any(dim=-1, keepdim=True))
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
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

    def forward(self, q, k, v, k_cache, v_cache, meta):
        write_kv(k_cache, v_cache, meta.slot_mapping, k, v)
        return paged_attention(q, k_cache, v_cache, meta)
