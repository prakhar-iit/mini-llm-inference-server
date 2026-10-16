"""Plain-PyTorch reference attention. Works on MPS, CUDA and CPU.

- M0/M1: `prefill` with scaled_dot_product_attention: causal, and on top of a KV cache.
- M3: a padding mask for left-padded batches (`key_valid`).
- M5 (to build): `decode` that gathers KV blocks through the block table, then runs SDPA.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import AttentionBackend


class TorchRefBackend(AttentionBackend):
    name = "torch_ref"

    def prefill(self, q, k, v, key_valid=None, **kwargs):
        # B, T, V - Batch Size, Sequence Token Length, and Vocab
        # q [B, H, Tq, Dh], k/v [B, Hkv, Tk, Dh] -> [B, H, Tq, Dh]. Scale defaults to 1/sqrt(Dh).
        # enable_gqa: each KV head is shared by H // Hkv query heads, without copying k and v.
        tq, tk = q.shape[2], k.shape[2]
        if key_valid is not None:
            # Batch with padding (M3): causal (bottom-right) AND key is real, OR the query's own position.
            causal = torch.ones(tq, tk, dtype=torch.bool, device=q.device).tril(diagonal=tk - tq)
            own = torch.zeros(tq, tk, dtype=torch.bool, device=q.device)
            own[torch.arange(tq), torch.arange(tq) + tk - tq] = True
            mask = causal & (key_valid[:, None, None, :] | own)  # [B, 1, Tq, Tk], broadcast over heads
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

    def decode(self, q, kv_cache, block_tables, context_lens, **kwargs):
        raise NotImplementedError("M5")
