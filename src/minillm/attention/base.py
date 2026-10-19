"""The attention-backend interface (decision 7).

- `prefill`: attention over contiguous k/v (no cache, M1's KVCache, M3's padded batches). Used by
  `generate.py` and the correctness tests.
- `forward`: the engine's path from M5: ragged tokens, k/v in the paged pool, described by `AttentionMetadata`.
  The torch reference (M5a), FlashInfer (M5b) and flash-attn (M5c) implement it on the same layout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class AttentionBackend(ABC):
    name: str = "base"

    @abstractmethod
    def prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                key_valid: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Causal attention of the new tokens over everything so far.

        q [B, H, Tq, Dh]; k, v [B, Hkv, Tk, Dh] with Tk >= Tq and H a multiple of Hkv (grouped-query
        attention). The queries are the LAST Tq of the Tk positions: query i sits at position
        Tk - Tq + i and sees keys 0 .. Tk - Tq + i. Tq == Tk is a fresh prompt, Tq == 1 a decode step,
        anything between a prompt continued on top of cached tokens.

        key_valid [B, Tk] (bool, optional): False marks keys that aren't real (M3's left-padding, or past a
        shorter row's end in an M4 slot batch). No query attends to one, except that a query with no real key
        at all may see its own position: softmax over zero keys would return NaN (its output is ignored).
        Returns [B, H, Tq, Dh]. Scale is 1/sqrt(Dh).
        """

    @abstractmethod
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor,
                v_cache: torch.Tensor, meta) -> torch.Tensor:
        """One layer of one engine step on paged KV.

        q [N, H, Dh], k, v [N, Hkv, Dh]: this step's new tokens, all sequences concatenated (ragged).
        k_cache, v_cache [num_blocks, block_size, Hkv, Dh]: the layer's pool. `meta` (AttentionMetadata)
        says which tokens belong to which sequence, where they go (`slot_mapping`) and which blocks hold each
        sequence's context (`block_tables`, `seq_lens`). Writes k, v into the pool, then returns each new
        token's causal attention over its sequence's context, [N, H, Dh].
        """
