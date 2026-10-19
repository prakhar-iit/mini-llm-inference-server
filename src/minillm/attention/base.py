"""The attention-backend interface. Prefill shapes are fixed at M0; decode/KV layout is decided at M5
(decision 7, see docs/decisions/)."""

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
    def decode(self, q: torch.Tensor, kv_cache, block_tables, context_lens, **kwargs) -> torch.Tensor:
        """One new token per sequence, attending to its cached KV via the block table."""
