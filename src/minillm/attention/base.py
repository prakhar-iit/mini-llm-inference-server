"""The attention-backend interface. Prefill shapes are fixed at M0; decode/KV layout is decided at M5
(decision 7, see docs/decisions/)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class AttentionBackend(ABC):
    name: str = "base"

    @abstractmethod
    def prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        """Causal attention over the prompt tokens.

        q [B, H, T, Dh]; k, v [B, Hkv, T, Dh] with H a multiple of Hkv (grouped-query attention).
        Returns [B, H, T, Dh]. Scale is 1/sqrt(Dh).
        """

    @abstractmethod
    def decode(self, q: torch.Tensor, kv_cache, block_tables, context_lens, **kwargs) -> torch.Tensor:
        """One new token per sequence, attending to its cached KV via the block table."""
