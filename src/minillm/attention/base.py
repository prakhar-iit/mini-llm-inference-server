"""The attention-backend interface. Shape conventions are decided at M5 (see docs/decisions/)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class AttentionBackend(ABC):
    name: str = "base"

    @abstractmethod
    def prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        """Attention over the prompt tokens (causal)."""

    @abstractmethod
    def decode(self, q: torch.Tensor, kv_cache, block_tables, context_lens, **kwargs) -> torch.Tensor:
        """One new token per sequence, attending to its cached KV via the block table."""
