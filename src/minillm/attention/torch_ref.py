"""Plain-PyTorch reference attention. Works on MPS, CUDA and CPU.

To build:
- M0/M1: `prefill` with torch.nn.functional.scaled_dot_product_attention(is_causal=True).
- M5: `decode` that gathers KV blocks through the block table, then runs SDPA.
"""

from __future__ import annotations

from .base import AttentionBackend


class TorchRefBackend(AttentionBackend):
    name = "torch_ref"

    def prefill(self, q, k, v, **kwargs):
        raise NotImplementedError("M0/M1")

    def decode(self, q, kv_cache, block_tables, context_lens, **kwargs):
        raise NotImplementedError("M5")
