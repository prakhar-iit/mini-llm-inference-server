"""Contiguous KV cache (M1).

Each layer keeps preallocated K and V buffers of shape [B, Hkv, max_len, Dh]. A forward pass writes its
new tokens' k/v at [seq_len, seq_len + T) and attends over [0, seq_len + T). Every sequence in the
batch shares one length: fine for batch 1 (M1) and static batching (M3). M5 replaces this with paged
blocks, which is what lets sequences of different lengths share memory.
"""

from __future__ import annotations

import torch

from .config import LlamaConfig


class KVCache:
    def __init__(
        self,
        config: LlamaConfig,
        batch_size: int,
        max_len: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        shape = (batch_size, config.num_key_value_heads, max_len, config.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.max_len = max_len
        self.seq_len = 0  # tokens already stored; also the position of the next token

    @staticmethod
    def bytes_per_token(config: LlamaConfig, dtype: torch.dtype) -> int:
        """2 (K and V) × layers × KV heads × head_dim × bytes per element."""
        elem = torch.tensor([], dtype=dtype).element_size()
        return 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * elem

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Store this step's k, v [B, Hkv, T, Dh] for one layer; return that layer's k, v for all tokens so far.

        Does not advance `seq_len`: every layer writes at the same offset, then the model calls
        `advance` once after the last layer.
        """
        start, end = self.seq_len, self.seq_len + k.shape[2]
        if end > self.max_len:
            raise ValueError(f"KV cache full: {end} tokens > max_len {self.max_len}")
        self.k[layer_idx][:, :, start:end] = k
        self.v[layer_idx][:, :, start:end] = v
        return self.k[layer_idx][:, :, :end], self.v[layer_idx][:, :, :end]

    def advance(self, num_tokens: int) -> None:
        self.seq_len += num_tokens
