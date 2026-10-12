"""RoPE frequencies, including Llama 3's frequency scaling.

This file only computes `inv_freq` (one frequency per pair of head dims). Rotating q and k with it is
part of the forward pass in `llama.py`.
"""

from __future__ import annotations

import math

import torch

from .config import LlamaConfig


def compute_inv_freq(config: LlamaConfig) -> torch.Tensor:
    """Return inv_freq of shape [head_dim // 2], float32, on CPU."""
    dim = config.head_dim
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))

    scaling = config.rope_scaling
    if scaling is None:
        return inv_freq
    if scaling["rope_type"] != "llama3":
        raise NotImplementedError(f"rope_type {scaling['rope_type']!r}")

    # Llama 3.1+ context extension: leave high frequencies alone, divide low frequencies by `factor`,
    # and interpolate smoothly between the two bands.
    factor = scaling["factor"]
    low_freq_factor = scaling["low_freq_factor"]
    high_freq_factor = scaling["high_freq_factor"]
    old_context_len = scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    scaled = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    is_medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return torch.where(is_medium, smoothed, scaled)
