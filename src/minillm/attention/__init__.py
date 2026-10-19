"""Attention backends.

- `torch_ref`: plain PyTorch. Runs on every device; on CUDA it is the correctness oracle. Its paged path
  (M5a) is in `paged_torch_ref.py`.
- `flashinfer`: paged-KV kernels, CUDA only (M5b).
- `flash_attn`: vLLM's fork of flash-attn, paged at block size 16, CUDA only (M5c).
- `metadata.py`: the per-step description of a ragged batch on paged KV that every backend reads.

Pick one with `get_backend(name)`; the engine never imports a backend directly.
"""

from .base import AttentionBackend


def get_backend(name: str) -> AttentionBackend:
    if name == "torch_ref":
        from .torch_ref import TorchRefBackend

        return TorchRefBackend()
    if name == "flashinfer":
        from .flashinfer_backend import FlashInferBackend  # CUDA only

        return FlashInferBackend()
    if name == "flash_attn":
        from .flash_attn_backend import FlashAttnBackend  # CUDA only, vLLM's fork

        return FlashAttnBackend()
    raise ValueError(f"unknown attention backend: {name!r}")


__all__ = ["AttentionBackend", "get_backend"]
