"""Attention backends.

- `torch_ref`: plain PyTorch. Runs on every device; on CUDA it is the correctness oracle.
- `flashinfer`: paged-KV kernels, CUDA only (added at M5).

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
    raise ValueError(f"unknown attention backend: {name!r}")


__all__ = ["AttentionBackend", "get_backend"]
