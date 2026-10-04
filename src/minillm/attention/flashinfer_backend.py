"""FlashInfer paged-KV attention. CUDA only; install with the `cuda` extra.

To build at M5 once the KV layout decision is made. Validate against TorchRefBackend on the same GPU.
"""

from __future__ import annotations

from .base import AttentionBackend


class FlashInferBackend(AttentionBackend):
    name = "flashinfer"

    def __init__(self) -> None:
        try:
            import flashinfer  # noqa: F401
        except ImportError as e:  # pragma: no cover - CUDA boxes only
            raise RuntimeError("FlashInfer is CUDA-only: pip install -e '.[cuda]' on Linux") from e

    def prefill(self, q, k, v, **kwargs):
        raise NotImplementedError("M5")

    def decode(self, q, kv_cache, block_tables, context_lens, **kwargs):
        raise NotImplementedError("M5")
