"""FlashInfer paged-KV attention (M5b). CUDA only; install with the `cuda` extra.

Reads decision 7's pages in place: per layer, K and V pools [num_blocks, 16, Hkv, Dh] (NHD), passed as a pair.
- Writes: the same `slot_mapping` scatter as the reference backend (one indexed copy per layer), so the pool is
  written identically whichever backend runs.
- Attention: FlashInfer splits planning (work partitioning from the step's page lists, on the host) from
  running. `plan()` runs once per step, on the first layer that sees a new `AttentionMetadata`; the other layers
  reuse it. A step whose rows are all single tokens uses the decode wrapper; anything else (prompts, prompts on
  top of cached tokens, mixed) uses the prefill wrapper with a causal mask aligned to the end of each context.
- The contiguous `prefill` path (generate.py, correctness tests) stays on the torch reference.
Validate against TorchRefBackend on the same GPU (tests/test_kernels.py, G1b's tolerance).
"""

from __future__ import annotations

import torch

from .base import AttentionBackend
from .paged_torch_ref import write_kv
from .torch_ref import TorchRefBackend

WORKSPACE_BYTES = 128 * 2**20  # per wrapper; FlashInfer's own examples use 128 MB


class FlashInferBackend(AttentionBackend):
    name = "flashinfer"

    def __init__(self) -> None:
        try:
            import flashinfer
        except ImportError as e:  # pragma: no cover - CUDA boxes only
            raise RuntimeError("FlashInfer is CUDA-only: pip install -e '.[cuda]' on Linux") from e
        self._fi = flashinfer
        self._ref = TorchRefBackend()
        self._prefill_wrapper = self._decode_wrapper = None
        self._planned = None  # the AttentionMetadata the wrappers were last planned for
        self._wrapper = None

    def prefill(self, q, k, v, key_valid=None, **kwargs):
        return self._ref.prefill(q, k, v, key_valid=key_valid, **kwargs)

    def forward(self, q, k, v, k_cache, v_cache, meta):
        write_kv(k_cache, v_cache, meta.slot_mapping, k, v)
        if self._planned is not meta:
            self._plan(q, k_cache, meta)
        return self._wrapper.run(q.contiguous(), (k_cache, v_cache))

    def _plan(self, q: torch.Tensor, k_cache: torch.Tensor, meta) -> None:
        if self._prefill_wrapper is None:
            ws = lambda: torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=q.device)  # noqa: E731
            self._prefill_wrapper = self._fi.BatchPrefillWithPagedKVCacheWrapper(ws(), kv_layout="NHD")
            self._decode_wrapper = self._fi.BatchDecodeWithPagedKVCacheWrapper(ws(), kv_layout="NHD")
        h, dh = q.shape[1], q.shape[2]
        page, hkv = k_cache.shape[1], k_cache.shape[2]
        if meta.max_query_len == 1:
            self._decode_wrapper.plan(meta.kv_indptr, meta.kv_indices, meta.kv_last_page_len, h, hkv, dh, page,
                                      pos_encoding_mode="NONE", q_data_type=q.dtype, kv_data_type=k_cache.dtype)
            self._wrapper = self._decode_wrapper
        else:
            self._prefill_wrapper.plan(meta.qo_indptr, meta.kv_indptr, meta.kv_indices, meta.kv_last_page_len,
                                       h, hkv, dh, page, causal=True, q_data_type=q.dtype,
                                       kv_data_type=k_cache.dtype)
            self._wrapper = self._prefill_wrapper
        self._planned = meta
