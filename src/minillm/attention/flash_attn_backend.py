"""flash-attn paged attention (M5c), through vLLM's fork. CUDA only; install with the `flashattn` extra.

Upstream flash-attn 2 pages KV only in blocks that are multiples of 256 tokens; vLLM's fork (`vllm-flash-attn`)
takes decision 7's 16-token blocks, so it reads the very same pools as the reference and FlashInfer:
K and V [num_blocks, 16, Hkv, Dh] (NHD) per layer.
- Writes: the shared `slot_mapping` scatter (`write_kv`).
- Attention: one varlen call per layer for the whole ragged step: cumulative query lengths (`qo_indptr`),
  each sequence's context length (`seqused_k`), and the int32 block table. Causal masks align to the end of
  each context, so decode rows, prompts and prompts on top of cached tokens all go through the same call.
  No planning step: the call reads the metadata tensors directly.
- The contiguous `prefill` path stays on the torch reference.
Validate against TorchRefBackend on the same GPU (tests/test_kernels.py, G1b's tolerance).
"""

from __future__ import annotations

from .base import AttentionBackend
from .paged_torch_ref import write_kv
from .torch_ref import TorchRefBackend


def _varlen():
    try:
        from vllm_flash_attn import flash_attn_varlen_func  # the standalone package
    except ImportError:
        from vllm.vllm_flash_attn import flash_attn_varlen_func  # bundled inside vLLM
    return flash_attn_varlen_func


class FlashAttnBackend(AttentionBackend):
    name = "flash_attn"

    def __init__(self) -> None:
        try:
            self._attn = _varlen()
        except ImportError as e:  # pragma: no cover - CUDA boxes only
            raise RuntimeError("vLLM's flash-attn is CUDA-only: pip install -e '.[flashattn]' on Linux") from e
        self._ref = TorchRefBackend()

    def prefill(self, q, k, v, key_valid=None, **kwargs):
        return self._ref.prefill(q, k, v, key_valid=key_valid, **kwargs)

    def forward(self, q, k, v, k_cache, v_cache, meta):
        write_kv(k_cache, v_cache, meta.slot_mapping, k, v)
        return self._attn(
            q=q.contiguous(), k=k_cache, v=v_cache,
            max_seqlen_q=meta.max_query_len, cu_seqlens_q=meta.qo_indptr,
            max_seqlen_k=meta.max_seq_len, seqused_k=meta.seq_lens_i32,
            block_table=meta.block_tables_i32,
            causal=True, softmax_scale=q.shape[-1] ** -0.5,
        )
