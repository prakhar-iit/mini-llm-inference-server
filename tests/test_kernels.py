"""M5b/M5c: the CUDA kernels on decision 7's pages must match the torch reference on the same GPU.

Skipped without CUDA (the Mac) or without the library, so the kernels are first exercised in the GPU session.
FlashInfer and flash-attn run in bf16/fp16 only, so outputs are compared within tolerance, and model logits
with G1b's check (decision 4) rather than exact tokens.
"""

import random

import pytest
import torch

from minillm.attention import get_backend
from minillm.attention.metadata import AttentionMetadata, SeqStep
from minillm.attention.paged_torch_ref import write_kv

KERNELS = ["flashinfer", "flash_attn"]


def kernel(name):
    if not torch.cuda.is_available():
        pytest.skip("CUDA only: runs in the GPU session")
    try:
        return get_backend(name)
    except RuntimeError as e:  # library not installed
        pytest.skip(str(e))


def mixed_step(h=8, hkv=2, dh=128, bs=16, dtype=torch.bfloat16):
    """A ragged step with every row kind: fresh prompt, decode, prompt on top of cache, 1-token prompt."""
    torch.manual_seed(0)
    dev = "cuda"
    steps = [(0, 37), (90, 1), (40, 25), (0, 1), (300, 1)]  # (cached, new)
    k_pool = torch.randn(64, bs, hkv, dh, device=dev, dtype=dtype)  # junk everywhere a sequence doesn't own
    v_pool = torch.randn_like(k_pool)
    free = list(range(64))
    random.Random(3).shuffle(free)
    seqs = []
    for cached, new in steps:
        n = cached + new
        blocks = [free.pop() for _ in range(-(-n // bs))]
        seqs.append(SeqStep(cached, new, blocks))
    meta = AttentionMetadata.build(seqs, bs, dev)
    n = meta.query_start_loc[-1]
    q = torch.randn(n, h, dh, device=dev, dtype=dtype)
    k = torch.randn(n, hkv, dh, device=dev, dtype=dtype)
    v = torch.randn(n, hkv, dh, device=dev, dtype=dtype)
    return q, k, v, k_pool, v_pool, meta


@pytest.mark.parametrize("name", KERNELS)
def test_kernel_matches_reference_on_a_mixed_step(name):
    backend = kernel(name)
    q, k, v, k_pool, v_pool, meta = mixed_step()
    ref_k, ref_v = k_pool.clone(), v_pool.clone()
    expected = get_backend("torch_ref").forward(q, k, v, ref_k, ref_v, meta)
    got = backend.forward(q, k, v, k_pool, v_pool, meta)
    torch.testing.assert_close(k_pool, ref_k, atol=0, rtol=0)  # identical writes
    torch.testing.assert_close(got.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("name", KERNELS)
def test_kernel_on_a_decode_only_step(name):
    backend = kernel(name)
    q, k, v, k_pool, v_pool, _ = mixed_step()
    meta = AttentionMetadata.build([SeqStep(c, 1, list(range(i * 4, i * 4 + 4))) for i, c in enumerate([5, 63, 17])],
                                   16, "cuda")
    q, k, v = q[:3], k[:3], v[:3]
    ref_k, ref_v = k_pool.clone(), v_pool.clone()
    expected = get_backend("torch_ref").forward(q, k, v, ref_k, ref_v, meta)
    torch.testing.assert_close(backend.forward(q, k, v, k_pool, v_pool, meta).float(), expected.float(),
                               atol=2e-2, rtol=2e-2)


@pytest.mark.slow
@pytest.mark.parametrize("name", KERNELS)
def test_kernel_model_drift_within_g1b(name):
    """Llama-3.2-1B in bf16, teacher-forced: the same ragged steps (one batched prefill of ragged prompts, then
    decode steps feeding fixed tokens) through the kernel and through the reference. G1b's check, with the
    reference as its own noise baseline, so only G1b's floors apply."""
    kernel(name)
    from conftest import real_model_dir
    from minillm.cache.paged import BLOCK_SIZE, PagedKVCache
    from minillm.correctness import drift, g1b_check
    from minillm.model.loader import load_model

    model_dir = real_model_dir()
    if model_dir is None:
        pytest.skip("Llama-3.2-1B not downloaded")
    rng = random.Random(0)
    seqs = [[rng.randrange(3, 128000) for _ in range(n + 24)] for n in (12, 200, 57, 1, 640, 33, 90, 8)]
    prompt_lens = [len(t) - 24 for t in seqs]

    @torch.inference_mode()
    def logits(backend_name):
        model = load_model(model_dir, "cuda", torch.bfloat16, backend=get_backend(backend_name))
        cache = PagedKVCache(model.config, 512, "cuda", torch.bfloat16)
        blocks = [list(range(i * 64, i * 64 + 64)) for i in range(len(seqs))]
        cached, out = [0] * len(seqs), []
        upto = list(prompt_lens)
        for _ in range(25):
            meta = AttentionMetadata.build([SeqStep(c, u - c, b) for c, u, b in zip(cached, upto, blocks)],
                                           BLOCK_SIZE, "cuda")
            ids = torch.tensor([[t for s, c, u in zip(seqs, cached, upto) for t in s[c:u]]], device="cuda")
            out.append(model(ids, kv_cache=cache.step(meta), last_index=meta.logits_index)[:, 0].float().cpu())
            cached, upto = upto, [u + 1 for u in upto]
        del model, cache
        torch.cuda.empty_cache()
        return out

    ref = logits("torch_ref")
    result = g1b_check(drift(ref, ref), drift(ref, logits(name)))
    print(f"\nG1b {name} vs torch_ref: {result}")
    assert result.passed, result
