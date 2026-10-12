"""M1: the KV cache must not change the model's output, only its cost."""

import gc

import pytest
import torch
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM as HFLlamaForCausalLM

from conftest import real_model_dir
from minillm.device import select_device
from minillm.generate import greedy_generate, greedy_generate_no_cache
from minillm.model.config import LlamaConfig
from minillm.model.kv_cache import KVCache
from minillm.model.loader import from_hf_model, load_model
from test_g0_correctness import G0_PROMPTS, TINY_VARIANTS, tiny_hf_model
from test_model_config import LLAMA_32_1B


@pytest.fixture(params=list(TINY_VARIANTS), ids=list(TINY_VARIANTS))
def tiny_pair(request):
    hf = tiny_hf_model(**TINY_VARIANTS[request.param])
    return hf, from_hf_model(hf)


def new_cache(model, max_len=128, batch=1):
    p = next(model.parameters())
    return KVCache(model.config, batch, max_len, p.device, p.dtype)


def test_bytes_per_token_matches_plan():
    # The plan's formula: 2 × layers × kv_heads × head_dim × bytes = 32 KB/token for Llama-3.2-1B in bf16.
    assert KVCache.bytes_per_token(LlamaConfig.from_dict(LLAMA_32_1B), torch.bfloat16) == 32 * 1024


@torch.inference_mode()
def test_decode_steps_match_full_pass(tiny_pair):
    # Prefill 20 tokens, then feed 10 one at a time: each step's logits must equal the full pass's row.
    _, ours = tiny_pair
    ids = torch.randint(0, 256, (1, 30), generator=torch.Generator().manual_seed(4))
    full = ours(ids)
    cache = new_cache(ours)
    rows = [ours(ids[:, :20], kv_cache=cache)]
    for t in range(20, 30):
        rows.append(ours(ids[:, t:t + 1], kv_cache=cache))
    torch.testing.assert_close(torch.cat(rows, dim=1), full, atol=1e-5, rtol=1e-5)
    assert cache.seq_len == 30


@torch.inference_mode()
def test_prompt_in_chunks_matches_full_pass(tiny_pair):
    # Tq > 1 on top of cached tokens: exercises the bottom-right causal mask (needed again at M6/M7).
    _, ours = tiny_pair
    ids = torch.randint(0, 256, (2, 33), generator=torch.Generator().manual_seed(5))
    full = ours(ids)
    cache = new_cache(ours, batch=2)
    chunks = [ours(ids[:, a:b], kv_cache=cache) for a, b in [(0, 12), (12, 20), (20, 33)]]
    torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=1e-5, rtol=1e-5)


@torch.inference_mode()
def test_last_only_is_last_row(tiny_pair):
    _, ours = tiny_pair
    ids = torch.randint(0, 256, (2, 9), generator=torch.Generator().manual_seed(6))
    torch.testing.assert_close(ours(ids, last_only=True), ours(ids)[:, -1:])


@torch.inference_mode()
def test_cached_greedy_matches_no_cache_and_hf(tiny_pair):
    hf, ours = tiny_pair
    ids = torch.randint(0, 256, (1, 12), generator=torch.Generator().manual_seed(2))
    cached = greedy_generate(ours, ids, max_new_tokens=40)
    assert cached == greedy_generate_no_cache(ours, ids, max_new_tokens=40)
    expected = hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=40, do_sample=False,
                           eos_token_id=None, pad_token_id=0)[0, ids.shape[1]:].tolist()
    assert cached == expected


def test_cached_greedy_stops_at_eos():
    ours = from_hf_model(tiny_hf_model())
    ids = torch.randint(0, 256, (1, 12), generator=torch.Generator().manual_seed(2))
    first = greedy_generate(ours, ids, max_new_tokens=8)
    assert greedy_generate(ours, ids, max_new_tokens=8, eos_token_ids=(first[2],)) == first[:3]
    assert greedy_generate(ours, ids, max_new_tokens=0) == []


def test_cache_overflow_raises():
    ours = from_hf_model(tiny_hf_model())
    cache = new_cache(ours, max_len=10)
    with torch.inference_mode():
        ours(torch.zeros(1, 8, dtype=torch.long), kv_cache=cache)
        with pytest.raises(ValueError, match="KV cache full"):
            ours(torch.zeros(1, 3, dtype=torch.long), kv_cache=cache)


@pytest.mark.slow
def test_llama_32_1b_cached_matches_hf():
    """G0 for the cached path: our cached greedy vs HF's cached greedy, bf16 on the local device."""
    model_dir = real_model_dir()
    if model_dir is None:
        pytest.skip("Llama-3.2-1B not downloaded (scripts/download_model.py)")
    tok = AutoTokenizer.from_pretrained(model_dir)
    info = select_device("auto")
    prompts = [tok(p, return_tensors="pt").input_ids.to(info.device) for p in G0_PROMPTS]

    with torch.inference_mode():
        hf = HFLlamaForCausalLM.from_pretrained(model_dir, dtype=info.dtype).to(info.device).eval()
        expected = [hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=64, do_sample=False,
                                pad_token_id=tok.eos_token_id)[0, ids.shape[1]:].tolist() for ids in prompts]
    del hf
    gc.collect()
    if info.device.type == "mps":
        torch.mps.empty_cache()

    ours = load_model(model_dir, device=info.device, dtype=info.dtype)
    for ids, exp in zip(prompts, expected):
        got = greedy_generate(ours, ids, max_new_tokens=64, eos_token_ids=ours.config.eos_token_ids)
        assert got == exp, f"ours: {tok.decode(got)!r}\nHF:   {tok.decode(exp)!r}"
