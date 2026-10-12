"""G0: our forward pass and greedy loop must match Hugging Face (batch 1, same device).

Tiny random configs run in fp32 on CPU, everywhere, in a second. The Llama-3.2-1B test runs in bf16 on
the best local device once the weights are in models/Llama-3.2-1B (scripts/download_model.py) or
MINILLM_MODEL_DIR.
"""

import gc

import pytest
import torch
from transformers import AutoTokenizer
from transformers import LlamaConfig as HFLlamaConfig
from transformers import LlamaForCausalLM as HFLlamaForCausalLM

from minillm.device import select_device
from minillm.generate import greedy_generate_no_cache
from minillm.model.loader import from_hf_model, load_model

from conftest import real_model_dir

TINY_VARIANTS = {
    "plain": {},
    "llama3-rope": {"rope_parameters": {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 32.0,
                                        "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                        "original_max_position_embeddings": 64}},
    "tied-embeddings": {"tie_word_embeddings": True},
    "mha": {"num_key_value_heads": 4},  # no GQA: one KV head per query head
}


def tiny_hf_model(**overrides) -> HFLlamaForCausalLM:
    torch.manual_seed(0)
    config = HFLlamaConfig(**{
        "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 256,
        **overrides,
    })
    return HFLlamaForCausalLM(config).eval()


@pytest.fixture(params=list(TINY_VARIANTS), ids=list(TINY_VARIANTS))
def tiny_pair(request):
    hf = tiny_hf_model(**TINY_VARIANTS[request.param])
    return hf, from_hf_model(hf)


@torch.inference_mode()
def test_logits_match_hf(tiny_pair):
    hf, ours = tiny_pair
    ids = torch.randint(0, 256, (2, 40), generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(ours(ids), hf(ids).logits, atol=1e-4, rtol=1e-4)


@torch.inference_mode()
def test_greedy_tokens_match_hf(tiny_pair):
    hf, ours = tiny_pair
    ids = torch.randint(0, 256, (1, 12), generator=torch.Generator().manual_seed(2))
    expected = hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=24, do_sample=False,
                           eos_token_id=None, pad_token_id=0)[0, ids.shape[1]:].tolist()
    assert greedy_generate_no_cache(ours, ids, max_new_tokens=24) == expected


def test_stops_at_eos():
    hf = tiny_hf_model()
    ours = from_hf_model(hf)
    ids = torch.randint(0, 256, (1, 12), generator=torch.Generator().manual_seed(2))
    first = greedy_generate_no_cache(ours, ids, max_new_tokens=8)
    assert greedy_generate_no_cache(ours, ids, max_new_tokens=8, eos_token_ids=(first[2],)) == first[:3]


G0_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "La photosynthèse est le processus par lequel",
    # Long-ish prompt so positions well past the first few hundred are exercised.
    "Summarize the following text.\n\n" + " ".join(f"Sentence {i} describes step {i} of the process." for i in range(60)),
]


@pytest.mark.slow
def test_llama_32_1b_matches_hf():
    model_dir = real_model_dir()
    if model_dir is None:
        pytest.skip("Llama-3.2-1B not downloaded (scripts/download_model.py)")
    tok = AutoTokenizer.from_pretrained(model_dir)
    info = select_device("auto")
    prompts = [tok(p, return_tensors="pt").input_ids.to(info.device) for p in G0_PROMPTS]

    # bf16 (~2.5 GB per model) and one model in memory at a time: two fp32 copies don't fit in 8 GB.
    # HF also runs without its KV cache, so both sides do the same full recompute each step; cached
    # and uncached attention round differently in bf16 and could legitimately pick different tokens.
    with torch.inference_mode():
        hf = HFLlamaForCausalLM.from_pretrained(model_dir, dtype=info.dtype).to(info.device).eval()
        expected = [hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=32, do_sample=False,
                                use_cache=False, pad_token_id=tok.eos_token_id)[0, ids.shape[1]:].tolist()
                    for ids in prompts]
    del hf
    gc.collect()
    if info.device.type == "mps":
        torch.mps.empty_cache()
    elif info.device.type == "cuda":
        torch.cuda.empty_cache()

    ours = load_model(model_dir, device=info.device, dtype=info.dtype)
    eos = ours.config.eos_token_ids
    for ids, exp in zip(prompts, expected):
        got = greedy_generate_no_cache(ours, ids, max_new_tokens=32, eos_token_ids=eos)
        assert got == exp, f"ours: {tok.decode(got)!r}\nHF:   {tok.decode(exp)!r}"


@pytest.mark.parametrize("variant", ["plain", "tied-embeddings", "llama3-rope"])
@torch.inference_mode()
def test_load_model_from_checkpoint_matches_hf(tmp_path, variant):
    # Exercises load_model (meta device -> to_empty -> safetensors), the path the real weights take.
    hf = tiny_hf_model(**TINY_VARIANTS[variant])
    hf.save_pretrained(tmp_path)
    ours = load_model(tmp_path, device="cpu", dtype=torch.float32)
    if variant == "tied-embeddings":
        assert ours.lm_head.weight is ours.model.embed_tokens.weight
    ids = torch.randint(0, 256, (1, 20), generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(ours(ids), hf(ids).logits, atol=1e-4, rtol=1e-4)
