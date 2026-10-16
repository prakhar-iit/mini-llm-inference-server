"""M3 batching correctness. G1a: in fp32, a left-padded batch must reproduce batch-1 results exactly."""

import pytest
import torch

from minillm.generate import greedy_generate, greedy_generate_batch, pad_left
from minillm.model.kv_cache import KVCache
from minillm.model.loader import from_hf_model
from test_g0_correctness import TINY_VARIANTS, tiny_hf_model

LENGTHS = [3, 17, 9, 1, 30, 12]  # deliberately ragged, including a one-token prompt


@pytest.fixture(params=list(TINY_VARIANTS), ids=list(TINY_VARIANTS))
def pair(request):
    hf = tiny_hf_model(**TINY_VARIANTS[request.param])
    return hf, from_hf_model(hf)


def prompts():
    g = torch.Generator().manual_seed(11)
    return [torch.randint(3, 256, (n,), generator=g).tolist() for n in LENGTHS]


def test_pad_left():
    ids, mask = pad_left([[5, 6], [7]], "cpu")
    assert ids.tolist() == [[5, 6], [0, 7]] and mask.tolist() == [[1, 1], [0, 1]]
    assert pad_left([[1, 2], [3, 4]], "cpu")[1] is None  # equal lengths: no mask, fast path


@torch.inference_mode()
def test_g1a_padded_prefill_matches_batch_1(pair):
    _, ours = pair
    ps = prompts()
    ids, mask = pad_left(ps, "cpu")
    batched = ours(ids, attention_mask=mask)  # no cache: padding mask applied directly
    for i, p in enumerate(ps):
        alone = ours(torch.tensor([p]))
        torch.testing.assert_close(batched[i, -len(p):], alone[0], atol=1e-4, rtol=1e-4)


@torch.inference_mode()
def test_g1a_padded_prefill_matches_hf_batched(pair):
    hf, ours = pair
    ps = prompts()
    ids, mask = pad_left(ps, "cpu")
    positions = (mask.cumsum(1) - 1).clamp_min(0)
    expected = hf(ids, attention_mask=mask, position_ids=positions).logits
    got = ours(ids, attention_mask=mask)
    real = mask.bool()
    torch.testing.assert_close(got[real], expected[real], atol=1e-4, rtol=1e-4)


@torch.inference_mode()
def test_g1a_batched_decode_logits_match_batch_1(pair):
    # Padded prefill through the cache, then decode steps: every row's logits equal its batch-1 run.
    _, ours = pair
    ps = prompts()
    gen = greedy_generate_batch(ours, ps, 6)
    p = next(ours.parameters())
    ids, mask = pad_left(ps, "cpu")
    cache = KVCache(ours.config, len(ps), ids.shape[1] + 6, "cpu", p.dtype)
    rows = [ours(ids, kv_cache=cache, last_only=True, attention_mask=mask)]
    for step in range(5):
        nxt = torch.tensor([[g[step]] for g in gen])
        rows.append(ours(nxt, kv_cache=cache, last_only=True))
    batched = torch.cat(rows, dim=1)  # [B, 6, V]
    for i, prompt in enumerate(ps):
        full = ours(torch.tensor([prompt + gen[i][:5]]))[0, len(prompt) - 1:]
        torch.testing.assert_close(batched[i], full, atol=1e-4, rtol=1e-4)


def test_g1a_batched_greedy_tokens_identical(pair):
    _, ours = pair
    ps = prompts()
    batched = greedy_generate_batch(ours, ps, 25)
    for p, got in zip(ps, batched):
        assert got == greedy_generate(ours, torch.tensor([p]), 25)


def test_padding_query_rows_stay_finite():
    # A padding position attends only to itself; without that, softmax over zero keys gives NaN.
    ours = from_hf_model(tiny_hf_model())
    ids, mask = pad_left([[5] * 10, [6]], "cpu")
    with torch.inference_mode():
        assert torch.isfinite(ours(ids, attention_mask=mask)).all()


@torch.inference_mode()
def test_positions_count_real_tokens_only():
    # Output alone can't catch shifted positions: RoPE scores depend only on distances, and padding shifts a
    # whole row equally. They matter once keys are reused across requests (M6 prefix caching), so check the
    # positions themselves: each row counts from 0 at its first real token, through prefill and decode.
    ours = from_hf_model(tiny_hf_model())
    seen = []
    orig = ours.model.rotary_emb.forward
    ours.model.rotary_emb.forward = lambda pos, dtype: (seen.append(pos.clone()), orig(pos, dtype))[1]
    ids, mask = pad_left([[5, 6, 7], [8]], "cpu")
    cache = KVCache(ours.config, 2, 8, "cpu", torch.float32)
    ours(ids, kv_cache=cache, attention_mask=mask)
    ours(torch.tensor([[1], [2]]), kv_cache=cache)
    ours(torch.tensor([[1], [2]]), kv_cache=cache)
    assert seen[0].tolist() == [[0, 1, 2], [0, 0, 0]]  # padding slots get 0 (masked anyway)
    assert seen[1].tolist() == [[3], [1]] and seen[2].tolist() == [[4], [2]]


# ---------- G1b: bf16 teacher-forced drift (decision 4) ----------

from minillm.correctness import Drift, drift, g1b_check, teacher_forced_logits  # noqa: E402


def test_drift_math():
    ref = [torch.tensor([[2.0, 0.0, -1.0], [0.1, 0.0, 0.05]])]
    assert drift(ref, ref) == Drift(0.0, 0.0, 0, 2)
    flipped = [torch.tensor([[0.0, 2.0, -1.0], [0.0, 0.1, 0.05]])]  # row 0: clear lead flipped; row 1: near-tie
    d = drift(ref, flipped)
    assert d.top1_disagreements == 1 and d.max_abs_diff == 2.0 and d.mean_kl > 0
    noise = Drift(1e-4, 0.3, 0, 10)
    assert g1b_check(noise, Drift(1.5e-4, 0.5, 0, 10)).passed
    assert not g1b_check(noise, Drift(3e-4, 0.5, 0, 10)).passed  # KL over 2× noise
    assert not g1b_check(noise, Drift(1e-4, 0.5, 1, 10)).passed  # a clear top-1 flip


def test_g1b_tiny_fp32_has_no_drift():
    ours = from_hf_model(tiny_hf_model())
    seqs = prompts()
    start = [max(1, len(s) - 4) for s in seqs]
    fwd = lambda ids, mask: ours(ids, attention_mask=mask)  # noqa: E731
    d = drift(teacher_forced_logits(fwd, seqs, start, 1, "cpu"), teacher_forced_logits(fwd, seqs, start, 6, "cpu"))
    assert d.max_abs_diff < 1e-4 and d.top1_disagreements == 0


@pytest.mark.slow
def test_g1b_llama_32_1b_bf16():
    """Batch 8 vs batch 1 in bf16 on the local device: our drift must be within 2× HF's own drift."""
    import gc

    from transformers import AutoTokenizer
    from transformers import LlamaForCausalLM as HFLlama

    from conftest import real_model_dir
    from minillm.device import select_device
    from minillm.model.loader import load_model
    from test_g0_correctness import G0_PROMPTS

    model_dir = real_model_dir()
    if model_dir is None:
        pytest.skip("Llama-3.2-1B not downloaded")
    tok = AutoTokenizer.from_pretrained(model_dir)
    info = select_device("auto")
    texts = G0_PROMPTS + ["Once upon a time", "import numpy as np\n", "Die Hauptstadt von Deutschland ist",
                          "List three prime numbers:"]
    prompt_ids = [tok(t).input_ids[:120] for t in texts]

    ours = load_model(model_dir, info.device, info.dtype)
    seqs = [p + greedy_generate(ours, torch.tensor([p], device=info.device), 24) for p in prompt_ids]
    start = [len(p) for p in prompt_ids]
    fwd = lambda ids, mask: ours(ids, attention_mask=mask)  # noqa: E731
    ours_drift = drift(teacher_forced_logits(fwd, seqs, start, 1, info.device),
                       teacher_forced_logits(fwd, seqs, start, 8, info.device))
    del ours
    gc.collect()
    if info.device.type == "mps":
        torch.mps.empty_cache()

    hf = HFLlama.from_pretrained(model_dir, dtype=info.dtype).to(info.device).eval()

    def hf_fwd(ids, mask):
        if mask is None:
            return hf(ids).logits
        return hf(ids, attention_mask=mask, position_ids=(mask.cumsum(1) - 1).clamp_min(0)).logits

    noise = drift(teacher_forced_logits(hf_fwd, seqs, start, 1, info.device),
                  teacher_forced_logits(hf_fwd, seqs, start, 8, info.device))
    result = g1b_check(noise, ours_drift)
    print(f"\nG1b {info.name}: HF noise {noise}\n    ours {ours_drift}\n    limits KL {result.kl_limit:.2e}, "
          f"max diff {result.max_diff_limit:.3f}")
    assert result.passed, result
