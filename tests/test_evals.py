"""G2 plumbing: the eval math and answer parsing, on tiny inputs (the real run is benchmarks/g2_quality.py)."""

import math

import pytest
import torch

from minillm.evals import extract_answer, gsm8k_prompt, perplexity
from minillm.model.loader import from_hf_model
from test_g0_correctness import tiny_hf_model


def test_uniform_logits_give_vocab_size_perplexity():
    vocab = 50
    ids = torch.randint(0, vocab, (1, 100), generator=torch.Generator().manual_seed(0))
    ppl = perplexity(lambda x: torch.zeros(1, x.shape[1], vocab), ids, window=32)
    assert ppl == pytest.approx(vocab, rel=1e-6)


def test_perfect_prediction_gives_perplexity_one():
    ids = torch.arange(40).unsqueeze(0) % 10

    def oracle(x):  # puts all the mass on the true next token
        logits = torch.full((1, x.shape[1], 10), -1e4)
        logits[0, torch.arange(x.shape[1] - 1), x[0, 1:]] = 0
        return logits

    assert perplexity(oracle, ids, window=16) == pytest.approx(1.0, abs=1e-6)


def test_our_perplexity_matches_hf_on_tiny_model():
    hf = tiny_hf_model()
    ours = from_hf_model(hf)
    ids = torch.randint(0, 256, (1, 300), generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        a = perplexity(ours, ids, window=64)
        b = perplexity(lambda x: hf(x).logits, ids, window=64)
    assert math.isclose(a, b, rel_tol=1e-5)


@pytest.mark.parametrize("completion,expected", [
    (" She has 16 - 3 - 4 = 9 eggs. She makes 9 * 2 = 18 dollars. The answer is 18.", "18"),
    (" Total is 1,250 + 50 = 1,300. The answer is $1,300.\n\nQuestion: next", "1300"),
    (" It costs 3.50 each so 7.0 in total.", "7"),
    (" The answer is -4.", "-4"),
    (" no number here", None),
    (" 5 apples.\n\nQuestion: Tom has 99", "5"),
])
def test_extract_answer(completion, expected):
    assert extract_answer(completion) == expected


def test_gsm8k_prompt_format():
    shots = [{"question": "What is 2+3?", "answer": "2+3=<<2+3=5>>5\n#### 5"}]
    p = gsm8k_prompt("What is 1+1?", shots)
    assert p == "Question: What is 2+3?\nAnswer: 2+3=5 The answer is 5.\n\nQuestion: What is 1+1?\nAnswer:"
