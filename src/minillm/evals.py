"""G2 quality eval: WikiText-2 perplexity and a GSM8K subset.

G0/G1 check that our numbers match Hugging Face's. G2 checks that the model is still *good* when exact
matching no longer applies: quantized weights (M9, K5), fp8 KV (M9), different kernels (M5). Each
milestone's result must be within noise of the HF baseline recorded in `benchmarks/baselines/`.

Both functions take a callable, so the same code scores our model and HF's.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

import torch
import torch.nn.functional as F

# ---------- Perplexity ----------


def wikitext_token_ids(tokenizer, num_tokens: int) -> torch.Tensor:
    """The first `num_tokens` tokens of WikiText-2 (raw) test, as [1, num_tokens]."""
    from datasets import load_dataset

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :num_tokens]
    if ids.shape[1] < num_tokens:
        raise ValueError(f"WikiText-2 test has only {ids.shape[1]} tokens")
    return ids


@torch.inference_mode()
def perplexity(logits_fn: Callable[[torch.Tensor], torch.Tensor], token_ids: torch.Tensor, window: int = 1024) -> float:
    """exp(mean negative log-likelihood) over `token_ids` [1, N], scored in non-overlapping windows.

    `logits_fn(ids [1, W]) -> logits [1, W, V]`. Within a window, position t predicts token t+1, so
    each window scores W-1 tokens. Non-overlapping windows give the first tokens of each window little
    context, which raises perplexity a little; that's fine for a regression check as long as every run
    uses the same window. Log-softmax runs in fp32: bf16 log-probs would add their own noise.
    """
    total_nll, total_tokens = 0.0, 0
    for start in range(0, token_ids.shape[1] - 1, window):
        chunk = token_ids[:, start:start + window]
        if chunk.shape[1] < 2:
            break
        logits = logits_fn(chunk).float()
        nll = F.cross_entropy(logits[0, :-1], chunk[0, 1:].to(logits.device), reduction="sum")
        total_nll += nll.item()
        total_tokens += chunk.shape[1] - 1
    return math.exp(total_nll / total_tokens)


# ---------- GSM8K ----------

_CALC = re.compile(r"<<[^>]*>>")  # GSM8K's calculator annotations, e.g. <<9*2=18>>
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _solution_text(answer: str) -> tuple[str, str]:
    """GSM8K answer field -> (reasoning on one line, final number)."""
    reasoning, final = answer.split("####")
    return _CALC.sub("", reasoning).strip().replace("\n", " "), final.strip().replace(",", "")


def gsm8k_prompt(question: str, shots: list[dict]) -> str:
    parts = []
    for s in shots:
        reasoning, final = _solution_text(s["answer"])
        parts.append(f"Question: {s['question']}\nAnswer: {reasoning} The answer is {final}.\n\n")
    return "".join(parts) + f"Question: {question}\nAnswer:"


def extract_answer(completion: str) -> str | None:
    """The number after "The answer is", else the last number in the first answer paragraph."""
    completion = completion.split("\n\n")[0].split("Question:")[0]
    m = re.search(r"The answer is\s*\$?(" + _NUMBER.pattern + ")", completion)
    if m:
        found = m.group(1)
    else:
        nums = _NUMBER.findall(completion)
        if not nums:
            return None
        found = nums[-1]
    found = found.replace(",", "").rstrip(".")
    return found[:-2] if found.endswith(".0") else found


def load_gsm8k(num_questions: int, num_shots: int = 4) -> tuple[list[dict], list[dict]]:
    """(first `num_questions` test items, first `num_shots` train items as few-shot examples)."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")
    return list(ds["test"].select(range(num_questions))), list(ds["train"].select(range(num_shots)))


def gsm8k_accuracy(
    generate_fn: Callable[[str], str],
    questions: list[dict],
    shots: list[dict],
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[float, list[dict]]:
    """Fraction answered correctly. `generate_fn(prompt) -> completion text` (greedy)."""
    records, correct = [], 0
    for i, q in enumerate(questions):
        completion = generate_fn(gsm8k_prompt(q["question"], shots))
        gold = _solution_text(q["answer"])[1]
        pred = extract_answer(completion)
        correct += pred == gold
        records.append({"gold": gold, "pred": pred, "completion": completion.split("\n\n")[0]})
        if on_progress:
            on_progress(i + 1, correct)
    return correct / len(questions), records
