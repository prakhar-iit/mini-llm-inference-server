"""G1b (decision 4): bf16 batched logits vs a batch-1 reference, with a tolerance calibrated from HF's own noise.

In bf16, a correct batched engine does not reproduce batch-1 logits exactly: kernels reduce in a different
order at different shapes. So the tolerance is measured, not guessed: run HF itself at batch 1 and at batch N
on the same sequences and record how far its logits move. Our engine passes if its batch-N-vs-batch-1 movement
is at most twice HF's, and it agrees on the top token wherever the reference's lead is clear.

Teacher forcing: every sequence is prompt + a FIXED continuation fed as input, so a near-tie that flips one
token can't send the rest of the sequence somewhere else; every position stays comparable.
Reused at M5 (FlashInfer vs torch_ref) and D1 (tensor parallel vs TP=1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .generate import pad_left

# forward_fn(input_ids [B, L], attention_mask [B, L] | None) -> logits [B, L, V]
ForwardFn = Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]

# Floors so a backend with (near-)zero noise doesn't make the tolerance zero. 0.125 is one bf16 step at a
# logit magnitude of 16-32, typical for Llama's top logits.
KL_FLOOR = 1e-5
MAX_DIFF_FLOOR = 0.125
MARGIN = 0.5  # top-1 must agree wherever the reference's top-1/top-2 gap exceeds this many logits


@torch.inference_mode()
def teacher_forced_logits(forward_fn: ForwardFn, sequences: list[list[int]], start: list[int], batch_size: int,
                          device) -> list[torch.Tensor]:
    """Logits at positions start[i]-1 .. end of each sequence (the positions that predict the continuation),
    computed in batches of `batch_size` with left padding. Returned as float32 on CPU, one [n_i, V] each."""
    out: list[torch.Tensor] = []
    for b in range(0, len(sequences), batch_size):
        chunk = sequences[b:b + batch_size]
        ids, mask = pad_left(chunk, device)
        logits = forward_fn(ids, mask)
        for row, seq in enumerate(chunk):
            n = len(seq) - (start[b + row] - 1)  # predictions for continuation tokens, plus the next one
            out.append(logits[row, -n:].float().cpu())
    return out


@dataclass
class Drift:
    mean_kl: float  # mean over positions of KL(reference || test)
    max_abs_diff: float  # largest |Δlogit| anywhere
    top1_disagreements: int  # positions with a clear reference lead (> MARGIN) where top-1 differs
    positions: int


def drift(reference: list[torch.Tensor], test: list[torch.Tensor]) -> Drift:
    kls, max_diff, bad, n = [], 0.0, 0, 0
    for ref, tst in zip(reference, test):
        lp_ref, lp_tst = F.log_softmax(ref, -1), F.log_softmax(tst, -1)
        kls.append((lp_ref.exp() * (lp_ref - lp_tst)).sum(-1))
        max_diff = max(max_diff, (ref - tst).abs().max().item())
        top2 = ref.topk(2, dim=-1).values
        clear = (top2[:, 0] - top2[:, 1]) > MARGIN
        bad += int((clear & (ref.argmax(-1) != tst.argmax(-1))).sum())
        n += ref.shape[0]
    return Drift(torch.cat(kls).mean().item(), max_diff, bad, n)


@dataclass
class G1bResult:
    passed: bool
    noise: Drift  # HF (or another trusted implementation) batch N vs batch 1
    ours: Drift  # engine under test, batch N vs its own batch 1
    kl_limit: float
    max_diff_limit: float


def g1b_check(noise: Drift, ours: Drift) -> G1bResult:
    kl_limit = 2 * max(noise.mean_kl, KL_FLOOR)
    diff_limit = 2 * max(noise.max_abs_diff, MAX_DIFF_FLOOR)
    passed = ours.mean_kl <= kl_limit and ours.max_abs_diff <= diff_limit and ours.top1_disagreements == 0
    return G1bResult(passed, noise, ours, kl_limit, diff_limit)
