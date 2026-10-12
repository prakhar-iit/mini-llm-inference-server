import math

import pytest
import torch

from minillm.attention import get_backend

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])


def naive_causal_gqa(q, k, v):
    """softmax(QKᵀ/√Dh + causal mask)·V, with each KV head repeated for its query heads."""
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    t = q.shape[2]
    future = torch.ones(t, t, dtype=torch.bool, device=q.device).triu(1)
    return scores.masked_fill(future, float("-inf")).softmax(-1) @ v


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("h,hkv", [(4, 4), (4, 2), (8, 1)], ids=["mha", "gqa", "mqa"])
def test_prefill_matches_naive(device, h, hkv):
    g = torch.Generator().manual_seed(0)
    b, t, dh = 2, 7, 16
    q, k, v = (torch.randn(b, n, t, dh, generator=g).to(device) for n in (h, hkv, hkv))
    out = get_backend("torch_ref").prefill(q, k, v)
    assert out.shape == (b, h, t, dh)
    torch.testing.assert_close(out, naive_causal_gqa(q, k, v), atol=1e-5, rtol=1e-5)


def test_prefill_is_causal():
    # Changing the last token's k/v must not change any earlier position's output.
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(1, 2, 5, 8, generator=g) for _ in range(3))
    backend = get_backend("torch_ref")
    before = backend.prefill(q, k, v)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, -1] += 10
    v2[:, :, -1] += 10
    after = backend.prefill(q, k2, v2)
    torch.testing.assert_close(before[:, :, :-1], after[:, :, :-1])
    assert not torch.allclose(before[:, :, -1], after[:, :, -1])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tq,tk", [(1, 9), (4, 9), (9, 9)], ids=["decode", "continue", "fresh"])
def test_prefill_on_top_of_cache_matches_naive(device, tq, tk):
    # The queries are the last tq of tk positions: compare with full attention's last tq rows.
    g = torch.Generator().manual_seed(2)
    q_all, k, v = (torch.randn(1, n, tk, 16, generator=g).to(device) for n in (4, 2, 2))
    q = q_all[:, :, -tq:]
    out = get_backend("torch_ref").prefill(q, k, v)
    torch.testing.assert_close(out, naive_causal_gqa(q_all, k, v)[:, :, -tq:], atol=1e-5, rtol=1e-5)
