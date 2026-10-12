"""Per-layer checks against Hugging Face, so each M0 piece can be verified before the full model runs."""

import pytest
import torch
from transformers import LlamaConfig as HFLlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention as HFAttention
from transformers.models.llama.modeling_llama import LlamaMLP as HFMLP
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFRotary
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as hf_apply_rotary
from transformers.models.llama.modeling_llama import LlamaRMSNorm as HFRMSNorm

from minillm.model.config import LlamaConfig
from minillm.attention import get_backend
from minillm.model.llama import LlamaAttention, LlamaMLP, RMSNorm, RotaryEmbedding, apply_rotary

HF_TINY = HFLlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2)
TINY = LlamaConfig.from_dict(HF_TINY.to_dict())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_rmsnorm_matches_hf(dtype):
    g = torch.Generator().manual_seed(0)
    dim = 64
    weight = torch.randn(dim, generator=g)
    ours, hf = RMSNorm(dim, eps=1e-5), HFRMSNorm(dim, eps=1e-5)
    with torch.no_grad():
        ours.weight.copy_(weight)
        hf.weight.copy_(weight)
    ours, hf = ours.to(dtype), hf.to(dtype)

    x = (torch.randn(2, 5, dim, generator=g) * 3).to(dtype)
    out = ours(x)
    assert out.dtype == dtype
    torch.testing.assert_close(out, hf(x), atol=0, rtol=0)  # same ops in the same order: bit-identical


def test_mlp_matches_hf():
    torch.manual_seed(0)
    hf = HFMLP(HF_TINY)
    ours = LlamaMLP(TINY)
    ours.load_state_dict(hf.state_dict())  # same parameter names: gate_proj, up_proj, down_proj

    x = torch.randn(2, 5, TINY.hidden_size)
    out = ours(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out, hf(x), atol=0, rtol=0)


LLAMA3_ROPE = {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 32.0, "low_freq_factor": 1.0,
               "high_freq_factor": 4.0, "original_max_position_embeddings": 64}


@pytest.mark.parametrize("rope", [None, LLAMA3_ROPE], ids=["default", "llama3"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_rope_matches_hf(rope, dtype):
    hf_config = HFLlamaConfig(**{**HF_TINY.to_dict(), **({"rope_parameters": rope} if rope else {})})
    config = LlamaConfig.from_dict(hf_config.to_dict())
    g = torch.Generator().manual_seed(0)
    # Uneven positions per row (as later batched/cached steps will have), including large ones.
    positions = torch.tensor([[0, 1, 2, 3, 4], [7, 100, 1000, 5000, 9000]])
    q = torch.randn(2, config.num_attention_heads, 5, config.head_dim, generator=g).to(dtype)
    k = torch.randn(2, config.num_key_value_heads, 5, config.head_dim, generator=g).to(dtype)

    cos, sin = RotaryEmbedding(config)(positions, dtype)
    hf_cos, hf_sin = HFRotary(hf_config)(q, positions)
    torch.testing.assert_close(cos, hf_cos, atol=0, rtol=0)
    torch.testing.assert_close(sin, hf_sin, atol=0, rtol=0)

    hf_q, hf_k = hf_apply_rotary(q, k, hf_cos, hf_sin)
    torch.testing.assert_close(apply_rotary(q, cos, sin), hf_q, atol=0, rtol=0)
    torch.testing.assert_close(apply_rotary(k, cos, sin), hf_k, atol=0, rtol=0)


def test_rope_score_depends_only_on_relative_position():
    # The property that makes RoPE work: q·k after rotation depends on (m - n), not on m and n.
    config = TINY
    rope = RotaryEmbedding(config)
    g = torch.Generator().manual_seed(1)
    q, k = (torch.randn(1, 1, 1, config.head_dim, generator=g) for _ in range(2))

    def score(m, n):
        cm, sm = rope(torch.tensor([[m]]), torch.float32)
        cn, sn = rope(torch.tensor([[n]]), torch.float32)
        return (apply_rotary(q, cm, sm) * apply_rotary(k, cn, sn)).sum()

    torch.testing.assert_close(score(10, 7), score(503, 500), atol=1e-4, rtol=1e-4)
    assert not torch.allclose(score(10, 7), score(10, 9))


@pytest.mark.parametrize("hkv", [2, 4, 1], ids=["gqa", "mha", "mqa"])
def test_attention_matches_hf(hkv):
    hf_config = HFLlamaConfig(**{**HF_TINY.to_dict(), "num_key_value_heads": hkv})
    hf_config._attn_implementation = "sdpa"
    config = LlamaConfig.from_dict(hf_config.to_dict())
    torch.manual_seed(0)
    hf = HFAttention(hf_config, layer_idx=0).eval()
    ours = LlamaAttention(config, get_backend("torch_ref"))
    ours.load_state_dict(hf.state_dict())  # q_proj, k_proj, v_proj, o_proj

    b, t = 2, 9
    x = torch.randn(b, t, config.hidden_size)
    positions = torch.arange(t).expand(b, t)
    cos, sin = RotaryEmbedding(config)(positions, x.dtype)
    causal = torch.full((t, t), float("-inf")).triu(1)[None, None]  # HF takes an explicit additive mask
    expected, _ = hf(x, position_embeddings=(cos, sin), attention_mask=causal)
    torch.testing.assert_close(ours(x, cos, sin), expected, atol=1e-5, rtol=1e-5)
