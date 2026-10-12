"""Llama forward pass (M0).

Module and parameter names mirror Hugging Face's `LlamaForCausalLM`, so `loader.py` can copy the
safetensors weights in by name. The constructors are done; every `forward` marked `YOURS (M0)` is the
milestone. `tests/test_g0_correctness.py` is the spec: it compares logits and greedy tokens with HF on
a tiny random config (fp32, CPU), then on Llama-3.2-1B once the weights are downloaded.

Shape legend: B batch, T tokens, D hidden_size, H query heads, Hkv KV heads, Dh head_dim, V vocab.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..attention import AttentionBackend, get_backend
from .config import LlamaConfig
from .rope import compute_inv_freq


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[..., D] -> [..., D].

        x / sqrt(mean(x², last dim) + eps) * weight. The normalization runs in float32 (squaring bf16
        values loses precision), is cast back to the input dtype, then scaled by `weight`, matching HF.
        """
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        # Not persistent: it's derived from the config, so it isn't in (or expected from) the checkpoint.
        self.register_buffer("inv_freq", compute_inv_freq(config), persistent=False)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """positions [B, T] (int) -> (cos, sin), each [B, T, Dh] in `dtype`.

        angle[b, t, i] = positions[b, t] * inv_freq[i] for i in [0, Dh/2). HF's layout repeats the
        half-size angles to full width, so dim i and dim i + Dh/2 (one rotation pair) share an angle.
        Angles are computed in float32: at position 100K, bf16 can't even represent the position exactly.
        """
        angles = positions[..., None].float() * self.inv_freq.float()  # [B, T, Dh/2]
        emb = torch.cat([angles, angles], dim=-1)  # [B, T, Dh]
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """cat([-x2, x1]) over the last dim, where x1, x2 are its first and second halves."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, heads, T, Dh], cos/sin [B, T, Dh] -> rotated x, same shape.

    HF checkpoints use the "rotate half" pairing (dim i with dim i + Dh/2), not interleaved pairs.
    For a pair (a, b) at angle θ: a' = a·cos θ − b·sin θ, b' = b·cos θ + a·sin θ, which is exactly
    x * cos + rotate_half(x) * sin.
    """
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # broadcast over the heads dim
    return x * cos + rotate_half(x) * sin


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, backend: AttentionBackend) -> None:
        super().__init__()
        self.config = config
        self.backend = backend
        h, hkv, dh, d = config.num_attention_heads, config.num_key_value_heads, config.head_dim, config.hidden_size
        self.q_proj = nn.Linear(d, h * dh, bias=False)
        self.k_proj = nn.Linear(d, hkv * dh, bias=False)
        self.v_proj = nn.Linear(d, hkv * dh, bias=False)
        self.o_proj = nn.Linear(h * dh, d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x [B, T, D] -> [B, T, D]. Causal self-attention over the whole sequence (no cache in M0).

        1. Project to q [B, H, T, Dh] and k, v [B, Hkv, T, Dh] (view + transpose).
        2. Apply RoPE to q and k.
        3. self.backend.prefill(q, k, v) -> [B, H, T, Dh]. Grouped-query attention: each KV head
           serves `config.num_kv_groups` query heads; the backend handles that.
        4. Back to [B, T, H*Dh], then o_proj.
        """
        b, t, _ = x.shape
        dh = self.config.head_dim
        # [B, T, heads*Dh] -> [B, T, heads, Dh] -> [B, heads, T, Dh]: attention runs per head.
        q = self.q_proj(x).view(b, t, -1, dh).transpose(1, 2)
        k = self.k_proj(x).view(b, t, -1, dh).transpose(1, 2)
        v = self.v_proj(x).view(b, t, -1, dh).transpose(1, 2)

        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)  # v carries content, not position
        out = self.backend.prefill(q, k, v)  # [B, H, T, Dh]

        # Concatenate the heads back into one vector per token, then mix them with o_proj.
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        d, f = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(d, f, bias=False)
        self.up_proj = nn.Linear(d, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> [B, T, D]. SwiGLU: down(silu(gate(x)) * up(x))."""
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, backend: AttentionBackend) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = LlamaAttention(config, backend)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Pre-norm residual block: x + attn(norm(x)), then + mlp(norm(.)).

        Each sublayer reads a normalized copy and adds its result back onto the unnormalized stream.
        """
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig, backend: AttentionBackend) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(LlamaDecoderLayer(config, backend) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """input_ids, positions [B, T] -> hidden states [B, T, D] after the final norm.

        cos/sin depend only on positions, which are the same for every layer, so compute them once.
        """
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(positions, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig, backend: AttentionBackend | None = None) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config, backend or get_backend("torch_ref"))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """input_ids [B, T] -> logits [B, T, V]. `positions` defaults to 0..T-1 for every row."""
        if positions is None:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        return self.lm_head(self.model(input_ids, positions))
