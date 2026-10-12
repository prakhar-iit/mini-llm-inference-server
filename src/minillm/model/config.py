"""Llama model config, read from a Hugging Face `config.json` (or an HF config object's dict)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    # None for plain RoPE; for Llama 3.x a dict with rope_type="llama3", factor, low/high_freq_factor,
    # original_max_position_embeddings.
    rope_scaling: dict[str, Any] | None = None
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 8192
    eos_token_ids: tuple[int, ...] = field(default_factory=tuple)

    @property
    def num_kv_groups(self) -> int:
        """Query heads per KV head (GQA)."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LlamaConfig:
        # transformers >= 5 nests theta and scaling under `rope_parameters`; raw hub config.json files
        # use top-level `rope_theta` + `rope_scaling`. Accept both.
        rope = d.get("rope_parameters") or {}
        scaling = d.get("rope_scaling") or (rope if rope.get("rope_type", "default") != "default" else None)
        if scaling is not None:
            scaling = dict(scaling)
            scaling.setdefault("rope_type", scaling.get("type"))
            if scaling["rope_type"] in (None, "default"):
                scaling = None

        eos = d.get("eos_token_id")
        eos_ids = tuple(eos) if isinstance(eos, list) else (() if eos is None else (eos,))

        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d.get("num_key_value_heads", d["num_attention_heads"]),
            head_dim=d.get("head_dim") or d["hidden_size"] // d["num_attention_heads"],
            rms_norm_eps=d["rms_norm_eps"],
            rope_theta=rope.get("rope_theta", d.get("rope_theta", 10000.0)),
            rope_scaling=scaling,
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            max_position_embeddings=d.get("max_position_embeddings", 8192),
            eos_token_ids=eos_ids,
        )

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> LlamaConfig:
        return cls.from_dict(json.loads((Path(model_dir) / "config.json").read_text()))
