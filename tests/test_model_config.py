import pytest
import torch
from transformers import LlamaConfig as HFLlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from minillm.model.config import LlamaConfig
from minillm.model.rope import compute_inv_freq

# Llama-3.2-1B's config.json as published on the Hub (the fields we read).
LLAMA_32_1B = {
    "vocab_size": 128256,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "tie_word_embeddings": True,
    "max_position_embeddings": 131072,
    "eos_token_id": 128001,
}

LLAMA3_ROPE = LLAMA_32_1B["rope_scaling"]


def test_parses_hub_config():
    c = LlamaConfig.from_dict(LLAMA_32_1B)
    assert c.num_kv_groups == 4
    assert c.rope_theta == 500000.0
    assert c.rope_scaling["rope_type"] == "llama3"
    assert c.tie_word_embeddings
    assert c.eos_token_ids == (128001,)


def _tiny_hf_config(rope_parameters):
    return HFLlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, rope_parameters=rope_parameters,
    )


@pytest.mark.parametrize(
    "rope_parameters",
    [{"rope_type": "default", "rope_theta": 10000.0}, {**LLAMA3_ROPE, "rope_theta": 500000.0}],
    ids=["default", "llama3"],
)
def test_inv_freq_matches_hf(rope_parameters):
    hf_config = _tiny_hf_config(rope_parameters)
    ours = compute_inv_freq(LlamaConfig.from_dict(hf_config.to_dict()))
    theirs = LlamaRotaryEmbedding(hf_config).inv_freq
    torch.testing.assert_close(ours, theirs.float())
