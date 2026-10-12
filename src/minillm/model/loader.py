"""Load Hugging Face Llama weights into `LlamaForCausalLM`."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from ..attention import AttentionBackend
from .config import LlamaConfig
from .llama import LlamaForCausalLM

# Present in some older checkpoints; recomputed from the config instead.
_IGNORED_SUFFIXES = ("rotary_emb.inv_freq",)


def load_state_dict(model: LlamaForCausalLM, state_dict: dict[str, torch.Tensor], assign: bool = False) -> None:
    """Load weights by name. Fails loudly on any missing or unexpected key.

    assign=False copies into the model's existing parameters. assign=True makes the given tensors the
    parameters (no second copy in memory); use it on a model built on the meta device.
    """
    state_dict = {k: v for k, v in state_dict.items() if not k.endswith(_IGNORED_SUFFIXES)}
    if model.config.tie_word_embeddings:
        state_dict.pop("lm_head.weight", None)  # tied to embed_tokens; checkpoints may or may not include it
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=assign)
    if model.config.tie_word_embeddings:
        missing = [k for k in missing if k != "lm_head.weight"]
        # Assigning replaces the embed_tokens Parameter object, so point lm_head at the new one.
        model.lm_head.weight = model.model.embed_tokens.weight
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")


def load_model(
    model_dir: str | Path,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    backend: AttentionBackend | None = None,
) -> LlamaForCausalLM:
    """Build the model from `model_dir` (config.json + *.safetensors) on `device` in `dtype`.

    The model is built on the meta device (shapes only, no memory) and the loaded tensors become its
    parameters, so the weights exist once, already in `dtype`. Allocating real parameters first would
    hold two copies (and an fp32 one at that), which on an 8 GB Mac silently corrupted weights on MPS.
    """
    model_dir = Path(model_dir)
    config = LlamaConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = LlamaForCausalLM(config, backend)

    state_dict: dict[str, torch.Tensor] = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        state_dict.update(load_file(f, device=str(device)))
    if not state_dict:
        raise FileNotFoundError(f"no *.safetensors in {model_dir}")
    load_state_dict(model, {k: v.to(dtype) for k, v in state_dict.items()}, assign=True)

    # inv_freq is derived from the config, not stored in the checkpoint, so it is still on meta.
    model.model.rotary_emb.__init__(config)
    model.model.rotary_emb.to(device)
    leftover = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if leftover:
        raise RuntimeError(f"tensors never loaded: {leftover}")
    return model.eval()


def from_hf_model(hf_model, backend: AttentionBackend | None = None) -> LlamaForCausalLM:
    """Build our model from an in-memory HF `LlamaForCausalLM` (tests use a tiny random one)."""
    config = LlamaConfig.from_dict(hf_model.config.to_dict())
    p = next(hf_model.parameters())
    model = LlamaForCausalLM(config, backend).to(device=p.device, dtype=p.dtype)
    load_state_dict(model, hf_model.state_dict())
    return model.eval()
