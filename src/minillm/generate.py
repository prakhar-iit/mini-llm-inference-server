"""Generation loops. M0: greedy, batch 1, no KV cache."""

from __future__ import annotations

import torch

from .model.llama import LlamaForCausalLM


@torch.inference_mode()
def greedy_generate_no_cache(
    model: LlamaForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...] = (),
) -> list[int]:
    """input_ids [1, T] -> the newly generated token ids (prompt excluded).

    No cache: every step runs the model over the whole sequence so far and takes argmax of the
    last position's logits. Stop after `max_new_tokens`, or right after emitting a token in
    `eos_token_ids` (include that token in the output, as HF does).

    This is deliberately O(n²) in total work; S0 measures exactly that.
    """
    new_tokens: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(input_ids)  # [1, T, V]: recomputes every position, every step (no cache)
        next_id = logits[0, -1].argmax().item()  # only the last position predicts the next token
        new_tokens.append(next_id)
        if next_id in eos_token_ids:
            break
        next_token = torch.tensor([[next_id]], dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([input_ids, next_token], dim=1)  # [1, T] -> [1, T+1]
    return new_tokens
