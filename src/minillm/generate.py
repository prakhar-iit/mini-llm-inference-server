"""Generation loops, greedy, batch 1. M0: no KV cache. M1: with a KV cache."""

from __future__ import annotations

import torch

from .model.kv_cache import KVCache
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


@torch.inference_mode()
def greedy_generate(
    model: LlamaForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...] = (),
) -> list[int]:
    """input_ids [1, T] -> the newly generated token ids (prompt excluded). Same contract as
    `greedy_generate_no_cache`, with a KV cache.

    Prefill: one pass over the whole prompt fills the cache and yields the first token.
    Decode: each later step feeds only the newest token; attention reads every earlier token's k/v
    from the cache instead of recomputing them. Total work becomes linear in length.
    """
    if max_new_tokens <= 0:
        return []
    param = next(model.parameters())
    cache = KVCache(model.config, batch_size=1, max_len=input_ids.shape[1] + max_new_tokens,
                    device=param.device, dtype=param.dtype)

    logits = model(input_ids, kv_cache=cache, last_only=True)  # prefill: [1, 1, V]
    new_tokens: list[int] = []
    while True:
        next_id = logits[0, -1].argmax().item()
        new_tokens.append(next_id)
        if next_id in eos_token_ids or len(new_tokens) == max_new_tokens:
            return new_tokens
        next_token = torch.tensor([[next_id]], dtype=input_ids.dtype, device=input_ids.device)
        logits = model(next_token, kv_cache=cache, last_only=True)  # decode: one token in, [1, 1, V] out


def pad_left(prompts: list[list[int]], device, pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Left-pad prompts to one length: (input_ids [B, L], attention_mask [B, L] or None if none needed).

    Left, not right: every row's newest token then sits in the last column, so one decode step appends
    one column for the whole batch.
    """
    longest = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), longest), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, longest - len(p):] = torch.tensor(p)
        mask[i, longest - len(p):] = 1
    padded = any(len(p) < longest for p in prompts)
    return ids.to(device), (mask.to(device) if padded else None)


@torch.inference_mode()
def greedy_generate_batch(
    model: LlamaForCausalLM,
    prompts: list[list[int]],
    max_new_tokens: int,
) -> list[list[int]]:
    """Static batch (M3): left-pad the prompts, prefill them together, decode all rows in lockstep for
    `max_new_tokens` steps (no early stop). Each row must equal greedy_generate on that prompt alone (G1a)."""
    param = next(model.parameters())
    ids, mask = pad_left(prompts, param.device)
    cache = KVCache(model.config, len(prompts), ids.shape[1] + max_new_tokens, param.device, param.dtype)
    logits = model(ids, kv_cache=cache, last_only=True, attention_mask=mask)
    out: list[list[int]] = [[] for _ in prompts]
    for step in range(max_new_tokens):
        nxt = logits[:, -1].argmax(dim=-1)  # [B]
        for row, tok in zip(out, nxt.tolist()):
            row.append(tok)
        if step + 1 < max_new_tokens:
            logits = model(nxt[:, None], kv_cache=cache, last_only=True)
    return out
