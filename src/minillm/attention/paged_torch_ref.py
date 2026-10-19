"""Paged attention, plain PyTorch (M5a). The Mac path, and the oracle that M5b/M5c's kernels must match.

Two steps per layer, on the ragged tokens of one engine step (shapes: N new tokens, S sequences):
1. Write: each new token's k/v goes to its flat slot (`slot_mapping`) in the layer's pool.
2. Attend: for each sequence, gather its context's k/v through its block table into a contiguous copy and run
   SDPA. A fused kernel reads the blocks in place; gathering costs a copy of the whole context per step, which is
   why this backend is a reference, not a fast path.

Decode rows (one new token) are gathered and attended together, masked to each row's length. Rows with several
new tokens (a prompt, or a prompt continued on top of cached tokens) run one at a time with a causal mask.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .metadata import AttentionMetadata


def write_kv(k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor, k: torch.Tensor,
             v: torch.Tensor) -> None:
    """k, v [N, Hkv, Dh] -> pools [num_blocks, block_size, Hkv, Dh], at flat slots `slot_mapping` [N]."""
    hkv, dh = k_cache.shape[-2:]
    k_cache.view(-1, hkv, dh).index_copy_(0, slot_mapping, k.to(k_cache.dtype))
    v_cache.view(-1, hkv, dh).index_copy_(0, slot_mapping, v.to(v_cache.dtype))


def gather(cache: torch.Tensor, block_tables: torch.Tensor, length: int) -> torch.Tensor:
    """cache [num_blocks, bs, Hkv, Dh], block_tables [B, nb] -> [B, Hkv, length, Dh]: the first `length` tokens."""
    b = block_tables.shape[0]
    bs, hkv, dh = cache.shape[1:]
    nb = math.ceil(length / bs)
    tokens = cache[block_tables[:, :nb]].reshape(b, nb * bs, hkv, dh)[:, :length]
    return tokens.transpose(1, 2)


def paged_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                    meta: AttentionMetadata) -> torch.Tensor:
    """q [N, H, Dh] (new tokens, ragged) -> [N, H, Dh]. Keys and values come from the pool; this step's own
    tokens must already be written (`write_kv`). Scale 1/sqrt(Dh); grouped-query attention via enable_gqa."""
    out = torch.empty_like(q)
    starts, lens, news = meta.query_start_loc, meta.seq_lens, meta.num_new

    decode = [i for i, n in enumerate(news) if n == 1]
    if decode:
        rows = torch.tensor(decode, device=q.device)
        longest = max(lens[i] for i in decode)
        k = gather(k_cache, meta.block_tables[rows], longest)  # [Sd, Hkv, L, Dh]
        v = gather(v_cache, meta.block_tables[rows], longest)
        token_idx = torch.tensor([starts[i] for i in decode], device=q.device)
        qd = q[token_idx][:, :, None]  # [Sd, H, 1, Dh]
        mask = None
        if any(lens[i] != longest for i in decode):  # past a shorter row's end are other requests' tokens
            ends = torch.tensor([lens[i] for i in decode], device=q.device)
            mask = (torch.arange(longest, device=q.device)[None, :] < ends[:, None])[:, None, None, :]
        out[token_idx] = F.scaled_dot_product_attention(qd, k, v, attn_mask=mask, enable_gqa=True)[:, :, 0]

    for i, n in enumerate(news):
        if n == 1:
            continue
        s, length = starts[i], lens[i]
        k = gather(k_cache, meta.block_tables[i:i + 1], length)  # [1, Hkv, L, Dh]
        v = gather(v_cache, meta.block_tables[i:i + 1], length)
        qi = q[s:s + n].transpose(0, 1)[None]  # [1, H, n, Dh]
        if n == length:  # a fresh prompt: plain causal
            o = F.scaled_dot_product_attention(qi, k, v, is_causal=True, enable_gqa=True)
        else:  # continued on top of cached tokens: causal, aligned bottom-right
            mask = torch.ones(n, length, dtype=torch.bool, device=q.device).tril(diagonal=length - n)
            o = F.scaled_dot_product_attention(qi, k, v, attn_mask=mask, enable_gqa=True)
        out[s:s + n] = o[0].transpose(0, 1)
    return out
