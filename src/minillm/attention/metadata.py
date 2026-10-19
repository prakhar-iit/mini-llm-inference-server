"""Per-step attention metadata for paged KV (M5, decision 7).

A step's tokens are ragged: every sequence's new tokens are concatenated into one row of N tokens, with no
padding. This metadata says which tokens belong to which sequence, where each new token's k/v goes in the
paged pool, and which blocks hold each sequence's context. It is built once per step, on the host, and every
layer reuses it.

Example: sequence 0 has 3 tokens cached and adds 1 (a decode step); sequence 1 is a fresh 5-token prompt.

    query_start_loc = [0, 1, 6]        # seq 0's new tokens are [0, 1), seq 1's are [1, 6)
    seq_lens        = [4, 5]           # context length per sequence after this step's tokens
    positions       = [3, 0, 1, 2, 3, 4]
    slot_mapping    = [b0*16 + 3, b1*16 + 0, ..., b1*16 + 4]   # flat slot = block * block_size + offset

The int32 tensors at the end are the same facts in the forms the CUDA kernels read: FlashInfer wants each
sequence's pages as one concatenated list with offsets (CSR: `kv_indptr`, `kv_indices`, `kv_last_page_len`);
flash-attn wants cumulative query lengths, context lengths and an int32 block table.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SeqStep:
    """One sequence in a step: `num_cached` tokens already in the pool, `num_new` added now."""

    num_cached: int
    num_new: int
    block_ids: list[int]  # covers num_cached + num_new tokens


@dataclass
class AttentionMetadata:
    block_size: int
    query_start_loc: list[int]  # host, len S + 1
    seq_lens: list[int]  # host: context length per sequence, this step's tokens included
    positions: torch.Tensor  # [N] RoPE position of every new token
    slot_mapping: torch.Tensor  # [N] flat pool slot each new token's k/v is written to
    block_tables: torch.Tensor  # [S, max_blocks] (long), padded with block 0 past a sequence's own blocks
    logits_index: torch.Tensor  # [S] index (into the N tokens) of each sequence's last new token
    # int32, for the CUDA kernels (M5b FlashInfer, M5c flash-attn)
    qo_indptr: torch.Tensor  # [S + 1] = query_start_loc
    seq_lens_i32: torch.Tensor  # [S]
    block_tables_i32: torch.Tensor  # [S, max_blocks]
    kv_indptr: torch.Tensor  # [S + 1] offsets into kv_indices
    kv_indices: torch.Tensor  # the pages each sequence uses, concatenated
    kv_last_page_len: torch.Tensor  # [S] tokens in each sequence's last page

    @property
    def max_query_len(self) -> int:
        return max(self.num_new)

    @property
    def max_seq_len(self) -> int:
        return max(self.seq_lens)

    @property
    def num_seqs(self) -> int:
        return len(self.seq_lens)

    @property
    def num_new(self) -> list[int]:
        q = self.query_start_loc
        return [q[i + 1] - q[i] for i in range(len(q) - 1)]

    @classmethod
    def build(cls, seqs: list[SeqStep], block_size: int, device: torch.device | str) -> AttentionMetadata:
        starts, lens, positions, slots = [0], [], [], []
        for s in seqs:
            end = s.num_cached + s.num_new
            if len(s.block_ids) * block_size < end:
                raise ValueError(f"{len(s.block_ids)} blocks can't hold {end} tokens")
            starts.append(starts[-1] + s.num_new)
            lens.append(end)
            for p in range(s.num_cached, end):
                positions.append(p)
                slots.append(s.block_ids[p // block_size] * block_size + p % block_size)
        width = max(len(s.block_ids) for s in seqs)
        tables = [s.block_ids + [0] * (width - len(s.block_ids)) for s in seqs]
        pages = [s.block_ids[:-(-n // block_size)] for s, n in zip(seqs, lens)]  # only the pages in use
        kv_indptr = [0]
        for pg in pages:
            kv_indptr.append(kv_indptr[-1] + len(pg))
        i32 = dict(dtype=torch.int32, device=device)
        return cls(
            block_size=block_size,
            query_start_loc=starts,
            seq_lens=lens,
            positions=torch.tensor(positions, dtype=torch.long, device=device),
            slot_mapping=torch.tensor(slots, dtype=torch.long, device=device),
            block_tables=torch.tensor(tables, dtype=torch.long, device=device),
            logits_index=torch.tensor(starts[1:], dtype=torch.long, device=device) - 1,
            qo_indptr=torch.tensor(starts, **i32),
            seq_lens_i32=torch.tensor(lens, **i32),
            block_tables_i32=torch.tensor(tables, **i32),
            kv_indptr=torch.tensor(kv_indptr, **i32),
            kv_indices=torch.tensor([b for pg in pages for b in pg], **i32),
            kv_last_page_len=torch.tensor([n - (len(pg) - 1) * block_size for n, pg in zip(lens, pages)], **i32),
        )
