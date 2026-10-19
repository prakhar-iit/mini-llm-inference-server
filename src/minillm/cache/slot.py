"""Slot KV cache (M4, continuous batching): one contiguous slot per running request.

The M4 engine kept its running requests here; since M5 the engine uses paged blocks (`paged.py`), and this
cache stays as the contiguous baseline S5 measures paged KV against (git tag m4 runs it in the engine).
"""

from __future__ import annotations

import torch

from ..model.config import LlamaConfig


class SlotKVCache:
    """Contiguous KV cache with one slot per running request (M4, continuous batching).

    Each layer keeps K and V buffers of shape [S, Hkv, max_len, Dh], allocated once. Slot s holds one
    request's tokens at [0, lens[s]); every slot reserves max_len, which is the contiguous baseline S5
    compares paged KV against. The engine keeps its running requests in slots [0, n) (it moves the last
    request into a slot that frees up), so a decode step works on a view of the buffers, never a gather.
    Slots are reused without clearing: anything at or past lens[s] is stale and is masked, never read.
    Lengths live in Python, so building a step's masks and positions never waits on the GPU.
    """

    def __init__(self, config: LlamaConfig, num_slots: int, max_len: int, device: torch.device | str,
                 dtype: torch.dtype) -> None:
        shape = (num_slots, config.num_key_value_heads, max_len, config.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        self.max_len = max_len
        self.device = torch.device(device)
        self.lens = [0] * num_slots  # tokens held per slot

    def batch(self, start: int, stop: int, num_new: list[int]) -> SlotBatch:
        """A view over slots [start, stop) for one forward pass that adds num_new[i] real tokens to row i.

        Prefill: fresh slots, input right-padded to max(num_new). Decode: one token per row.
        """
        return SlotBatch(self, start, stop, num_new)

    def move(self, src: int, dst: int) -> None:
        """Copy slot src's tokens into slot dst (compaction when a request leaves); src becomes empty."""
        n = self.lens[src]
        for buf in self.k + self.v:
            buf[dst, :, :n] = buf[src, :, :n]
        self.lens[dst], self.lens[src] = n, 0

    def free(self, slot: int) -> None:
        self.lens[slot] = 0


class SlotBatch:
    """The cache as one forward pass sees it: the methods `LlamaModel` calls on a KV cache, for a run of slots.

    Row i starts with old[i] tokens and gains num_new[i]. Two shapes are supported: every row at the same
    offset (a prefill into fresh slots, right-padded: real tokens first, so causality alone keeps real queries
    off the padding), or one new token per row at its own offset (a decode step, written with a scatter).
    """

    def __init__(self, cache: SlotKVCache, start: int, stop: int, num_new: list[int]) -> None:
        self.cache, self.start, self.stop = cache, start, stop
        self.old = cache.lens[start:stop]
        self.num_new = num_new
        self.width = 0  # T, the padded number of new tokens in this call (set by `mark`)
        ends = [o + n for o, n in zip(self.old, num_new)]
        if max(ends) > cache.max_len:
            raise ValueError(f"KV cache full: {max(ends)} tokens > max_len {cache.max_len}")
        self._ends = ends
        self._valid: torch.Tensor | None = None
        self._rows = self._offsets = None

    def positions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Row i's new tokens sit at positions old[i], old[i] + 1, ... (padding gets positions too; it's masked)."""
        t = input_ids.shape[1]
        old = torch.tensor(self.old, device=input_ids.device)
        return old[:, None] + torch.arange(t, device=input_ids.device)

    def mark(self, attention_mask: torch.Tensor | None, num_tokens: int) -> None:
        # Real tokens per row come from `num_new` (known on the host), not from a mask.
        if attention_mask is not None:
            raise ValueError("SlotBatch takes per-row token counts, not an attention mask")
        same_offset = len(set(self.old)) == 1
        if num_tokens > 1 and not same_offset:
            raise ValueError("a multi-token call needs every row at the same offset")
        self.width = num_tokens
        dev = self.cache.device
        self.tk = (self.old[0] + num_tokens) if same_offset else max(self.old) + 1
        if not same_offset:
            self._rows = torch.arange(self.start, self.stop, device=dev)
            self._offsets = torch.tensor(self.old, device=dev)
        if any(e != self.tk for e in self._ends):  # some row is shorter than the keys the call attends over
            ends = torch.tensor(self._ends, device=dev)
            self._valid = torch.arange(self.tk, device=dev)[None, :] < ends[:, None]

    def key_valid(self, num_new: int) -> torch.Tensor | None:
        return self._valid

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kb, vb = self.cache.k[layer_idx], self.cache.v[layer_idx]
        if self._rows is None:
            o = self.old[0]
            kb[self.start:self.stop, :, o:o + self.width] = k
            vb[self.start:self.stop, :, o:o + self.width] = v
        else:
            kb[self._rows, :, self._offsets] = k[:, :, 0]
            vb[self._rows, :, self._offsets] = v[:, :, 0]
        return kb[self.start:self.stop, :, :self.tk], vb[self.start:self.stop, :, :self.tk]

    def advance(self, num_tokens: int) -> None:
        self.cache.lens[self.start:self.stop] = self._ends
