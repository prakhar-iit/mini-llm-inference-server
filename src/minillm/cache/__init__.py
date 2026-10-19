"""Every KV cache the engine has used, one file each.

- `contiguous.py` — KVCache (M1, M3): one buffer per layer, every row the same length (left-padded batches).
  Used by `generate.py`, the correctness tests and the S1/S3 scripts.
- `slot.py` — SlotKVCache (M4): one contiguous slot per running request, max_model_len reserved each.
- `paged.py` — PagedKVCache (M5a, decision 7): a pool of 16-token blocks per layer; the engine's cache since M5.
- M6 adds the prefix cache (block hashes, refcounts, LRU) on top of `paged.py`.

The blocks themselves are handed out by `engine/block_manager.py`; these files only hold memory.
"""

from .contiguous import KVCache
from .paged import BLOCK_SIZE, PagedKVCache, PagedStep
from .slot import SlotBatch, SlotKVCache

__all__ = ["BLOCK_SIZE", "KVCache", "PagedKVCache", "PagedStep", "SlotBatch", "SlotKVCache"]
