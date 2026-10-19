"""Engine core: everything between the API and the model.

- M3: static batching.
- M4: continuous (iteration-level) batching scheduler (`scheduler.py`), admission control in the frontend.
- M5: block manager (`block_manager.py`), paged KV (pool in `cache/paged.py`), preemption by recompute.
- M6: prefix cache (chained block hashes, refcount, LRU).
- M7: chunked prefill (token budget per step).

Runs as its own process, split from the API frontend and the GPU workers (decision 1 in the plan).
"""
