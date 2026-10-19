# 07 — Attention backend and KV layout

**Decided:** 2026-10-18 · **Needed by:** M5 · **Status:** accepted

## Choice
- **M5 is built in three stages, all on one KV layout:**
  - **M5a, paged KV with `TorchRefBackend`** (Mac): block manager, block tables, paged cache, ragged step
    metadata, eviction by recompute, R3. The reference backend gathers KV through the block table and runs SDPA.
    It's the only path on the Mac and the correctness oracle for both kernels below.
  - **M5b, FlashInfer** (CUDA): its paged prefill and decode wrappers read the same pages.
  - **M5c, flash-attn**, using vLLM's fork (`vllm-flash-attn`), whose paged path runs at block size 16 (CUDA).
  - **Order:** M5a → M5b → M5c → M6, all coded on the Mac. M5b and M5c can't run there: their tests skip
    without CUDA, and both kernels are first exercised in one GPU session, together with every other GPU check.
- **Block size: 16 tokens.**
- **Layout: per layer, separate K and V tensors, each `[num_blocks, 16, Hkv, Dh]` (NHD, token-major inside a
  block).** Both FlashInfer (K and V passed as a pair) and vLLM's flash-attn read this shape directly, so M5b and
  M5c never change the cache. A block id means the same slot in every layer, so one block table serves all of them.
- **Ragged batches, no padding.** From M5a, a step's tokens are flattened to `[num_tokens, D]` and described by
  metadata built once per step: `query_start_loc`, `context_lens`, `block_tables`, and `slot_mapping` (the flat
  slot `block × 16 + offset` of each new token). Every layer reuses it, and FlashInfer's `plan()` runs once per
  step, not once per layer. This replaces both M3's padding and M4's right-padded prefill, and it's the shape M7
  needs to mix prefill chunks with decode tokens.
- **Backend interface:** `prefill`/`decode` become one `forward(q, k_new, v_new, layer_kv, meta)`. It writes the
  new tokens through `slot_mapping` and attends over the cached context through `block_tables`. Whether a row is
  prefill or decode comes from `meta`. The three backends implement exactly this.
- **KV pool size:** on CUDA, `num_blocks` = (GPU memory × `--gpu-memory-utilization` − weights − peak
  activations, from a profiling forward at startup) ÷ block bytes. On the Mac, a fixed `--kv-cache-gb`
  (default 1).

## Numbers behind it
Bytes per block per layer = 2 (K and V) × 16 × Hkv × Dh × 2 (bf16):

| Model | One block, one layer | One block, all layers | 8K-token prompt, one layer |
|---|---|---|---|
| Llama-3.2-1B (16 layers, 8 KV heads, Dh 64) | 32 KB | 512 KB | 16 MB |
| Llama-3.1-8B (32 layers, 8 KV heads, Dh 128) | 64 KB | 2 MB | 32 MB |

- Waste from partly filled last blocks averages 7.5 tokens per request, which is under 5% for any request longer
  than about 150 tokens. That's S5's prediction for paged vs contiguous.
- M6 (prefix caching) hashes whole blocks. With 16 tokens per block, a shared prefix is reused to within 15 tokens.

## Why
- **Three stages:** M5a holds nearly all the logic and is the only stage the Mac can run. Each kernel then drops
  in behind the same interface and is checked against `TorchRefBackend` on the same GPU, so a kernel bug can't
  hide behind a scheduling bug. The two kernels together give S5 (and S11) a three-way kernel comparison on
  identical pages. Writing both kernels' glue right after M5a, while the interface is fresh, means the one GPU
  session tests finished code instead of writing it on a billed instance.
- **FlashInfer and vLLM's flash-attn:** they're what SGLang and vLLM run, so the comparison is against real
  serving kernels. Both take a page size of 16, both separate planning from running (so CUDA graphs fit at M8).
- **Block size 16:** it's the common default (vLLM, SGLang). It's small enough for low waste and fine prefix
  hits, and large enough that block tables and transfer lists stay short.
- **Per-layer tensors (layer-major):** D3/N3 can send layer *l*'s blocks as soon as layer *l*'s prefill
  finishes, so the transfer overlaps the remaining prefill instead of waiting for all of it. It also matches
  both kernels' per-layer call.
- **Separate K and V, NHD:** the one shape both kernels accept. A token's K (or V) for all heads is one
  contiguous write. The layout is a single constant owned by the cache class, so switching to HND later is a
  local change.

## Rejected
- **Upstream flash-attn 2's paged path:** it requires blocks that are multiples of 256 tokens. That forces either
  a different block size for M5c (an unfair comparison) or 256 everywhere, which wastes up to 255 tokens per
  request and makes M6's prefix hits coarse.
- **Combined `[num_blocks, 2, 16, Hkv, Dh]` per layer:** FlashInfer reads it, but vLLM's flash-attn wants K and V
  separately, so M5c would need a different cache.
- **Block-major (one tensor `[num_blocks, L, …]`):** a whole block is contiguous across layers, which suits D4's
  offload of entire blocks to DRAM. But it serializes P/D transfer behind the full prefill, and every layer's
  attention reads a strided slice. D4 can batch its copies of one block's L pieces instead.
- **Block size 32 or 64:** shorter tables and bigger transfer chunks, but 2–4× the waste and coarser prefix hits.
  Revisit if S5's profile shows block-table overhead.
- **Keeping padded `[B, T]` batches:** the padding compute grows with length variance (the Mix workload), and
  M7 couldn't mix prefill and decode without it becoming ragged anyway.
- **Running each kernel stage on a GPU as it's written:** several short billed sessions instead of one, for
  code that is mostly glue.

## Costs to watch
- **Kernel code untested until the GPU session:** M5b and M5c are written against the libraries' documented
  APIs and land with skipped tests, and M6 is built before any kernel has touched the pages. The layout was
  chosen to be what both kernels read, but a surprise in the session would mean reworking M5a–M6. Keep the kernel
  glue thin (the shared interface does the bookkeeping) so most bugs stay findable on the Mac.
- **Two CUDA dependencies:** `flashinfer-python` and `vllm-flash-attn` pin torch/CUDA versions and may not agree.
  If they conflict, M5c installs in its own environment for the GPU session.
- **D3/N3 transfers gather:** a request's blocks are scattered in the pool, so an 8K prompt on the 8B is 512
  separate 64 KB pieces per layer (K and V each). NCCL needs them packed first (one gather kernel per layer);
  NIXL takes the list directly. Measure it in D3.
- **HND for mismatched TP degrees:** decision 18 may need HND, because each head's tokens are contiguous inside
  a block, so a TP=8 shard can be resliced for a TP=4 engine. vLLM switches its KV layout to HND for that case.
  Put the layout in the cache key (decision 18) so the two never mix.
- **The Mac path is slow:** gathering through the block table copies the whole context every step. It's fine
  for correctness, and Mac numbers never go into a study anyway.
- **The one GPU session (g6.xlarge, about 2–3 hours, after M6):** each kernel's logits vs `TorchRefBackend` on the
  same L4 within G1b's tolerance; `plan()` reuse across layers confirmed in a profile; decode step time of the
  three backends and M4's contiguous slots at batch 1–32.
