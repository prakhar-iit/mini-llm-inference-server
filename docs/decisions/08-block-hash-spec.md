# 08 — Block-hash spec

**Decided:** 2026-10-18 · **Needed by:** M6 · **Status:** accepted (the plan's recommendation, adopted while
building M6; revisit at review)

## Choice
A block's identity is a SHA-256 chain over full blocks, salted at the root, so the same tokens share KV only when
everything that shapes the KV matches.

```
root      = SHA-256( "minillm-kv-v1" || 0x00 || canonical JSON of the extra keys )
hash[i]   = SHA-256( hash[i-1] || token_ids[16·i : 16·i + 16] as uint32 little-endian )     (hash[-1] = root)
```

- **Full blocks only.** A partly filled last block has no hash and is never shared. A block is hashed the
  moment its 16th token is cached.
- **Extra keys** (canonical JSON: keys sorted, no whitespace, UTF-8):
  `{"block_size": 16, "kv_dtype": "bfloat16", "layout": "NHD", "model": "<name>@<revision>", "salt": "<salt or empty>"}`.
  - `model` is the served model name plus the checkpoint's revision (the Hub commit, or a hash of `config.json`
    and the safetensors index when there's no revision).
  - `salt` comes from an optional `cache_salt` request field (vLLM's name for it). Requests with different salts
    never share blocks, which is P2's tenant isolation as a policy, not a migration.
  - LoRA id and TP degree join the keys when they exist (decision 18).
- **Digests are 32 raw bytes** in the engine; hex in logs, events and the router.
- **Lookup:** the longest prefix of a request's block hashes found in the cache. The last token is always
  recomputed, so a request whose prompt is fully cached still produces logits: hits are counted only over
  blocks lying entirely within its first `num_tokens − 1` tokens.

## Why
- **A specified hash, not Python's `hash()`:** `hash()` is salted per process (`PYTHONHASHSEED`), so it differs
  between engine replicas and can't be reproduced by the Go/Rust router (decision 10/11). SHA-256 is identical
  everywhere and available in every language's standard library.
- **Chained:** a block's hash commits to every token before it. Two prompts that share block 3's tokens but not
  blocks 0–2 must not share block 3's KV, because attention made its k/v depend on all earlier tokens.
- **Salted at the root:** one extra hash per request covers every key, instead of mixing keys into every block.
- **uint32 little-endian tokens:** vocabularies fit in 32 bits (Llama 3: 128,256), and the byte order is fixed
  so the router and the engine agree.

## Rejected
- **xxh3-128:** about 10× faster, but not in Go's or Rust's standard library, and at 16 tokens per block hashing
  is far below a millisecond per request either way. Revisit if S6's profile shows hashing on the hot path.
- **Python `hash()` / tuple hashing:** process-salted, as above.
- **Hashing partial blocks:** more hits on short prompts, but a block's identity would change as it fills,
  and the router's index would churn every decode step.

## Costs to watch
- SHA-256 of 64 bytes + 32 bytes per block: microseconds per 8K-token prompt. S6 checks it stays negligible.
- Changing anything in the root (model revision, dtype) invalidates the whole cache. That's intended.
