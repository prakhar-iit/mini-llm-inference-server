# S6 — Shared prefixes: prefix caching on vs off

**Milestone:** M6 · **Compared against:** M6 with `enable_prefix_caching=False` (same build; git tag `m5` once
tagged) · **Date:**

## Setup
- Hardware: one A100/H100 in the v1 GPU session (L4 + Llama-3.2-1B for a first pass); MacBook for the shape check
- Model / dtype: Llama-3.1-8B, bf16
- Workload, in-process (no server), one request at a time, so TTFT is queue-free prefill time:
  - Part A: 20 requests of 8,000 tokens (workload A's shape) whose first f × 8,000 tokens are shared (one paper,
    many questions), f = 0, 0.25, 0.5, 0.75, 0.9; 8 output tokens. The first request of each point is cold.
  - Part B: 24 workload C sessions (decision 2: shared per-agent-type prefix of ~3.5K tokens, growing history,
    tool outputs between turns), turns interleaved round-robin, each turn appending the real reply.
- Command(s): `python benchmarks/s6_shared_prefixes.py --model-dir models/Llama-3.1-8B --gpu a100-80`
- Correctness: `tests/test_prefix_cache.py` (fp32 tokens identical with hits, sharing, eviction and resume from
  cache; the hash spec byte for byte) and R3 with refcounts, before measuring.

## Prediction (written before running)
<!-- Fill in now, while M6 is fresh.
     1. Part A: an 8K prefill of the 8B ≈ 0.85 s on an A100 (decision 3). With a fraction f cached, the prefill
        computes (1 − f) of the tokens, but attention for those tokens still reads the whole context. Is TTFT
        ∝ (1 − f), or does it flatten at high f? What's the floor (fixed per-step overhead from S2)?
     2. At f = 0: what does hashing cost? SHA-256 of 500 blocks × 96 bytes, against a 0.85 s prefill.
     3. Part B: which share of a turn's prompt is cached: (a) the agent type's prefix on a session's first turn,
        (b) the whole previous history on every later turn? Predict the hit rate from decision 2's distributions
        (prefix ~3.5K, 6 turns, ~300 new tokens per turn).
     4. Part B's TTFT: turns get longer with history, so with caching off TTFT grows turn by turn; with caching on
        it should track the new tokens only. What does that do to p99 TTFT against the 1 s SLO (decision 3)?
     5. What breaks this: a pool too small to keep every session's history (eviction), and preemption. Which
        session state would D4's KV tiering keep instead? -->

Mac shape check (Llama-3.2-1B, MPS, 1,024-token prompts, 6 requests per point; 6 sessions with sizes scaled
by 0.2; not for the write-up):
- Part A, TTFT off → on: 1,614 → 1,632 ms at f = 0 (hashing ~1%), 1,621 → 1,484 ms at 0.25 (1.09×), 1,640 → 883 ms
  at 0.5 (1.86×), 1,641 → 462 ms at 0.75 (3.55×), 1,627 → 316 ms at 0.9 (5.15× of an ideal 10×). The gap at high
  f is the part of each step that doesn't shrink with fewer prefill tokens.
- Part B (37 turns): mean TTFT 1,865 → 400 ms, p99 2,349 → 1,724 ms; prefilled tokens 42,453 → 6,053; hit rate 85.7%.

## Result
<table and chart>

## Gap and explanation
<why the measured number differs from the prediction — this is the learning>

## What it motivates next
