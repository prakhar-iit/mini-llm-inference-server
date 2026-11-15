# S6 — Shared prefixes: prefix caching on vs off

**Milestone:** M6 · **Compared against:** M6 with `enable_prefix_caching=False` (same build; git tag `m5` once
tagged) · **Date:**

## Setup
- Hardware: one H100 (p5.4xlarge, decision 25) in the v1 GPU session (smoke run only on the L4)
- Model / dtype: Llama-3.1-8B, bf16
- Workload, in-process (no server), one request at a time, so TTFT is queue-free prefill time:
  - Part A: 20 requests of 8,000 tokens (workload A's shape) whose first f × 8,000 tokens are shared (one paper,
    many questions), f = 0, 0.25, 0.5, 0.75, 0.9; 8 output tokens. The first request of each point is cold.
  - Part B: 24 workload C sessions (decision 2: shared per-agent-type prefix of ~3.5K tokens, growing history,
    tool outputs between turns), turns interleaved round-robin, each turn appending the real reply.
- Command(s): `python benchmarks/s6_shared_prefixes.py --model-dir models/Llama-3.1-8B --gpu h100`
- Correctness: `tests/test_prefix_cache.py` (fp32 tokens identical with hits, sharing, eviction and resume from
  cache; the hash spec byte for byte) and R3 with refcounts, before measuring.

## Prediction (written before running)
<!-- Fill in now, while M6 is fresh.
     1. Part A: an 8K prefill of the 8B ≈ 0.26 s on an H100 (decision 3). With a fraction f cached, the prefill
        computes (1 − f) of the tokens, but attention for those tokens still reads the whole context. Is TTFT
        ∝ (1 − f), or does it flatten at high f? What's the floor (fixed per-step overhead from S2)?
     2. At f = 0: what does hashing cost? SHA-256 of 500 blocks × 96 bytes, against a 0.26 s prefill.
     3. Part B: which share of a turn's prompt is cached: (a) the agent type's prefix on a session's first turn,
        (b) the whole previous history on every later turn? Predict the hit rate from decision 2's distributions
        (prefix ~3.5K, 6 turns, ~300 new tokens per turn).
     4. Part B's TTFT: turns get longer with history, so with caching off TTFT grows turn by turn; with caching on
        it should track the new tokens only. What does that do to p99 TTFT against the 1 s SLO (decision 3)?
     5. What breaks this: a pool too small to keep every session's history (eviction), and preemption. Which
        session state would D4's KV tiering keep instead? -->

## Result
<table and chart>

## Gap and explanation
<why the measured number differs from the prediction — this is the learning>

## What it motivates next
