# Benchmarks

One harness, reused from M2 through Phase 5. The same client runs against this engine and vLLM, so
comparisons are fair.

## Workloads

| Workload | Shape | Data | Stresses |
|---|---|---|---|
| **A — Summarize** | ~8,000 tokens in → ~150 out | arXiv papers (`ccdv/arxiv-summarization`) | Prefill (compute-bound) |
| **B — Generate** | ~100 tokens in → ~1,000 out | ShareGPT conversations | Decode (memory-bound) |
| **C — Agent** | Shared system prompt + tool schemas (~3–4K tokens), multi-turn sessions, tool-call pauses | Synthetic generator (spec TBD — plan decision 2) | Prefix reuse, session state, bursty arrivals |

| **Mix — A+B** | A and B interleaved (e.g. 30/70 by request) | Both | Scheduler fairness, head-of-line blocking |

One in-house client drives all three workloads against any OpenAI-compatible server (ours or vLLM's).
It is cross-checked against `vllm bench serve` in the GPU session (decision 5).

## Standard sweeps

Vary one thing, hold the rest: prompt length (128 → 16K), output length (1 → 2,000), batch/concurrency
(1 → 64), Poisson arrival rate (up to the latency knee), KV memory budget, shared-prefix fraction (0 → 90%).

## Gates and robustness tests

Run before any study is measured (details in the plan's "Testing program"):
- **G0–G2 correctness/quality:** greedy match vs HF, teacher-forced logit tolerance, perplexity + GSM8K subset.
- **R1 cancellation** (M2), **R2 overload/backpressure** (M4), **R3 KV block accounting** (M5),
  **R4 soak** 1–2 h (v1, v1.1, D5).

## Metrics

TTFT p50/p99 · ITL (TPOT) p50/p99 · output tokens/s · goodput (requests/s meeting the SLO) · max
concurrency at SLO · KV memory per request · prefix-cache hit rate · error/rejection/preemption rates ·
prefill MFU · decode MBU. Each request's TTFT is also split into HTTP, tokenization, queue, prefill
and emission.

SLO targets are not set yet (plan decision 3).

## Output

Raw results go to `results/` (git-ignored). Each study's summary and charts go to `docs/studies/`.
