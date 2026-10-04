# Benchmarks

One harness, reused from M2 through Phase 5. The same client runs against this engine and vLLM, so
comparisons are fair.

## Workloads

| Workload | Shape | Data | Stresses |
|---|---|---|---|
| **A — Summarize** | ~8,000 tokens in → ~150 out | arXiv papers (`ccdv/arxiv-summarization`) | Prefill (compute-bound) |
| **B — Generate** | ~100 tokens in → ~1,000 out | ShareGPT conversations | Decode (memory-bound) |
| **C — Agent** | Shared system prompt + tool schemas (~3–4K tokens), multi-turn sessions, tool-call pauses | Synthetic generator (spec TBD — plan decision 2) | Prefix reuse, session state, bursty arrivals |

A and B can use `vllm bench serve` as the load generator against our OpenAI-compatible server. C needs
its own generator.

## Metrics

TTFT p50/p99 · ITL (TPOT) p50/p99 · output tokens/s · goodput (requests/s meeting the SLO) · max
concurrency at SLO · KV memory per request · prefill MFU · decode MBU.

SLO targets are not set yet (plan decision 3).

## Output

Raw results go to `results/` (git-ignored). Each study's summary and charts go to `docs/studies/`.
