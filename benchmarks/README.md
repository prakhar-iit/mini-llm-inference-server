# Benchmarks

One harness, reused from M2 through Phase 5. The same client runs against this engine and vLLM, so
comparisons are fair.

## Workloads

| Workload | Shape | Data | Stresses |
|---|---|---|---|
| **A — Summarize** | ~8,000 tokens in → ~150 out | arXiv papers (`ccdv/arxiv-summarization`) | Prefill (compute-bound) |
| **B — Generate** | ~100 tokens in → ~1,000 out | UltraChat-200k first-turn prompts of 40–250 tokens (ShareGPT's only public copy is one 670 MB JSON; same shape, streams) | Decode (memory-bound) |
| **C — Agent** | Shared system prompt + tool schemas (~3–4K tokens), multi-turn sessions, tool-call pauses | Synthetic generator (`workloads.py`, decision 2) | Prefix reuse, session state, bursty arrivals |
| **Mix — A+B** | A and B interleaved (e.g. 30/70 by request) | Both (`workloads.mix`) | Scheduler fairness, head-of-line blocking |

One in-house client drives all three workloads against any OpenAI-compatible server (ours or vLLM's).
It is cross-checked against `vllm bench serve` in the GPU session (decision 5).

## Tools

| File | What it does |
|---|---|
| `workloads.py` | Builds requests as token ids: `random`, A (arXiv), B (UltraChat), C (agent sessions; JSONL save/load) |
| `client.py` | Open-loop client for any OpenAI-compatible server: gamma arrivals, per-token timestamps, sessions, SLO summary |
| `s0_no_cache.py` … `s4_mixed_traffic.py` | Study scripts (S0–S4); each docstring has the full and the Mac smoke command |
| `g2_quality.py` | G2 quality gate (perplexity + GSM8K) with baseline save/check |

```bash
python -m minillm.server --model-dir models/Llama-3.2-1B --timeline-log results/timeline.jsonl &
python benchmarks/client.py --base-url http://127.0.0.1:8000 --workload b --tokenizer models/Llama-3.2-1B \
    --num 1200 --rate 2 --out results/b.jsonl
```

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
