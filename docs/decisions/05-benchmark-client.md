# 05 — Benchmark client

**Decided:** 2026-10-11 · **Needed by:** M2 · **Status:** accepted (changes the plan's original recommendation)

## Choice
One client written in-house, `benchmarks/client.py` (asyncio, streaming HTTP), drives all three workloads
against any OpenAI-compatible server: ours and vLLM's.

## What it does
- Open-loop arrivals: Poisson or gamma with a burstiness knob (decision 3). Records its own lag between
  scheduled and actual send time, so a client bottleneck shows up instead of hiding.
- Streams every response and timestamps every token: TTFT, TPOT, pooled ITL, E2E (decision 3).
- Workload C sessions: token-id prompts, real replies appended to the history, tool pauses, seeded JSONL
  replay (decision 2).
- Excludes the 30 s warm-up, requires ≥ 1,000 requests per point, computes goodput against the SLOs.
- Sends a request id and joins the server's per-request timeline (queue, prefill, decode; decision 6).
- Runs on the Mac for smoke tests.

## Credibility check
In the first GPU session, run one workload-A and one workload-B configuration with both this client and
`vllm bench serve` against the same server. p50/p99 TTFT and TPOT must agree within ~5%; if they don't,
find out why before any study number is used.

## Why it changes the plan
The plan recommended `vllm bench serve` for A/B before two later decisions: no GPU during development
(vLLM can't be installed on the Mac, so A/B couldn't be smoke-tested) and workload C's token-id sessions
with exact timer definitions (a second tool would measure A/B differently from C).

## Rejected
- `vllm bench serve` for A/B + own client for C: Linux-GPU-only for A/B; two metric implementations.
- genai-perf or guidellm for A/B + own client for C: same split, less common tool.

## Costs to watch
~400–600 lines of client code; a Python client can bottleneck at very high request rates (not expected
at tens to hundreds of requests/s; the send-lag metric shows it if it happens).
