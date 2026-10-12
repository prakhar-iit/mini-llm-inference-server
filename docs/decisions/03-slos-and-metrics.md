# 03 — SLOs and metric definitions

**Decided:** 2026-10-11 · **Needed by:** M2 · **Status:** accepted

## Metric definitions
Measured at the client unless noted; that is what a user sees, queueing and server overhead included.

| Metric | Definition |
|---|---|
| TTFT | Request send → first generated token received in the stream |
| TPOT (per request) | (E2E − TTFT) ÷ (output tokens − 1) |
| ITL (pooled) | Every gap between consecutive streamed tokens, across all requests; its p99 exposes stalls TPOT averages away (S7) |
| E2E | Request send → last token |
| Throughput | Output tokens/s and requests/s over the measurement window |
| Goodput | Requests/s that complete AND meet both the TTFT and TPOT SLOs. Workload C: a turn is a request; session goodput counts sessions whose every turn met the SLO |
| Capacity | Highest arrival rate at which p99 TTFT and p99 TPOT still meet the SLO (the knee) |

Engine-side timings (queue, prefill, decode, cache-hit tokens) come from the per-request timeline
(decision 6) and explain the client numbers; they don't replace them.

## Measurement rules
- **Open-loop load:** arrivals follow their schedule (Poisson or gamma) regardless of server speed. A
  closed-loop client slows down with the server and hides queueing (coordinated omission).
- **Warm-up:** the first 30 s of every run are excluded.
- **Samples:** at least 1,000 requests per measured point, so p99 rests on ≥ 10 samples.
- **Comparability:** one chart = one hardware tier and one model.

## SLOs

| Workload | TTFT p99 | TPOT p99 |
|---|---|---|
| A — summarize (~8K in → ~150 out) | ≤ 3 s | ≤ 50 ms |
| B — generate (~100 in → ~1,000 out) | ≤ 1 s | ≤ 50 ms |
| C — agent turn | ≤ 1 s | ≤ 50 ms |

## Hardware tiers
- **Official:** Llama-3.1-8B on one A100-80GB or H100. Decode floor ≈ 8–10 ms (A100), ≈ 5 ms (H100).
- **L4 first pass:** Llama-3.2-1B (decode floor ≈ 8 ms). Llama-3.1-8B on the L4 has a ≈ 54 ms decode
  floor, above the TPOT SLO, so it can never pass; don't use it for goodput.
- **Mac:** shape checks only; never compared against SLOs.

## Sanity checks behind the numbers
- A: an 8K-token 8B prefill ≈ 1.3×10¹⁴ FLOPs ≈ 0.85 s on an A100 at 50% MFU, leaving ~2 s for queueing.
- TPOT 50 ms ≈ 20 tokens/s, faster than reading speed; ~5× the A100 floor, leaving room to batch.
- C at 1 s is hard before M6 (each turn re-prefills its whole history); S6 is meant to show that.

## Rejected
- Tighter chat-grade targets (500 ms / 30 ms): lower capacity everywhere, batching wins harder to show.
- Looser batch targets (2 s / 100 ms): hide the latency problems S4 and S7 exist to expose.
