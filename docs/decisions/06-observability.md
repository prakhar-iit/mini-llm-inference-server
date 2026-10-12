# 06 — Observability

**Decided:** 2026-10-11 · **Needed by:** M2 · **Status:** accepted

## Choice
Three records from M2 on; distributed tracing (OpenTelemetry) added at D2.

### 1. Per-request timeline
Timestamps (`time.monotonic_ns`, shared by both processes on one machine) at each stage:

| Event | Process |
|---|---|
| `received` (HTTP request parsed) | frontend |
| `tokenized` | frontend |
| `queued` (arrived at the engine core) | engine |
| `scheduled` (first time in a batch) | engine |
| `first_token` (end of prefill) | engine |
| `finished` | engine |
| `sent` (last chunk written to the client) | frontend |

Plus counts: prompt tokens, cached tokens (M6), output tokens, preemptions (M5), finish reason.
Returned in the final response as an extra field (`minillm_timeline`) and appended to a JSONL file; the
client joins it by request id (decision 5).

### 2. Engine step log
One row per engine step: `step`, `t_start`, `duration`, `num_seqs`, `prefill_tokens`, `decode_tokens`,
`kv_blocks_used`, `kv_blocks_free`, `queue_len`. Behind a flag (`--step-log`), buffered in memory and
written off the hot path. S3 (batch size), S5 (KV pressure) and S7 (prefill stalls) read it directly.

### 3. Prometheus `/metrics` (frontend)
- Counters: requests (by finish reason), rejected requests, prompt tokens, generated tokens.
- Histograms: TTFT, TPOT, E2E, queue time.
- Gauges: running requests, waiting requests, KV-cache usage (sent from the engine with step outputs).
- Names follow vLLM's (`minillm:time_to_first_token_seconds`, …) so S11 dashboards line up.

## Why
Client numbers say *that* a request was slow; these records say *why* (queueing vs prefill vs overhead),
which is the "explain the gap" step of every study. Prometheus is also what D2 routing and D5
autoscaling read.

## Rules
- Recording on the engine's hot path is an append to a list; files are written outside the step loop.
- S2 measures the overhead with recording on and off; it must stay under ~1%.
- Monotonic timestamps are only comparable on one machine; cross-machine timing (Phase 3) is one reason
  tracing arrives later.

## Rejected
- Logs only: studies couldn't explain their gaps.
- Timeline + Prometheus without the step log (the plan's original): S3/S5/S7 would infer batch and KV
  behaviour indirectly.
- OpenTelemetry from M2: setup cost with two processes on one machine; added at D2 with the router.
