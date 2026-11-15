# S2 — Overhead and queueing

**Milestone:** M2 · **Compared against:** git tag `m1` (S1, in-process engine) · **Date:**

## Setup
- Hardware: one A100/H100 in the v1 GPU session (L4 + Llama-3.2-1B for a first pass)
- Model / dtype: Llama-3.1-8B, bf16
- Workload and load:
  - Part 1 (overhead): 100 sequential requests, 32-token prompt, 1 output token, idle server.
  - Part 2 (queueing): 128 in / 32 out; service time S measured first; Poisson arrivals at
    rho = 0.2 … 1.2; ≥ 1,000 requests per point after a 30 s warm-up (decision 3).
- Command(s):
  `python -m minillm.server --model-dir models/Llama-3.1-8B --timeline-log results/s2_timeline.jsonl`
  `python benchmarks/s2_overhead_queueing.py --base-url http://127.0.0.1:8000 --vocab-size 128256 --per-rate 1200`

## Prediction (written before running)
<!-- Fill in now, while M2 is fresh.
     1. Non-GPU overhead per request (HTTP + tokenize + ZMQ both ways + SSE): how many ms, and what share of
        TTFT for a 32-token prompt on the 8B model? Which stage do you expect to be largest?
     2. Service time S for 128 in / 32 out on the 8B: prefill + 31 × decode floor (plan's GPU table).
     3. M/D/1 mean wait at rho = 0.5, 0.8, 0.9 in units of S. At what rho does p99 TTFT cross the 1 s SLO?
     4. Above rho = 1, how does wait grow with the number of requests sent, and why does throughput cap at 1/S?
     5. What fraction of the GPU sits idle at the capacity limit, given decode is bandwidth-bound (S1)?
        That idle fraction is what M3/M4 batching recovers. -->

## Result
<table and chart>

## Gap and explanation
<why the measured number differs from the prediction — this is the learning>

## What it motivates next
