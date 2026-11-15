# S4 — Mixed traffic: static vs continuous batching

**Milestone:** M4 · **Compared against:** git tag `m3` (static batching) · **Date:**

## Setup
- Hardware: one H100 (p5.4xlarge, decision 25) in the v1 GPU session (smoke run only on the L4)
- Model / dtype: Llama-3.1-8B, bf16
- Workload: Mix, 30% A (8,000 tokens in → 150 out, arXiv) and 70% B (~100 in → 1,000 out, UltraChat), open-loop
  Poisson arrivals at rising rates up to 2–3× M4's knee, ≥ 1,000 requests per rate, 30 s warm-up excluded.
- Both servers: `--max-batch-size 16`, same `--max-model-len`; M4 also `--max-num-batched-tokens 8192`,
  `--max-waiting 64`.
- Command(s): M3 from a worktree at tag `m3` on port 8001, M4 on port 8000, then
  `python benchmarks/s4_mixed_traffic.py --base-url ... --tokenizer models/Llama-3.1-8B --tag m3|m4` and
  `--compare m3,m4` (full commands in the script's docstring).
- Correctness: G1a for M4 (`tests/test_continuous_batching.py`, fp32 exact under joins, leaves and slot reuse) and
  G1b pass before measuring. R2 is checked by the same run: the rates past the knee.

## Prediction (written before running)
<!-- Fill in now, while M4 is fresh.
     1. Service time: an A request ≈ one 8K prefill (≈ 0.26 s on an H100 at 50% MFU, decision 3) + 150 decode steps;
        a B request ≈ 1,000 decode steps. With batch 16 and the decode step time from S3, what is each engine's
        capacity in req/s on the 30/70 mix? (M3: a batch lasts as long as its longest member. M4: a slot is busy
        only for its own request.)
     2. Goodput at M3's knee and at M4's knee: the plan predicts several× on high length variance. Where does the
        factor come from: rows wasted on finished requests (S3 part B) plus queueing behind the longest?
     3. p99 TTFT for B below the knee: M3 = wait for a whole batch to drain; M4 = wait for one slot to free
        (≈ mean remaining time of 16 running requests ÷ 16). Put numbers on both.
     4. The new problem: B's p99 ITL under M4. Every A admission stalls all running B requests for one 8K prefill
        (≈ 0.26 s ≫ the 50 ms TPOT SLO). How often does a given B request see one during its 1,000 tokens at rate λ?
        Does its TPOT p99 still meet the SLO, or only its median? (M7 chunked prefill is the fix.)
     5. R2: at 2–3× the knee, M4 rejects with 503 once 16 + 64 requests are in flight. Is 64 waiting small enough
        that admitted requests still meet the TTFT SLO (queue ≈ 64 ÷ throughput)? What does M3 do instead? -->

## Result
<table and chart>

## Gap and explanation
<why the measured number differs from the prediction — this is the learning>

## What it motivates next
