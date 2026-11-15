# S3 — Batch-size sweep and the cost of static batching

**Milestone:** M3 · **Compared against:** git tag `m2` (one request at a time) · **Date:**

## Setup
- Hardware: one A100/H100 in the v1 GPU session (L4 + Llama-3.2-1B for a first pass)
- Model / dtype: Llama-3.1-8B, bf16
- Workload:
  - Part A: B = 1 → 64 requests of workload-B shape (100-token prompt) decoding together, equal lengths.
  - Part B: 128 requests, output lengths uniform 16–512, all queued at once, static batches of 16.
- Command(s): `python benchmarks/s3_batch_sweep.py --model-dir models/Llama-3.1-8B --gpu a100-80`
- Correctness: G1a (fp32 exact) and G1b (bf16 within 2× HF's own batch noise) pass before measuring.

## Prediction (written before running)
<!-- Fill in now, while M3 is fresh.
     1. Decode step time at B = 1 (the decode floor from S1) and at B = 32: how much slower, and why?
        Use bytes read per step = weights + B × context × KV bytes/token.
     2. Arithmetic intensity at batch B ≈ B FLOP/B (2 FLOPs per weight per token, 2 bytes per bf16 weight).
        At what B does it reach the GPU's balance point (plan's table)? That is where the knee should be.
     3. Throughput at B = 64 vs B = 1: the ideal is 64×; what do you expect, and what limits it besides compute
        (KV reads growing with B, attention, Python per-step overhead)?
     4. Part B: with lengths uniform 16–512 in batches of 16, what fraction of decode rows is wasted? (Hint:
        expected max of 16 uniform draws vs their mean.) How long does the last batch's requests wait?
     5. Which of these costs does continuous batching (M4) remove, and which remain? -->

## Result
<table and chart>

## Gap and explanation
<why the measured number differs from the prediction — this is the learning>

## What it motivates next
