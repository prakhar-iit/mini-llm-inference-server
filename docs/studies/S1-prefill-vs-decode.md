# S1 — Prefill vs decode

**Milestone:** M1 · **Compared against:** git tag `m0` (S0, no cache) · **Date:**

## Setup
- Hardware: one A100/H100 (or L4 for a first pass) in the v1 GPU session; MacBook only for the shape check
- Model / dtype: Llama-3.1-8B, bf16 (Llama-3.2-1B for comparison)
- Workload and load: batch 1, random token ids. Sweep A: prompt 128 → 16,384 tokens, 16 output tokens.
  Sweep B: 128-token prompt, 2,000 output tokens. Workloads A and B arrive with the M2 harness.
- Command(s): `python benchmarks/s1_prefill_decode.py --model-dir models/Llama-3.1-8B --gpu a100-80`

## Prediction (written before running)
<!-- Fill in now, while M1 is fresh. Use the plan's formulas and GPU table.
     1. Decode floor for the 8B on your GPU (weight bytes ÷ bandwidth). Will measured ITL land near it?
        What MBU do you expect, and why not 100%?
     2. TTFT at 128 vs 4,096 tokens: linear in prompt length? Past roughly what length does the n²
        attention term make it grow faster than linear?
     3. Prefill MFU at 4K tokens vs decode MFU. Why are they orders of magnitude apart?
     4. Does ITL rise as the context grows from 128 to 2,000 tokens? By how much (KV bytes read per step)?
     5. Versus S0 (tag m0): the per-token time at context 2,048 without and with the cache. -->

Mac shape check (Llama-3.2-1B, MPS, not for the write-up): TTFT 223 ms at 128 tokens → 3,475 ms at 2,048
(linear); ITL flat at ~35 ms from context 129 to 328; decode reads ~70 GB/s.

## Result
<table and chart>

## Gap and explanation
<why the measured number differs from the prediction — this is the learning>

## What it motivates next
