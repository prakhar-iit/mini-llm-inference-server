# S0 — Cost of no cache

**Milestone:** M0 · **Compared against:** none (first milestone) · **Date:**

## Setup
- Hardware: g6.xlarge (1× L4 24 GB) for the reported numbers; MacBook (MPS) only to check the shape
- Model / dtype: Llama-3.2-1B, bf16 (Llama-3.1-8B too if time allows)
- Workload and load: batch 1, random token ids; step time at L = 128 … 4,096; 256 tokens generated from a
  16-token prompt
- Command(s): `python benchmarks/s0_no_cache.py --model-dir models/Llama-3.2-1B`

## Prediction (written before running)
<!-- Fill in before running. Use the plan's formulas:
     FLOPs per token ≈ 2 × params; prompt attention ≈ 2·n²·d·L; decode floor = weight bytes ÷ bandwidth.
     1. Step time at L=128 on the L4: compute-bound or bandwidth-bound? Give a number in ms.
     2. Step time at L=4,096: how many × the L=128 time, and why?
     3. At what L does the n² attention term become a noticeable share of the step's FLOPs?
     4. Shape of cumulative time for 256 generated tokens: linear, quadratic, or in between? -->

## Result
<table and chart>

## Gap and explanation
<why the measured number differs from the prediction — this is the learning>

## What it motivates next
