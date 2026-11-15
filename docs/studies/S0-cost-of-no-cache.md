# S0 — Cost of no cache

**Milestone:** M0 · **Compared against:** none (first milestone) · **Date:**

## Setup
- Hardware: one H100 (p5.4xlarge, decision 25) in the v1 GPU session; smoke run on the g6.xlarge
- Model / dtype: Llama-3.1-8B, bf16
- Workload and load: batch 1, random token ids; step time at L = 128 … 4,096; 256 tokens generated from a
  16-token prompt
- Command(s): `python benchmarks/s0_no_cache.py --model-dir models/Llama-3.1-8B`

## Prediction (written before running)
<!-- Fill in before running. Use the plan's formulas:
     FLOPs per token ≈ 2 × params; prompt attention ≈ 2·n²·d·L; decode floor = weight bytes ÷ bandwidth.
     1. Step time at L=128 on the H100: compute-bound or bandwidth-bound? Give a number in ms.
     2. Step time at L=4,096: how many × the L=128 time, and why?
     3. At what L does the n² attention term become a noticeable share of the step's FLOPs?
     4. Shape of cumulative time for 256 generated tokens: linear, quadratic, or in between? -->

## Result
<table and chart>

## Gap and explanation
<why the measured number differs from the prediction — this is the learning>

## What it motivates next
