# S5 — KV pressure: contiguous slots vs a paged pool

**Milestone:** M5a (kernel part: M5b, M5c) · **Compared against:** git tag `m4` (contiguous slots) · **Date:**

## Setup
- Hardware: one A100/H100 in the v1 GPU session (L4 + Llama-3.2-1B for a first pass); MacBook for the shape check
- Model / dtype: Llama-3.1-8B, bf16
- Workload: 256 workload-B-shaped requests (100-token prompts, outputs uniform 16–1,000), all queued at once,
  in-process (no server), `max_model_len` 8,192.
  - Part A: an 8 GiB KV budget. Contiguous = M4's rule (one slot of max_model_len per request), reproduced on the
    same engine by capping the batch at that slot count; paged = up to 256 running.
  - Part B: the same requests with the pool at 1, ½, ¼, ⅛, 1⁄16 of part A's.
  - Part C (GPU session, after M5b/M5c): decode step time at batch 1/8/32 × context 512/4,096 for M4's contiguous
    slots, the torch reference, FlashInfer and flash-attn on the same pages.
- Command(s): `python benchmarks/s5_kv_pressure.py --model-dir models/Llama-3.1-8B --kv-gb 8 --gpu a100-80`
  (add `--backends torch_ref,flashinfer,flash_attn` once both kernels are installed)
- Correctness: G0/G1a on paged KV (`tests/test_paged.py`, fp32 exact, preempted requests included), R3 after
  every run (the script checks it too), and in the GPU session `tests/test_kernels.py` (each kernel vs the
  reference within G1b) before part C is measured.

## Prediction (written before running)
<!-- Fill in now, while M5 is fresh.
     1. Part A: KV bytes per token for the 8B (plan's table: 128 KB). How many contiguous slots fit in 8 GiB at
        max_model_len 8,192? An average request here is ~100 + 508 tokens: what fraction of each slot holds real
        tokens? How many such requests fit paged (ceil(tokens / 16) blocks each)?
     2. Block fill: the last block of a request is on average half empty (7.5 tokens). What share of held block
        slots hold real tokens for ~600-token requests? (The plan predicts waste under ~5%.)
     3. Throughput: with several× more requests running, how much faster is decode, given S3's batch curve (memory-
        bound until the knee)? Where does it stop helping?
     4. Part B: at which pool size do preemptions start, from the peak blocks the full run held? What does each
        preemption cost (its prompt + output so far, recomputed)? Where do TTFT and ITL fall off a cliff, and why is
        the cliff sharp rather than gradual?
     5. Part C: the reference gathers the whole context every decode step; at 4,096 tokens × batch 32 how many bytes
        is that copy vs the weights read? How close should FlashInfer and flash-attn get to contiguous slots? -->

Mac shape check (Llama-3.2-1B, MPS, 0.25 GiB of KV = 512 blocks, `max_model_len` 2,048, 32 requests with outputs
16–256; not for the write-up):
- Part A: contiguous fits 4 slots of 2,048 tokens, and a request averages ~236 tokens, so 11.5% of each
  reservation holds real tokens. Paged ran 15.0 requests on average (max 32, every request at once) against 3.6,
  for 98.2 vs 36.9 tok/s (2.66×), with 96.5% of held block slots holding a real token.
- Part B: one preemption at ½ the pool, 17 at ¼ (2,079 tokens recomputed, 69.7 tok/s, TTFT p99 36 s), 25 at ⅛
  (3,121 recomputed, 7.3 tok/s, TTFT p99 498 s, ITL max 44 s): the cliff sits between ¼ and ⅛.
- Part C (torch reference vs contiguous, decode step): 48.6 vs 38.1 ms at B = 1, ctx 256; 112.6 vs 56.7 ms at
  B = 8, ctx 1,024. The gather's cost grows with batch × context, as expected of a reference path.

## Result
<table and chart>

## Gap and explanation
<why the measured number differs from the prediction — this is the learning>

## What it motivates next
