# 04 — Correctness method

**Decided:** 2026-10-11 · **Needed by:** M2 (used from M3) · **Status:** accepted

## The problem
At batch 1 our engine and HF do the same math in the same order, so greedy tokens match exactly (G0).
Batched, bf16 results shift with batch shape: kernels pick different tiling and reduction orders,
floating-point addition is not associative, and bf16 keeps ~3 significant digits. Logits move by
~0.01–0.1; at a near-tie the argmax flips and the request follows a different, equally valid path.
Exact token matching would then fail with no bug present, while real batching bugs (padding masks that
leak, off-by-one positions, one request reading another's KV) cause large errors.

## Choice: layered gates

| Gate | From | Method | Catches |
|---|---|---|---|
| G0 | M0 | Batch 1, greedy tokens identical to HF (bf16, same device) | Model math |
| G1a | M3 | Batched engine in **fp32**: tokens identical to batch-1 runs; logits within 1e-4 | Logic bugs: masks, positions, cache mix-ups. Deterministic; tiny configs run on the Mac |
| G1b | M3, M5, D1 | **bf16 teacher-forced**: every request is fed a fixed reference continuation; per-position logits compared with a batch-1 reference | Kernel numerics: FlashInfer layout (M5), TP reductions (D1) |
| G2 | M1 (gate from M9) | WikiText-2 perplexity within 1%, GSM8K within 2 standard errors of the HF baseline | Quality loss: quantization, fp8 KV |

Speculative decoding (M10) must reproduce non-speculative greedy output exactly under G0/G1a.

## G1b tolerance, calibrated rather than guessed
1. Run HF itself at batch 1 and batch 8 on the same prompts; record the per-position mean KL divergence
   and max |Δlogit| between the two. That is the noise a correct implementation shows.
2. Our engine passes if its batched-vs-batch-1 KL and max |Δlogit| are ≤ 2× HF's, and its top-1 token
   agrees at every position where the reference's top-1/top-2 margin exceeds 0.5 logits.
3. Recalibrate per hardware tier and per kernel backend (torch_ref vs FlashInfer).

## Rejected
- Exact greedy match only: blind to batching bugs.
- One bf16 tolerance only: small logic bugs can hide inside it, and nothing runs exactly on the Mac.
- Batch-invariant kernels (exact results across batch sizes): a kernel project; deferred kernel track.

## At M2
Batch size is still 1, so M2 adds **G0 through the API**: tokens streamed by the server must equal
`greedy_generate`'s for the same prompt. G1a/G1b arrive with batching at M3.
