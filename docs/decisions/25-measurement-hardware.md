# 25 — Measurement hardware per experiment

**Decided:** 2026-11-14 · **Needed by:** the first official GPU session (v1) · **Status:** accepted
· **Also settles:** decision 16 (multi-GPU dev hardware)

## Problem
Changing the GPU changes the results:
- Decode is bound by memory bandwidth. The 8B's decode floor is ~54 ms on an L4, ~7.9 ms on an A100 80 GB
  and ~4.8 ms on an H100.
- Prefill is bound by compute. The H100 has ~3× the A100's bf16 FLOP/s and has FP8 tensor cores; the A100
  has none.

The studies form chains: S0 → S10 each beat the milestone before, and Phase 2 starts from the TP=1 engine.
If two links of a chain run on different GPUs, the hardware's gain can't be told apart from the code's.

The plan used to say "A100/H100" for official runs. That made the GPU choice open, and AWS sells A100s and
H100s mainly in 8-GPU instances that bill for every GPU. Since 2025 AWS also sells **p5.4xlarge, a single
H100**. Live us-east-1 prices on 2026-11-14:

| Instance | GPUs | On-demand $/h | Lowest spot $/h |
|---|---|---|---|
| g6.xlarge | 1× L4 24 GB | 0.80 | 0.38 |
| p5.4xlarge | 1× H100 80 GB | 6.88 | 2.87 |
| p4d.24xlarge | 8× A100 40 GB | 21.96 | 15.21 |
| p4de.24xlarge | 8× A100 80 GB | 27.45 | 20.68 |
| p5.48xlarge | 8× H100 80 GB | 55.04 | 22.03 |

## Choice
1. **One measurement GPU per chain of studies.** Every study in a chain runs on the same GPU. Official
   numbers on different GPUs are never compared directly.
2. **Choose by need:** memory (model + KV), interconnect (NVLink inside a node, EFA across nodes) and
   features (bf16 and flash-attn need Ampere or newer; FP8 tensor cores need Ada or Hopper).
3. **When the GPU must change, measure a baseline on the new one**, and report MBU (decode) and MFU
   (prefill) next to the milliseconds, since those carry across GPUs.

| Experiments | GPU | Instance | Why |
|---|---|---|---|
| Development, `pytest`, smoke runs, first pass of every study | L4 | g6.xlarge | Cheapest instance that runs everything (bf16, FlashInfer, flash-attn). Numbers are labelled first-pass. The T4 (g4dn) is cheaper but has no bf16 and no flash-attn |
| Multi-GPU correctness: D1–D3, Moonlight K1–K5 code | 4× L4, PCIe | g6.12xlarge | Correctness only; no NVLink, so no performance numbers (this settles decision 16) |
| **Phase 1 official: S0–S11** (8B, plus vLLM for S11) | **1× H100 80 GB** | **p5.4xlarge** | The cheapest official GPU; the whole chain on one card; has FP8, so S9a's prefill prediction can be tested |
| **Phase 2: D1–D5** | **8× H100, NVLink** | **p5.48xlarge** | Same chip as Phase 1, so Phase 1's numbers are the TP=1 baseline; local NVMe for D4 |
| Phase 3: N1 | Any 2 nodes | 2× g6.xlarge over TCP | Launch, NCCL, failure modes only; nothing measured |
| Phase 3: N2–N4, P1–P3 | 2× 8× H100 | 2× p5.48xlarge, EFA | Same chip; EFA at 3,200 Gbps. p4d's 400 Gbps would dominate the KV-transfer results |
| Phase 3: P4 | none | API calls | |
| Phase 4: EKS stack, self-host vs API cost | 8× H100 | p5.48xlarge node groups | The cost comparison uses the hardware it would serve on |
| Phase 5: K0, K6, K7 (Kimi K2) | 8× H200 | p5en.48xlarge | The model needs ~600 GB+. These numbers stand on their own: own engine vs the SGLang/vLLM reference on the same p5en. 2× p5 stays an option only for cross-node EP (decision 24) |

The H100 is the single official chip from Phase 1 through Phase 4. Only Phase 5 changes generation, and it
brings its own reference run.

## Consequences
- **Quota:** p5.4xlarge needs 16 vCPUs of "Running On-Demand P instances". The account has 8 in us-east-1 and 0
  in us-east-2 and us-west-2 (checked 2026-11-14), so a quota increase is needed before the v1 session. Spot
  quotas are 0 everywhere (decision 20).
- **Predictions:** worked numbers that assumed an A100 now use the H100. An 8K prefill of the 8B is
  1.3×10¹⁴ FLOPs ≈ 0.26 s at 50% MFU (it was 0.85 s on the A100); the decode floor is ~4.8 ms (was 7.9 ms).
- **Commands:** official study commands pass `--gpu h100`.
- **S8 (CUDA graphs)** depends on the host CPU too. p5.4xlarge has 16 vCPUs; record the CPU model in the study
  note.

## Rejected
- **A100 for Phase 1 (p4de):** $27.45/h for 8 GPUs when a study uses one, and no FP8 for S9a.
- **L4 for official numbers:** the 8B's ~54 ms decode floor is above the 50 ms TPOT SLO (decision 3).
- **Mixing GPUs inside a chain** to use whatever is cheapest that day: makes milestone-to-milestone gains
  unreadable.

## Budget alternative (decision 21)
Phase 2 could run on a p4de.24xlarge (8× A100 80 GB, $27.45/h), about half the p5.48xlarge price. The cost is
continuity: Phase 1's H100 numbers stop being the TP=1 baseline, so TP=1 is remeasured on the A100 node
(~1 hour; each D study compares with its own TP=1 run anyway). Not recommended for Phase 3, where p4d's
slower EFA distorts the P/D and KV-transfer results.
