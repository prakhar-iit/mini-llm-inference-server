# Design decisions

One entry per decision from the plan's "Design review — open decisions" list. Add a row when a
decision is made (the plan's "Implementation order" shows where each one is needed); write a short note `NN-<topic>.md` if the reasoning needs more than a line.

| # | Decision | Needed by | Choice | Date |
|---|---|---|---|---|
| 1 | Engine process architecture | M2 | Frontend + engine core over ZMQ + msgpack; executor interface for per-GPU workers at D1 ([note](01-process-architecture.md)) | 2026-10-11 |
| 2 | Workload C (agent) spec | M2 | Synthetic generator with sweepable distributions, token-id prompts, seeded JSONL; burstiness fitted to Azure/BurstGPT; Mooncake replay at P1 ([note](02-workload-c.md)) | 2026-10-11 |
| 3 | SLO and metric definitions | M2 | TTFT p99 ≤ 3 s (A) / 1 s (B, C), TPOT p99 ≤ 50 ms; official tier 8B on A100/H100, L4 tier 1B; client-side open-loop timing ([note](03-slos-and-metrics.md)) | 2026-10-11 |
| 4 | Correctness method | M2 | Layered: G0 batch-1 exact; G1a fp32 batched exact; G1b bf16 teacher-forced logits within 2× HF's own batch noise; G2 quality ([note](04-correctness.md)) | 2026-10-11 |
| 5 | Benchmark client | M2 | Own asyncio client for A, B and C; cross-checked against `vllm bench serve` in the GPU session (within ~5%) ([note](05-benchmark-client.md)) | 2026-10-11 |
| 6 | Observability | M2 | Per-request timeline (returned to client + JSONL), engine step log behind a flag, Prometheus `/metrics`; OpenTelemetry at D2 ([note](06-observability.md)) | 2026-10-11 |
| 7 | Attention backend and KV layout | M5 | Three stages on one layout: M5a torch_ref (Mac, oracle), M5b FlashInfer, M5c vLLM's flash-attn; block size 16, per-layer K/V `[blocks, 16, Hkv, Dh]`, ragged per-step metadata; all three coded before M6, GPU-tested in one session ([note](07-attention-backend-kv-layout.md)) | 2026-10-18 |
| 8 | Block-hash spec | M6 | SHA-256 chain over full 16-token blocks (uint32 LE token ids), root salted with model@revision, KV dtype, layout, block size and an optional per-request `cache_salt`; last token always recomputed ([note](08-block-hash-spec.md)) | 2026-10-18 |
| 9 | Scheduler policy | M4 | Iteration-level, FCFS without skipping, slots reserving max_model_len, prefill-first with a token budget, 503 past max_waiting; preempt newest first from M5 ([note](09-scheduler-policy.md)) | 2026-10-18 |
| 10 | Systems language (Go / Rust) | D2 | | |
| 11 | Router state and tokenization | D2 | | |
| 12 | Session / agent API contract | D4 | | |
| 13 | Pipeline parallelism gap | N2 | | |
| 14 | KV transfer mechanism | D3 | | |
| 15 | Failure model | D5 | | |
| 16 | Multi-GPU dev hardware | D1 | | |
| 17 | Cluster cache data vs control plane | P1 | | |
| 18 | KV compatibility across engines | P1 | | |
| 19 | Eviction and TTL across tiers | D4 | | |
| 20 | AWS access and quotas | Phase 0 (G-quota), N1 (P-instances) | | |
| 21 | Budget per phase | Phase 0 | | |
| 24 | Phase 5 choices (checkpoint, hardware, EFA all-to-all, weights, budget) | K3 | | |
