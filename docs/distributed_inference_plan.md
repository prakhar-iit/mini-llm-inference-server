# Distributed Inference Engineer — Track & Roadmap

*Decided by Prakhar 2026-10-03; revised 2026-10-04. Supersedes the standalone "vLLM deployment +
benchmark" project in `action_items.md` Track 2. Companion docs: `job_search/inference_engineering_notes.md`
(theory) and `inference_optimization_theory.md` (taxonomy).*

*2026-10-04 change: **Experiment 1 on stock vLLM and the ablation study are discarded**, and so is the
Word doc "LLM_Inference_Knowledge_Project_Plan.docx" that described them. Their studies (prefill vs
decode, MFU/MBU, batching, chunked prefill, prefix caching, KV pressure, quantization, speculative
decoding, CUDA graphs) now run **on the mini inference server, one study per milestone**, so each
latency and throughput gain is measured against Prakhar's own previous version. A design review with
the open decisions is at the end of the roadmap.*

---

## The decision

- **Chosen track: distributed inference engineer.** Owns the serving system *above* the kernels:
  engine configuration and scheduling, multi-GPU and multi-node parallelism, routing, KV-cache management
  across tiers and nodes, prefill/decode disaggregation, autoscaling, reliability, capacity and cost.
  This is the "top three rows" lane from the inference notes' software-stack table, and it makes the
  positioning line literally true: *a backend and distributed-systems engineer who understands agents
  and can make inference fast and cheap.*
- **Deferred: kernel track** (writing attention kernels). Kept for later, not dropped — see the end of
  this doc. Phase 1 uses borrowed kernels (FlashInfer / FlashAttention paged-KV support).

## Why this order works

Toggling optimizations on stock vLLM was rejected as "validation, not learning": vLLM already ships with
paged KV, continuous batching, CUDA graphs, chunked prefill and prefix caching on, so the deltas are small
and none of the work is his. Instead: **build a mini inference server, add one optimization at a time,
and run a study at each step** — predict the effect, measure it against the previous milestone, explain
the gap. Then take the same engine distributed.

---

## How every study works

**Protocol (same for every milestone):**
1. Write the prediction down first (a number with a reason, from the formulas below).
2. Run the study against the previous milestone's git tag, same workload, same hardware.
3. Explain the gap between predicted and measured. Most of the learning is in this step.
4. Record it in `studies/NN-<name>.md`: setup, prediction, chart, result, one-paragraph explanation.

**Model ladders (each ladder is one architecture, so the same code runs at every size):**
- **Dense, Phases 1–4 (Llama):**
  - Develop and debug on **Llama-3.2-1B**.
  - Run studies on **Llama-3.1-8B**.
  - Use **Llama-3.3-70B** for Phase 3.
- **MoE + MLA, Phase 5 (DeepSeek-V3 architecture, which Kimi K2 uses):**
  - Tiny randomly initialized DeepseekV3 config for unit tests.
  - **Moonlight-16B-A3B**, Moonshot's own small model on the same architecture.
  - **Kimi K2.x**: 1T total parameters, 32B active.

**Development environment — three tiers, same code everywhere:**
1. **MacBook (PyTorch on MPS), tiny configs and Llama-3.2-1B — correctness only.**
   - What runs: scheduler, block manager, prefix-cache hashing, server, chunked-prefill logic,
     speculative-decoding loop, router and replicas as separate processes.
   - TP sharding can be checked with 2 processes on PyTorch's CPU backend (gloo).
   - What doesn't run: FlashInfer, FlashAttention, CUDA graphs, NCCL, Triton and FP8, all CUDA-only.
   - Mac numbers never go in a study write-up. bf16 on MPS needs macOS 14+.
2. **g6.xlarge (1× L4) and g6.12xlarge (4× L4) — the CUDA path.**
   - Fix CUDA-path bugs: FlashInfer integration, KV layout, CUDA graphs, FP8.
   - Dry-run every study; check multi-GPU correctness for D1–D3.
   - Moonlight-16B (~32 GB in bf16) fits on the g6.12xlarge.
3. **p4/p5 (and p5en for Kimi K2) — scripted runs only.** Official study numbers, multi-GPU and
   multi-node performance, Phase 5.

*Code rules that make this work:*
- A `--device mps|cuda|cpu` flag.
- An attention-backend switch: a plain-PyTorch reference that gathers KV through the block table, and
  FlashInfer. On CUDA the reference doubles as FlashInfer's correctness oracle.
- One test suite on every tier.
- Compare against Hugging Face on the same device only, never Mac vs CUDA.
- Platform-specific dependency pins, so FlashInfer installs only on Linux/CUDA.

**Study hardware:** run studies on the L4 as each milestone lands; it's cheap and the effects are
visible. Then rerun the whole suite on one A100/H100 of the p4/p5 in a single scripted session: once at
v1 ship, once at the end of Phase 1.

**Three workloads (the harness is reused from M2 through Phase 4):**

| Workload | Shape | Data | What it stresses |
|---|---|---|---|
| **A — Summarize** | ~8,000 tokens in → ~150 out | arXiv papers (`ccdv/arxiv-summarization`) | Prefill (compute-bound) |
| **B — Generate** | ~100 tokens in → ~1,000 out | ShareGPT conversations | Decode (memory-bound) |
| **C — Agent** | Shared system prompt + tool schemas (~3–4K tokens), multi-turn sessions, tool-call pauses between turns | Synthetic generator with set distributions (see open decision 2) | Prefix reuse, session state, bursty arrivals |

**Metrics:**
- TTFT p50/p99.
- Inter-token latency (ITL/TPOT) p50/p99.
- Output tokens/s.
- **Goodput:** requests per second that meet the SLO.
- Max concurrent requests at SLO.
- KV memory per request.
- **MFU for prefill, MBU for decode.**

**Formulas used for every prediction:**
- **FLOPs per token** ≈ 2 × params. Prompt attention ≈ 2·n²·d·L. Attention per decode token ≈ 4·ctx·d·L.
- **KV bytes per token** = 2 × layers × kv_heads × head_dim × bytes:
  - Llama-3.2-1B: 32 KB.
  - Llama-3.1-8B: 128 KB.
  - 70B: 320 KB in bf16, 160 KB in fp8.
- **MFU** = achieved FLOP/s ÷ peak. **MBU** = (weight bytes + KV bytes read) / step time ÷ peak bandwidth.
- **Decode floor**, one user = weight bytes ÷ bandwidth.

| GPU | bf16 dense | Bandwidth | Balance point | 8B decode floor | 1B decode floor |
|---|---|---|---|---|---|
| L4 (g6) | ~121 TFLOPS | ~300 GB/s | ~400 FLOP/B | ~54 ms | ~8 ms |
| A100 40GB (p4d) | 312 TFLOPS | 1.56 TB/s | ~200 FLOP/B | ~10.3 ms | ~1.6 ms |
| A100 80GB (p4de) | 312 TFLOPS | 2.04 TB/s | ~153 FLOP/B | ~7.9 ms | ~1.2 ms |
| H100 (p5) | ~989 TFLOPS | 3.35 TB/s | ~295 FLOP/B | ~4.8 ms | ~0.7 ms |

---

## Roadmap (nights and weekends, roughly 8 months including the Phase 5 capstone)

### Phase 1 — Mini inference server, single GPU, a study at every step · ~10 weekends
Python + PyTorch, borrowed attention kernels. Correctness gate at every milestone (see open decision 4).

**Phase 1a — core engine (M0–M6) → ship v1 · ~6 weekends**

| # | Build | Study on own engine | Prediction to check | What it teaches |
|---|---|---|---|---|
| M0 | Own generation loop, no KV cache | **S0 Cost of no cache:** time per token vs sequence length | Per-token time grows linearly with length; total cost is quadratic | Why the KV cache exists |
| M1 | KV cache | **S1 Prefill vs decode** (the former Experiment 1): sweep prompt 128 → 16K at 16 output tokens; then 1 → 2,000 output tokens; workloads A and B | TTFT linear in prompt length, then superlinear past ~8K (attention n²). ITL flat near the decode floor. Prefill MFU 40–60%; decode MFU < 1%, MBU 70–85%. A is ~30–40% prefill; B is ~99% decode | The two phases are different problems: compute-bound vs memory-bound. Every later optimization targets one of them |
| M2 | OpenAI-compatible server, streaming, request queue | **S2 Overhead and queueing:** client vs engine-internal TTFT; Poisson arrivals at rising rate, one request at a time | Fixed overhead of a few ms. Latency hockey-sticks once the arrival rate passes 1 ÷ service time | Queueing theory; why serving one at a time wastes the GPU |
| M3 | Static batching | **S3 Batch-size sweep** 1 → 64 on B, then mixed lengths | Throughput near-linear in batch while memory-bound (ITL nearly flat), knee where arithmetic intensity reaches the balance point. On mixed lengths, short requests wait for the longest | Batching uses idle compute; the head-of-line problem |
| M4 | Continuous batching scheduler (iteration-level) | **S4 Mixed traffic:** M3 vs M4 on A+B mix under Poisson load | Several× goodput on high length variance; p99 TTFT drops. New problem: ITL spikes when a long prompt joins the batch | Orca's insight; sets up chunked prefill |
| M5 | Paged KV cache + block manager + eviction by recompute | **S5 KV pressure:** max concurrency at fixed memory, contiguous (reserve max_len) vs paged; shrink memory until preemptions start | Contiguous wastes most reserved KV memory; paged waste under ~5%, so concurrency rises several×. Preemption shows up as TTFT/ITL cliffs | Memory, not compute, caps decode throughput |
| M6 | Prefix caching (chained block hashes, refcount, LRU) | **S6 Shared prefixes:** workload C, and many questions on one paper (A); vary shared-prefix fraction | TTFT drops in proportion to the cached fraction; ~zero gain on unique prompts (small hashing overhead) | Why agents and RAG are cache-friendly; hit rate as a first-class metric |

**Ship v1 here** (repo, README with the S0–S6 charts, one write-up). Applications for inference-platform
roles open at this point.

**Phase 1b — optimizations as studies (M7–M10) → v1.1 · ~4 weekends**

| # | Build | Study on own engine | Prediction to check | What it teaches |
|---|---|---|---|---|
| M7 | Chunked prefill (token budget per step) | **S7 Prefill/decode interference:** decode users' ITL while 8K prompts arrive; chunk size 256 / 512 / 2,048 | p99 ITL falls from ~whole-prefill time to ~one chunk's time; TTFT rises slightly; chunk size trades one for the other | The colocated trade-off disaggregation removes. **Required before D3**, or D3's baseline is a strawman |
| M8 | CUDA graphs for decode | **S8 CPU launch overhead:** decode step time at batch 1–32, 1B vs 8B; profile with the PyTorch profiler / Nsight Systems | Large gain for 1B at small batch on A100/H100 (step time near the launch-overhead floor); negligible for 8B at large batch | When the GPU waits on Python, and how to see it in a profile |
| M9 | Quantization: (a) weight-only 8-bit/FP8 via a borrowed kernel; (b) fp8 KV cache | **S9a** ITL at batch 1 and quality on a small eval; **S9b** max concurrency and long-context ITL | (a) ITL improves by roughly the byte ratio at small batch; quality loss small. Prefill gains only where FP8 tensor cores exist (L4, H100; not A100). (b) ~2× concurrent tokens at the same memory | Bytes moved are the decode cost; precision as a capacity lever |
| M10 | Speculative decoding: n-gram (prompt lookup), then a 1B draft for the 8B | **S10 Acceptance vs speedup:** by workload (A copies a lot; B is creative) and by batch size | 1.5–2.5× at batch 1 on A; little on B. Gain shrinks and can go negative as batch grows, because spare compute disappears | Speculation spends idle compute; it is a low-load optimization |

**Capstone S11 — own engine vs vLLM** on all three workloads at the same settings. Explain the remaining
gap (kernels, scheduler details, CPU overhead). An honest gap analysis is worth more than a close number.

### Phase 2 — Distributed, one node · ~6–8 weekends, on the 8-GPU instance

| # | Build | Study | What it demonstrates |
|---|---|---|---|
| D1 | **Tensor parallelism** across 2–4 GPUs (column/row split, NCCL all-reduce) | Latency vs per-GPU throughput at TP=1/2/4; share of step time in all-reduce | Two all-reduces per layer; TP buys latency, not efficiency |
| D2 | **Multi-replica + router**: round-robin vs least-loaded vs prefix/session-aware | Workload C at rising load: hit rate, p99 TTFT, goodput per policy | Cache-aware routing; the agent-serving differentiator |
| D3 | **Prefill/decode disaggregation**: prefill on one GPU, decode on another, KV transfer between them | SLO attainment vs colocated **with chunked prefill** (M7) | When separating the phases pays (DistServe-style), and when transfer cost eats it |
| D4 | **KV-cache tiering, single node**: offload idle KV (e.g. during agent tool calls) HBM → CPU DRAM → local NVMe, restore on resume | Fetch-vs-recompute crossover per tier and prompt length | Ties directly to patent US12547545B2 (HA via in-memory cache). Phase 3 takes it cluster-wide |
| D5 | **Reliability & elasticity**: replica failure mid-generation, retry/resume, autoscaling, cold-start mitigation | Recovery time, requests lost, cold-start time by weight source | Distributed-systems fundamentals applied to serving |

### Phase 3 — Multi-node cluster and cluster-scale prompt caching · ~4–6 weekends
Phase 2 stays inside one 8-GPU box. This phase crosses the node boundary, where bandwidth drops by an
order of magnitude and the design rules change.

**Interconnect facts to design around:**
- **Inside a node:** NVLink ~900 GB/s per GPU.
- **Between nodes (AWS EFA):** p4d ~400 Gbps (~50 GB/s per node); p5 ~3,200 Gbps (~400 GB/s per node).

**Rule of thumb:** tensor parallelism inside a node; pipeline parallelism, replicas or prefill/decode
disaggregation across nodes. Each experiment below either confirms the rule or finds the exception.

**Multi-node GPU cluster**

| # | Build / experiment | What it demonstrates |
|---|---|---|
| N1 | **Mechanics on cheap hardware**: 2 inexpensive GPU instances over plain TCP. Launch a 2-node job with Ray or Kubernetes, run NCCL all-reduce tests, then force cross-node TP and watch it crawl | Multi-node launch, NCCL tuning, failure modes — learned for a few dollars, not on p5 time |
| N2 | **Parallelism layout on 2× p4/p5 over EFA**: 70B model, TP=16 vs TP=8 + PP=2 | Why TP stops at the node boundary; pipeline bubbles vs all-reduce cost |
| N3 | **Cross-node P/D disaggregation** (extends D3): prefill on node A, decode on node B, KV shipped over EFA | Prediction: 70B with fp8 KV ≈ 160 KB/token, so an 8K prompt ≈ 1.3 GB → ~26 ms on p4d, ~3 ms on p5, vs hundreds of ms to recompute (bf16 KV doubles the transfer). Moving bytes beats recomputing for long prompts; for short ones, fixed transfer overhead can flip it. Find the crossover |
| N4 | **Replicas on both nodes behind the D2 router**, with cross-node cache-aware routing | Routing to where the prefix already lives vs load balance; the cost of a cross-node cache miss |

**Prompt caching at cluster scale** works in four layers; Phase 1 and Phase 2 build the first two:
1. Engine prefix cache (M6).
2. Multi-tier on one node: HBM → DRAM → NVMe (D4).
3. Cluster-wide cache (this phase).
4. Product layer (this phase).

| # | Build / experiment | What it demonstrates |
|---|---|---|
| P1 | **Cluster-wide KV store**: a global prefix index the router consults, plus a shared KV store. The Redis-over-TCP baseline is compared with an RDMA data plane (NIXL or Mooncake Transfer Engine) and with LMCache | Hit rate vs cache size; cross-node fetch vs local recompute, by prompt length and transport |
| P2 | **Product layer**: TTL sweep vs memory cost, minimum cacheable length, a pricing model (cf. Anthropic ~1.25× write, ~0.1× read, 5-min TTL), tenant isolation with salted cache keys | How cache policy becomes a price; the timing side channel that salting closes |
| P3 | **Agent angle**: offload an idle session's KV during a tool call, restore on resume; predictive pre-population (patent US12547545B2) to prefetch KV *before* the tool returns | The agent-serving differentiator at cluster scale: resume TTFT with and without prefetch |
| P4 | **External comparison**: black-box probe of Anthropic / Bedrock prompt caching — cached vs uncached TTFT, observed TTL, cost per call | Grounds the home-built numbers against a production system |

**Cost plan:** build and debug everything on cheap instances (N1). Run p4/p5 multi-node work as one or
two planned sessions with every script ready before launch. Reserve with EC2 Capacity Blocks so the
second node is actually available.

### Phase 4 — Production stacks and scale · ~3–4 weekends
- Deploy **vLLM + llm-d or NVIDIA Dynamo on EKS**; compare against the mini server and explain the gap.
- **Self-host vs hyperscaler API:** run the same 2,000-paper job three ways — p4/p5 + vLLM, Bedrock
  real-time and Bedrock batch. Report cost per 1,000 papers and the break-even utilization.
- Optional: the same job on Trainium/Inferentia via vLLM-on-Neuron (closes the Neuron gap).
- Once D2–D4 and Phase 3 data exists: substantive comments or a small PR on the open agent-aware
  KV-cache issues in llm-d, SGLang and vLLM.

### Phase 5 — Capstone: Kimi K2 (1T-parameter MoE) on the mini server · ~6–8 weekends
*Added 2026-10-04.* The final goal: run the same engine, studies and agent workload on a frontier-scale
open model.

**Model facts (Kimi K2 family; K2.6 is the current release, and every K2.x checkpoint so far has been
1T total parameters — there's no 2T Kimi model as of 2026-10):**
- **Size:** 1T total parameters, 32B active per token.
- **MoE:** 384 experts, 8 routed + 1 shared per token.
- **Shape:** 61 layers (1 dense), hidden size 7,168, 64 attention heads.
- **Attention:** MLA (Multi-head Latent Attention), the DeepSeek-V3 architecture.
- **Weights:** K2-Instruct ships block-FP8 (~1 TB). K2 Thinking / K2.5 / K2.6 ship native INT4 from
  quantization-aware training (~594 GB). bf16 would be ~2 TB.

**What it changes, versus Llama:**
- **KV cache is tiny.** MLA stores one compressed latent (512 + 64 RoPE dims) per layer instead of
  per-head K and V.
  - K2: ≈ 69 KB/token in bf16. A 70B GQA model: 320 KB.
  - An 8K prompt is ≈ 0.56 GB, so the Phase 3 transfer and caching math shifts in favour of moving KV.
  - Catch: the latent isn't split by head, so tensor parallelism replicates it on every GPU. Hence
    **data-parallel attention**.
- **Decode cost depends on batch size.** At batch 1 a step reads only the ~32B active parameters. At
  large batch almost all 384 experts are touched, so the step reads nearly all weights. Serving
  economics depend on large decode batches.
- **The new communication pattern is all-to-all.** Expert parallelism sends each token to the GPUs that
  hold its 8 experts and back, every layer. Over EFA between nodes this is the hardest part of the
  whole roadmap.
- **Hardware.**
  - INT4 (~594 GB) technically fits on 8× H100 80GB (640 GB) but leaves almost nothing for KV and
    activations. Realistic minimums: **1× p5en (8× H200, 1,128 GB)**, or **2× p5 (16× H100)** over EFA.
  - FP8 K2-Instruct: Moonshot's deploy guide puts the smallest unit at 16 GPUs (H200/H20 class).
  - p4d (A100 40GB) is out.
- **Reference point.** LMSYS served Kimi K2 on 128 H200s (4 prefill nodes, 12 decode nodes) with P/D
  disaggregation, large-scale expert parallelism (DeepEP) and expert load balancing (EPLB). Phase 5
  is a 1–2 node version of that design.

| # | Build | Study | What it teaches |
|---|---|---|---|
| K0 | **Reference run**: serve K2.x INT4 with SGLang or vLLM on 1× p5en or 2× p5; agent workload C | Baseline TTFT / ITL / throughput — the target for K7 | What production MoE serving looks like, measured |
| K1 | **MLA in the mini server** (Moonlight-16B): latent KV cache, weight absorption for decode | KV bytes/token and max concurrency vs the 8B GQA model | Why MLA changes memory, caching and transfer math |
| K2 | **MoE layer**: top-8 router + shared expert, grouped GEMM via a borrowed fused-MoE kernel | Decode step time vs batch: the batch size at which nearly all experts are hot | Why MoE needs big decode batches |
| K3 | **Expert parallelism + data-parallel attention** on one 8-GPU node (Moonlight, EP=8): all-to-all dispatch/combine | All-to-all share of step time; per-expert load imbalance | The core MoE serving skill |
| K4 | **Expert load balancing**: redundant hot experts, periodic rebalancing (EPLB-style) | p99 ITL with and without balancing on skewed traffic | Stragglers set the step time in EP |
| K5 | **Quantized MoE weights**: load INT4 (compressed-tensors) or block-FP8 with borrowed kernels (Marlin-MoE / DeepGEMM) | Bytes per step and ITL vs bf16 Moonlight; quality spot-check | Why K2 ships INT4 natively |
| K6 | **Kimi K2.x across 2 nodes**: EP inside each node; across nodes either EP=16 over EFA or a prefill node + decode node (P/D) | All-to-all over EFA vs NVLink; MLA KV transfer cost for P/D; agent prefix caching at 1T scale | The capstone: frontier-model serving on own engine |
| K7 | **Gap analysis vs K0** | Explain the remaining gap (kernels, comms library, scheduler) | Same honesty rule as S11 |

**Phase 5 pending decisions:**
- **Which checkpoint:** recommend the newest text-only INT4 K2.x; skip the K2.5/K2.6 vision encoder.
- **Hardware:** 1× p5en vs 2× p5. 1× p5en is simpler for K1–K5 on the real model; 2× p5 is needed to
  practise cross-node EP.
- **All-to-all library on EFA:** DeepEP targets InfiniBand. Check current EFA support for DeepEP,
  pplx-kernels and NVSHMEM-over-libfabric before booking nodes. A plain NCCL all-to-all is the
  fallback baseline.
- **Weight logistics:** 0.6–1 TB of weights. Stage once into S3 in the same region, then load onto local
  NVMe (or FSx for Lustre) before the clock-critical session. Load time is a D5 cold-start data point.
- **Budget:** each p5-class node costs tens of dollars an hour on demand. Do K1–K5 on Moonlight (cheap),
  then book a short Capacity Block for K0 + K6 with every script ready.
- **The Mac can't run any of this beyond tiny random configs.** Moonlight in bf16 (~32 GB) needs a
  48 GB+ Mac even for slow correctness runs.

---

## Design review (2026-10-04) — open decisions

The roadmap is sound in order and scope, but the items below aren't designed yet. The first group causes
rework if decided late, so settle it before writing M2.

**A. Decide before M2 (they shape the whole codebase)**

| # | Open decision | Options | Recommendation |
|---|---|---|---|
| 1 | **Engine process architecture** | Single-process asyncio loop vs split API frontend ↔ engine core (scheduler) ↔ one GPU worker per device, over ZMQ or queues | Split at M2. D1 (TP needs one worker per GPU), D2 (replicas), D3 (separate prefill/decode engines) all assume it; a monolith means a rewrite at Phase 2 |
| 2 | **Workload C (agent) spec** | Purely synthetic vs replaying public traces (Mooncake/Kimi, Azure LLM inference traces, BurstGPT) | Synthetic generator with explicit distributions: turns per session, shared-prefix size, tool-pause duration, Poisson arrivals. Validate arrival shape against a public trace. Abstract patterns only, no Oracle data |
| 3 | **SLO and metric definitions** | Fixed numbers per workload | E.g. TTFT p99 ≤ 1 s (B, C), ≤ 3 s (A); TPOT p99 ≤ 50 ms. Goodput = requests/s meeting both. Fix them now so S4–S10 and D3 are comparable |
| 4 | **Correctness method** | Greedy token match vs logit tolerance | Greedy match vs Hugging Face holds only at batch 1. bf16 kernels aren't batch-invariant, so batched runs diverge legitimately. Add teacher-forced logit comparison with a tolerance, plus an fp32 test mode |
| 5 | **Benchmark client** | Write one vs reuse `vllm bench serve` against own OpenAI-compatible server | Reuse vLLM's client for A/B (it's only the load generator), write own generator for C. Same client against both engines makes S11 fair |
| 6 | **Observability** | Logs only vs metrics + per-request timeline | Prometheus metrics plus a per-request timeline (queue, prefill, decode, cache-hit tokens) from M2. Every study reads from it |

**B. Decide by M5/M6**

| # | Open decision | Options | Recommendation |
|---|---|---|---|
| 7 | **Attention backend and KV layout** | FlashInfer paged API vs flash-attn `flash_attn_with_kvcache` with block table; block size; NHD vs HND; layer-major vs block-major | FlashInfer, block size 16, per-layer paged tensors. Check the layout lets D3/N3 ship whole blocks layer by layer (overlaps transfer with prefill) |
| 8 | **Block-hash spec** | Python `hash()` vs a specified hash | A specified hash: sha256 (or xxh3-128) over parent hash + token-id bytes + extra keys (model + revision, KV dtype, tenant salt, LoRA id). The Go/Rust router must compute identical hashes; salting from day one makes P2 a policy change, not a migration |
| 9 | **Scheduler policy** | FCFS vs priority; preemption victim choice | Token budget per step, FCFS admission, preempt newest first. Leave a priority hook so resumed agent sessions can jump the queue later (D4/P3) |

**C. Decide before Phase 2**

| # | Open decision | Options | Recommendation |
|---|---|---|---|
| 10 | **Systems language** | Go (llm-d, Kubernetes) vs Rust (SGLang router, Dynamo core) | Decide before D2; the router is where it's used |
| 11 | **Router state and tokenization** | Router keeps an approximate radix tree per replica (SGLang) vs replicas publish KV block events (llm-d, Dynamo); tokenize in router vs route on text | KV events: exact, and it becomes P1's global index unchanged. Tokenize in the router so hashes match decision 8 |
| 12 | **Session / agent API contract** | Stateless OpenAI API vs extension | Add `session_id` plus tool-call pause/resume hints (and optional expected tool duration) via header or `extra_body`. D4 and P3 can't work without the engine knowing a session is paused |
| 13 | **Pipeline parallelism gap** | N2 compares TP=16 vs TP=8+PP=2, but PP is never built | Add D1b (pipeline parallelism, 2 stages) to the mini server, or run N2 on vLLM explicitly as a measurement, not a build |
| 14 | **KV transfer mechanism** | Intra-node: CUDA IPC / NCCL send-recv. Inter-node: NCCL over EFA, NIXL, UCX, or gRPC via CPU | NCCL send/recv for D3 (simplest), then NIXL for N3/P1 to match what Dynamo and llm-d use |
| 15 | **Failure model for D5** | Which failures; streaming resume semantics | Process kill, GPU OOM, node loss. Router replays prompt + already-emitted tokens to another replica (recompute). Measure cold start by weight source: S3 vs local NVMe vs peer GPU |
| 16 | **Multi-GPU dev hardware** | Develop on the p4/p5 vs a cheap multi-GPU box | g6.12xlarge (4× L4, PCIe) for correctness of D1–D3. No NVLink, so performance numbers only on p4/p5 |

**D. Decide before Phase 3**

| # | Open decision | Options | Recommendation |
|---|---|---|---|
| 17 | **Data plane vs control plane for the cluster cache** | Redis for everything vs Redis/etcd for the index plus RDMA for bytes | Split them. Redis can't do GPU-direct RDMA, so a Redis-only P1 measures TCP and serialization, not EFA. Keep Redis-over-TCP as the explicit baseline |
| 18 | **KV compatibility across engines** | Blocks produced at TP=8 are sharded by head; a TP=4 engine can't use them directly | Put layout (TP degree, KV dtype) in the cache key, or reshard on fetch and measure that cost |
| 19 | **Eviction and TTL across tiers** | LRU vs cost-aware | Score = recompute cost (∝ prefix length) × reuse probability (paused agent sessions high). Write-back to DRAM on eviction from HBM |
| 20 | **AWS access** | — | Before booking anything: check P-instance vCPU quota (often 0 by default), Capacity Blocks for 2 nodes, cluster placement group in one AZ, EFA-enabled AMI, AWS Budgets alarm. Confirm the account is personal, so the work and IP are clearly his |

**E. Cross-cutting**

| # | Open item | Recommendation |
|---|---|---|
| 21 | **Dollar budget per phase** | Not set. Set one per phase and enforce it with AWS Budgets alerts |
| 22 | **Exit criteria and write-ups** | Each milestone is done when its study note is written; each phase is done when its charts are in the README and one public write-up ships |
| 23 | **Timeline realism / MVP cut** | ~25 weekends (plus ~7 for Phase 5) assumes almost every weekend alongside Track 1. If time runs short, the interview-ready core is **Phase 1a + M7 + D2 + D3**. D5, N1–N4, P2 and P4 are the first to cut. Phase 5 is a capstone, and applications never wait for it |
| 24 | **Phase 5 design choices** | See "Phase 5 pending decisions" above (checkpoint, hardware, EFA all-to-all library, weight logistics, budget). Decide before K3, because K3's process layout must support EP + DP-attention |

---

## Practicalities

- **Cost hygiene:** logic on the MacBook; CUDA path on a single-L4 g6.xlarge and a g6.12xlarge for
  multi-GPU correctness. Use the p4/p5 (bills for all 8 GPUs) only for:
  - the two Phase 1 study reruns
  - Phase 2 multi-GPU performance runs
  - the planned Phase 3 multi-node sessions
  - the Phase 5 Kimi K2 session
  - final benchmarks

  Script runs before starting the instance.
- **Language decision (supersedes "Rust or C++?" for this track):** the distributed-serving ecosystem
  leans **Go** (llm-d, Kubernetes) and **Rust** (SGLang router, Dynamo's core). C++ matters mainly for the
  deferred kernel track and quant/chip roles. Decide before D2.
- **Reading, in order of need:**
  - Phase 1: Orca and vLLM; Sarathi-Serve (M7); speculative decoding papers (M10).
  - Phase 2: Megatron-LM (D1); DistServe, Splitwise and Mooncake (D3–D4); llm-d and Dynamo docs (D2).
  - Phase 3: Mooncake, LMCache and NIXL docs; NCCL/EFA tuning guides.
  - Phase 4: llm-d and Dynamo docs.
  - Phase 5: DeepSeek-V2 paper (MLA), DeepSeek-V3 technical report (MoE, EP, EPLB), Kimi K2
    technical report, LMSYS "Deploying Kimi K2 … on 128 H200 GPUs", DeepEP README.
  - Throughout: *How to Scale Your Model* ch. 7–8.
  - Summaries live in `inference_engineering_notes.md` §21.
- **References to compare against, not copy:** nano-vllm (~1,200 lines), mini-sglang (LMSYS). Credit
  them in READMEs.
- **Model access:** the Llama models are gated on Hugging Face, so request access before M0.
- **Applications:** Track 1 (agent / AI-engineer roles) never waits for this. The inference-platform
  queue opens once mini-server v1 ships with the S0–S6 studies.

## Target resume bullets (placeholders until real numbers exist — never ship them unfilled)

- *Built LLM inference server from scratch — continuous batching, paged KV cache, prefix caching, chunked
  prefill, CUDA graphs, quantization, speculative decoding — measuring each step: ⟨X⟩ → ⟨Y⟩ tok/s on
  mixed traffic, ⟨Z⟩× TTFT reduction on shared-prefix agent workloads; benchmarked against vLLM.*
- *Extended it to multi-GPU serving — tensor parallelism, prefix-aware routing across replicas,
  prefill/decode disaggregation and KV-cache offload — cutting p99 agent-task latency ⟨X⟩% and
  GPU-seconds per task ⟨Y⟩% vs round-robin colocated serving.*
- *Scaled it across 2-node GPU cluster over EFA with cluster-wide KV cache and predictive prefetch
  for agent tool calls: ⟨X⟩% cache hit rate, ⟨Y⟩× faster resume TTFT vs recompute, and the measured
  fetch-vs-recompute crossover at ⟨Z⟩ tokens.*
- *Served 1T-parameter MoE model (Kimi K2) on own engine across ⟨N⟩ GPUs with expert parallelism,
  data-parallel MLA attention and expert load balancing: ⟨X⟩ tok/s decode, within ⟨Y⟩% of SGLang.*

---

## Deferred: kernel track (revisit after Phase 2, or if a target role requires it)

| # | Kernel | Notes |
|---|---|---|
| K1 | Naive attention in PyTorch | Materializes the n×n matrix; baseline |
| K2 | Fused prefill attention in **Triton** (FlashAttention-style tiling, online softmax) | Follow Triton's official fused-attention tutorial; ~1–2 weekends |
| K3 | Paged decode attention in Triton, reading KV via the block table | Drops into mini-server M5 in place of FlashInfer; measure by MBU |

Prerequisites: GPU execution model, SRAM vs HBM, tiling/coalescing, Nsight Compute; *Programming
Massively Parallel Processors* ch. 1–6. Dedicated kernel-engineering roles are a separate career track —
the goal here would be "can write, profile and reason about attention kernels", not a pivot.
