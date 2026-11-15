const FIRST = "g6.xlarge (1× L4 24 GB), Llama-3.2-1B, bf16";
const OFFICIAL = "p4de (A100 80 GB) or p5 (H100), Llama-3.1-8B, bf16";

const UNITS = [
{id:"0",key:"unit-0",title:"Orientation",short:"How the course works",
 goal:"Learn how every module is built (lesson, prediction, lab, takeaway), what hardware and models we use, and which tests guard every measurement.",
 outcome:"You can read any later module and know what it is testing, on which instance, against which baseline."},
{id:"1a",key:"unit-1a",title:"Core engine → v1",short:"M0–M6 on one GPU",
 goal:"Build a working LLM server from a bare generation loop up to paged KV and prefix caching, and measure what each step buys.",
 outcome:"Tag v1: gates G0–G2 and R1–R4 green, S0–S6 measured on A100/H100, README charts and one write-up. Inference-platform applications open here.",
 facts:[["Milestones","M0 → M6, all built and tagged m0–m6"],["Studies","S0–S6, scripts written, pending EC2 runs"],["Hardware","L4 first pass, A100/H100 official"],["Model","Llama-3.2-1B dev, Llama-3.1-8B studies"]]},
{id:"1b",key:"unit-1b",title:"Optimizations → v1.1",short:"M7–M10 + vs vLLM",
 goal:"Add the four optimizations production engines ship with, each as its own measured study, then compare the whole engine with vLLM.",
 outcome:"Tag v1.1: S0–S10 rerun on A100/H100, R4 soak rerun, and an honest gap analysis against vLLM (S11).",
 facts:[["Milestones","M7 chunked prefill, M8 CUDA graphs, M9 quantization, M10 speculative decoding"],["Hard gates","G2 for M9, exact G0 for M10"],["Hardware","A100/H100"],["Status","Planned"]]},
{id:"2",key:"unit-2",title:"Distributed, one node",short:"TP, router, P/D, tiers",
 goal:"Take the engine across the GPUs of one 8-GPU node: split a model, route between replicas, separate prefill from decode, and tier the KV cache.",
 outcome:"Measured answers to when each multi-GPU technique pays, on the agent workload, with every run matching TP=1 output (G1).",
 facts:[["Builds","D1 TP, D1b PP (optional), D2 router, D3 P/D, D4 KV tiering, D5 reliability"],["Dev hardware","g6.12xlarge (4× L4) for correctness"],["Perf hardware","p4/p5 (NVLink)"],["Status","Planned"]]},
{id:"3",key:"unit-3",title:"Multi-node and cluster caching",short:"EFA, N1–N4, P1–P4",
 goal:"Cross the node boundary, where bandwidth drops by about 10×, and build prompt caching that works across the cluster.",
 outcome:"The TP-inside / replicas-across rule confirmed or broken with numbers, a measured fetch-vs-recompute crossover, and a cluster KV cache with agent prefetch.",
 facts:[["Model","Llama-3.3-70B"],["Hardware","2× cheap GPU nodes (N1), then 2× p4d/p5 over EFA"],["Cache layers","engine → node tiers → cluster → product"],["Status","Planned"]]},
{id:"4",key:"unit-4",title:"Production stacks",short:"llm-d / Dynamo, cost",
 goal:"Check the home-built system against production tooling and real prices.",
 outcome:"A gap analysis against vLLM + llm-d or Dynamo on EKS, and cost per 1,000 papers self-hosted vs Bedrock, with the break-even utilisation.",
 facts:[["Stack","EKS, vLLM, llm-d or NVIDIA Dynamo"],["Cost study","p4/p5 + vLLM vs Bedrock real-time vs batch"],["Optional","Trainium/Inferentia via Neuron"],["Status","Planned"]]},
{id:"5",key:"unit-5",title:"Kimi K2 capstone",short:"MLA, MoE, EP",
 goal:"Serve a 1T-parameter MoE model (32B active) on the same engine across 1–2 nodes: a small version of LMSYS's 128-GPU deployment.",
 outcome:"Kimi K2 decode throughput on your engine and a gap analysis against SGLang/vLLM on the same hardware.",
 facts:[["Dev model","Moonlight-16B-A3B (same architecture)"],["Target","Kimi K2.x INT4, ~594 GB"],["Hardware","1× p5en or 2× p5"],["Status","Planned"]]},
{id:"A",key:"unit-appendix",title:"Reference",short:"Formulas, GPUs, decisions",
 goal:"Everything you reach for while predicting: formulas, hardware numbers, the decision log, the glossary.",outcome:""}
];

const ORIENTATION = `
<div class="sec"><h2>Why build your own engine</h2>
<p>Stock vLLM already has paged KV, continuous batching, CUDA graphs, chunked prefill and prefix caching switched on. Toggling them measures small deltas on someone else's design. Here you build the server one optimization at a time, and every step is measured against your own previous version, so each gain has a cause you wrote.</p></div>
<div class="sec"><h2>How every module works</h2>
<ol class="steps">
<li><div><b>Lesson.</b> The concept in plain words, then the formula, then a worked example with real numbers.</div></li>
<li><div><b>Predict before you run.</b> Questions you answer from the lesson's formulas. Write a number and a reason. Your answers are kept in this browser.</div></li>
<li><div><b>Pass the gates.</b> Correctness tests (G0–G2) run before anything is measured. A speedup on wrong output is not a result.</div></li>
<li><div><b>Lab on EC2.</b> Run the study against the previous milestone's git tag, same workload, same hardware. Capture one profile (PyTorch profiler, Nsight Systems).</div></li>
<li><div><b>Explain the gap.</b> Compare measured with predicted and explain the difference using the profile. Most of the learning happens here.</div></li>
<li><div><b>Record it.</b> <code>docs/studies/SN-name.md</code>: setup, prediction, chart, result, one paragraph of explanation.</div></li>
</ol></div>
<div class="sec"><h2>Hardware: EC2 only</h2>
<p>Every run, including first-pass checks, happens on EC2. Scripts are ready before an instance starts, and each session runs a batch of studies.</p>
<div class="tbl"><table><thead><tr><th>Tier</th><th>Instance</th><th>Used for</th></tr></thead><tbody>
<tr><td>First pass</td><td>g6.xlarge (1× L4 24 GB)</td><td>CUDA path, kernel integration, study dry runs on Llama-3.2-1B</td></tr>
<tr><td>Multi-GPU correctness</td><td>g6.12xlarge (4× L4, PCIe)</td><td>D1–D3 correctness, Moonlight-16B (Phase 5)</td></tr>
<tr><td>Official numbers</td><td>p4d / p4de (A100), p5 (H100)</td><td>S0–S10 reruns, Phase 2 performance, Phase 3 multi-node</td></tr>
<tr><td>Capstone</td><td>p5en (8× H200) or 2× p5</td><td>Kimi K2 (K0, K6)</td></tr>
</tbody></table></div></div>
<div class="sec"><h2>Models</h2>
<div class="tbl"><table><thead><tr><th>Model</th><th>Role</th><th>KV per token</th></tr></thead><tbody>
<tr><td>Llama-3.2-1B</td><td>First-pass runs, draft model for M10, small-model side of S8</td><td class="n">32 KB</td></tr>
<tr><td>Llama-3.1-8B</td><td>Every official study in Phases 1–2</td><td class="n">128 KB</td></tr>
<tr><td>Llama-3.3-70B</td><td>Phase 3 multi-node layouts</td><td class="n">320 KB bf16 · 160 KB fp8</td></tr>
<tr><td>Moonlight-16B-A3B</td><td>Phase 5 development (DeepSeek-V3 architecture)</td><td class="n">MLA latent</td></tr>
<tr><td>Kimi K2.x</td><td>Phase 5 target: 1T total, 32B active</td><td class="n">≈ 69 KB</td></tr>
</tbody></table></div></div>
<div class="sec"><h2>Workloads</h2>
<div class="tbl"><table><thead><tr><th>Workload</th><th>Shape</th><th>Data</th><th>Stresses</th></tr></thead><tbody>
<tr><td>A · Summarize</td><td class="n">~8,000 in → ~150 out</td><td>arXiv papers</td><td>Prefill (compute)</td></tr>
<tr><td>B · Generate</td><td class="n">~100 in → ~1,000 out</td><td>UltraChat-200k</td><td>Decode (memory)</td></tr>
<tr><td>C · Agent</td><td class="n">~3.5K shared prefix, ~6 turns</td><td>Seeded synthetic generator (decision 2)</td><td>Prefix reuse, sessions, bursts</td></tr>
<tr><td>Mix</td><td class="n">30% A / 70% B</td><td>Both</td><td>Fairness, head-of-line blocking, interference</td></tr>
</tbody></table></div></div>
<div class="sec"><h2>Service-level objectives</h2>
<p>Fixed once (decision 3) so every study is comparable. <b>Goodput</b> is requests per second that meet both targets.</p>
<div class="tbl"><table><thead><tr><th>Metric</th><th>A</th><th>B, C</th></tr></thead><tbody>
<tr><td>TTFT p99</td><td class="n">≤ 3 s</td><td class="n">≤ 1 s</td></tr>
<tr><td>TPOT p99</td><td class="n">≤ 50 ms</td><td class="n">≤ 50 ms</td></tr>
</tbody></table></div></div>
<div class="sec"><h2>Tests that guard every measurement</h2>
<p>Each starts at one milestone and reruns at every later one.</p>
<div class="tbl"><table><thead><tr><th>Test</th><th>From</th><th>Checks</th><th>Passes when</th></tr></thead><tbody>
<tr><td>G0</td><td>M0</td><td>Greedy tokens vs Hugging Face, batch 1, same device</td><td>Identical tokens on short, long, multilingual and code prompts</td></tr>
<tr><td>G1a</td><td>M3</td><td>Batched vs batch-1 in fp32</td><td>Identical tokens, logits within 1e-4</td></tr>
<tr><td>G1b</td><td>M3, M5, D1</td><td>Batched vs batch-1 in bf16, teacher-forced logits</td><td>KL and max logit diff ≤ 2× HF's own batch noise; top-1 agrees where margin &gt; 0.5</td></tr>
<tr><td>G2</td><td>M1</td><td>WikiText-2 perplexity + ~200 GSM8K questions</td><td>Within noise of the HF baseline; hard gate for M9 and K5</td></tr>
<tr><td>R1</td><td>M2</td><td>Clients disconnect mid-stream</td><td>Request leaves the scheduler; from M5 its blocks are freed</td></tr>
<tr><td>R2</td><td>M4</td><td>2–3× the load at the knee</td><td>Bounded queue, 429/503 rejections, no OOM; admitted goodput holds</td></tr>
<tr><td>R3</td><td>M5</td><td>KV block accounting after every run</td><td>free + used = total; refcounts return to zero (M6)</td></tr>
<tr><td>R4</td><td>v1</td><td>1–2 hour soak on Mix + C</td><td>No memory growth or leaks; p99 drift under 10%</td></tr>
</tbody></table></div></div>`;

const ITEMS = [];
const M = o => ITEMS.push(Object.assign({kind:"module"},o));
const D = o => ITEMS.push(Object.assign({kind:"decision"},o));

/* ---------------- Unit 1a ---------------- */
M({unit:"1a",id:"m1",n:1,title:"How an LLM generates text",tags:["M0","S0","G0",["memory","m"]],status:"ready",
obj:["Trace one forward pass of a decoder-only transformer","Explain why generation is one token at a time","Predict the cost of generating without a KV cache"],
lesson:`<p>A decoder-only model like Llama turns a sequence of token ids into a probability for the next token. Each of its layers does two things: <b>attention</b>, where every position looks back at earlier positions, and an <b>MLP</b>, which transforms each position on its own. Greedy decoding picks the most likely token, appends it, and runs the model again.</p>
<p>Done naively, every new token re-runs the model over the <i>whole</i> sequence. Step <i>t</i> processes <i>t</i> tokens, so the work per token grows with length and the total work for <i>n</i> tokens grows with <i>n</i>². Almost all of that is repeated: the keys and values for earlier positions never change.</p>
<p>Two costs set the time of a step. The model reads all its weights from GPU memory once per step, and it does about 2 FLOPs per weight per token processed. Short sequences are limited by the weight read; long ones by the arithmetic.</p>`,
formula:`FLOPs per token          ≈ 2 × params
step t without a cache   ≈ 2 × params × t  (+ attention 2·t²·d·L)
total for n tokens       ≈ params × n²        → quadratic`,
example:`<b>Worked example</b>Llama-3.2-1B (1.24B params, 2.5 GB in bf16) on an L4 (121 TFLOPS, 300 GB/s). At t = 128 a step needs 3.2×10¹¹ FLOPs (~5 ms at 50% MFU) but must read 2.5 GB of weights (~8 ms), so it is memory-bound. At t = 4,096 it needs 1×10¹³ FLOPs (~170 ms): now compute-bound, ~20× slower than at 128.`,
diag:"prefilldecode",
predict:["Step time at L = 128 on the L4: compute-bound or bandwidth-bound? Give a number in ms.","Step time at L = 4,096: how many times the L = 128 time, and why?","At what L does the n² attention term become a noticeable share of the step's FLOPs?","Shape of cumulative time for 256 generated tokens: linear, quadratic, or in between?"],
lab:{name:"S0 · Cost of no cache",why:"Measure the waste the KV cache removes, so M1's gain has a baseline you produced yourself.",
spec:[["Instance","g6.xlarge (1× L4)"],["Model","Llama-3.2-1B, bf16 (8B if time allows)"],["Workload","Batch 1, random token ids; step time at L = 128 … 4,096; 256 tokens from a 16-token prompt"],["Command","<code>python benchmarks/s0_no_cache.py --model-dir models/Llama-3.2-1B</code>"],["Gates","G0: greedy tokens identical to Hugging Face"]],
metrics:[["Step time vs L","Where the line bends from flat (memory) to rising (compute)"],["Cumulative time","Linear at first, then quadratic"],["MFU / MBU per L","Which resource limits each length"]],
expected:"Per-token time roughly flat while weight reads dominate, then rising linearly with length; total time for n tokens bends quadratic."},
learn:"The KV cache exists because recomputing the past is quadratic and almost entirely repeated work.",
next:"Store each position's keys and values once and reuse them (M1).",
check:["Why does a short no-cache step cost about the same as a short cached one?","Which term makes total cost quadratic?"]});

M({unit:"1a",id:"m2",n:2,title:"KV cache, prefill vs decode",tags:["M1","S1","G2",["compute","c"],["memory","m"]],status:"ready",
obj:["Size the KV cache per token","Separate a request into prefill and decode","Place each on the roofline and predict MFU and MBU"],
lesson:`<p>The <b>KV cache</b> keeps each layer's key and value vectors for every position, so a new token only computes its own and attends to the stored ones. That splits a request into two phases with opposite characters.</p>
<p><b>Prefill</b> processes the whole prompt in one pass. Thousands of tokens share one read of the weights, so it does many FLOPs per byte: compute-bound. It sets <b>TTFT</b>. <b>Decode</b> produces one token per step and re-reads every weight to do it: about 1 FLOP per byte at batch 1, deeply memory-bound. It sets <b>ITL</b>.</p>
<p>The <b>roofline</b> makes this exact. A GPU's balance point is peak FLOP/s divided by memory bandwidth. Work with lower arithmetic intensity is limited by bandwidth; higher, by compute. <b>MFU</b> (model FLOPs utilisation) measures prefill; <b>MBU</b> (memory bandwidth utilisation) measures decode. Every later optimization targets one side of this line.</p>`,
formula:`KV bytes/token = 2 × layers × kv_heads × head_dim × bytes
  8B: 2 × 32 × 8 × 128 × 2 = 128 KB
decode floor (batch 1) = weight bytes ÷ bandwidth
MFU = achieved FLOP/s ÷ peak     MBU = (weights + KV read) / step ÷ peak BW`,
example:`<b>Worked example</b>8B on an A100 80 GB: 16 GB of weights ÷ 2.04 TB/s = <b>7.9 ms</b> per decode token at best. That step does 16 GFLOPs, i.e. 2 TFLOP/s, so decode MFU ≈ 0.7%. An 8K prefill needs 2 × 8×10⁹ × 8,192 ≈ 1.3×10¹⁴ FLOPs: ~0.85 s at 50% MFU. At context 2,000 the KV read adds 256 MB to the 16 GB of weights: about +1.6% on ITL.`,
diag:"roofline",
predict:["Decode floor for the 8B on your GPU. Will measured ITL land near it? What MBU do you expect, and why not 100%?","TTFT at 128 vs 4,096 tokens: linear in prompt length? Past roughly what length does the n² attention term make it grow faster?","Prefill MFU at 4K tokens vs decode MFU. Why are they orders of magnitude apart?","Does ITL rise as context grows from 128 to 2,000 tokens? By how much?","Versus S0 (tag m0): per-token time at context 2,048 without and with the cache."],
lab:{name:"S1 · Prefill vs decode",why:"Show that the two phases are different problems, with numbers you will reuse in every later prediction.",
spec:[["First pass",FIRST],["Official",OFFICIAL],["Workload","Batch 1, random ids. Sweep A: prompt 128 → 16,384, 16 out. Sweep B: 128 in, 2,000 out. Then workloads A and B."],["Command","<code>python benchmarks/s1_prefill_decode.py --model-dir models/Llama-3.1-8B --gpu a100-80</code>"],["Gates","G0; G2 baseline recorded with <code>benchmarks/g2_quality.py --engine hf --save-baseline</code>"]],
metrics:[["TTFT vs prompt length","Linear, then superlinear where attention n² grows"],["ITL vs context","Flat near the decode floor, slow rise from KV reads"],["Prefill MFU","Expected 40–60%"],["Decode MBU / MFU","Expected MBU 70–85%, MFU under 1%"],["Phase share of A and B","A ≈ 30–40% prefill; B ≈ 99% decode"]],
expected:"TTFT linear in prompt length, superlinear past ~8K. ITL flat near the decode floor. Prefill MFU 40–60%; decode MBU 70–85% with MFU below 1%."},
learn:"Prefill is compute-bound and decode is memory-bound. A single request leaves most of the GPU's arithmetic idle during decode.",
next:"Serve requests over HTTP and see what one-at-a-time serving costs under load (M2).",
check:["Why does batch-1 decode use under 1% of the GPU's FLOPs?","What is the KV size of an 8K prompt on the 8B?"]});

D({unit:"1a",n:"1–6",title:"Decisions before M2",status:"decided",
choice:"(1) Frontend and engine core in separate processes over ZMQ + msgpack. (2) Workload C is a seeded synthetic generator with sweepable distributions. (3) SLOs: TTFT p99 ≤ 3 s (A) / 1 s (B, C), TPOT p99 ≤ 50 ms. (4) Layered correctness: G0, G1a, G1b, G2. (5) Own asyncio benchmark client, cross-checked with vllm bench serve. (6) Per-request timeline, engine step log, Prometheus /metrics.",
why:"They shape the whole codebase. TP, replicas and P/D all assume a split process design; deciding later means a rewrite at Phase 2."});

M({unit:"1a",id:"m3",n:3,title:"Serving and queueing",tags:["M2","S2","R1"],status:"ready",
obj:["Break a request's TTFT into its stages","Use utilisation to predict queue wait","Explain why serving one request at a time wastes the GPU"],
lesson:`<p>A server adds stages around the model: HTTP and JSON parsing, tokenization, a hop from the frontend to the engine process, time in the queue, prefill, and streaming the first token back. The <b>per-request timeline</b> records each, so fixed overhead is never mistaken for GPU time.</p>
<p>Once requests arrive at random, queueing theory takes over. With service time <i>S</i> and arrival rate λ, <b>utilisation</b> ρ = λ·S. With Poisson arrivals and fixed service time (an M/D/1 queue), mean wait grows as ρ/(1−ρ): modest at ρ = 0.5, four and a half service times at 0.9, unbounded past 1. Throughput caps at 1/S however much load you offer.</p>
<p>That cap is the problem. While one request decodes, the GPU's compute sits almost idle (S1). Serving one at a time throws that capacity away, and the queue grows to show it.</p>`,
formula:`ρ = λ × S
M/D/1 mean wait  W_q = ρ·S / (2(1 − ρ))
  ρ = 0.5 → 0.5 S    ρ = 0.8 → 2 S    ρ = 0.9 → 4.5 S
throughput ≤ 1 / S`,
example:`<b>Worked example</b>8B on an A100, 128 in / 32 out: prefill ≈ 13 ms plus 31 decode steps near 8 ms ≈ 0.26 s, so capacity is about 3.8 req/s. At ρ = 0.9 the mean wait is ~1.2 s, already past the 1 s TTFT SLO before any tail.`,
diag:"queue",
predict:["Non-GPU overhead per request (HTTP, tokenize, ZMQ both ways, SSE): how many ms, and what share of TTFT for a 32-token prompt? Which stage is largest?","Service time S for 128 in / 32 out on the 8B.","M/D/1 mean wait at ρ = 0.5, 0.8, 0.9 in units of S. At what ρ does p99 TTFT cross 1 s?","Above ρ = 1, how does wait grow with requests sent, and why does throughput cap at 1/S?","What fraction of the GPU sits idle at the capacity limit?"],
lab:{name:"S2 · Overhead and queueing",why:"Separate fixed serving overhead from GPU time, and find the load where one-at-a-time serving breaks the SLO.",
spec:[["First pass",FIRST],["Official",OFFICIAL],["Part 1","100 sequential requests, 32-token prompt, 1 output token, idle server"],["Part 2","128 in / 32 out; measure S; Poisson arrivals at ρ = 0.2 … 1.2; ≥ 1,000 requests per point after a 30 s warm-up"],["Command","<code>python -m minillm.server --model-dir models/Llama-3.1-8B --timeline-log results/s2_timeline.jsonl</code> then <code>python benchmarks/s2_overhead_queueing.py --base-url http://127.0.0.1:8000 --vocab-size 128256 --per-rate 1200</code>"],["Gates","G0 through the API; R1 cancellation"]],
metrics:[["TTFT breakdown","HTTP, tokenize, frontend→engine, queue, prefill, emit"],["Queue wait vs ρ","Against the M/D/1 curve"],["p99 TTFT vs ρ","Where it crosses the 1 s SLO"],["Throughput","Flattens at 1/S"]],
expected:"A few ms of fixed overhead. Latency hockey-sticks as ρ approaches 1, and throughput stops at 1/S."},
learn:"Queueing theory predicts the knee. One-at-a-time serving hits it while the GPU is mostly idle.",
next:"Run several requests in one step so they share the weight read (M3).",
check:["Why does a 30 s warm-up matter for the measured wait?","What does R1 protect against?"]});

M({unit:"1a",id:"m4",n:4,title:"Batching",tags:["M3","S3","G1a","G1b",["memory","m"]],status:"ready",
obj:["Explain why batching is nearly free while decode is memory-bound","Find the batch size where the knee should be","Quantify the padding and head-of-line cost of static batching"],
lesson:`<p>A decode step reads all the weights whether it serves 1 request or 32. Put B requests in one step and they share that read: B times the tokens for almost the same time. Arithmetic intensity rises to about B FLOPs per byte, so throughput grows nearly linearly until B reaches the balance point, where the step turns compute-bound.</p>
<p>Where the knee lands depends on the GPU. The roofline puts it near B ≈ 153 on an A100 80 GB and B ≈ 400 on an L4, both beyond a sweep that stops at 64. So inside the sweep the flattening comes from other costs, not from compute running out.</p>
<p>Two things erode the ideal. Each request also reads its own KV cache, which grows with B × context. And <b>static batching</b> holds the batch together until its longest member finishes: finished rows keep occupying slots, and new requests wait for the whole batch to drain. This is head-of-line blocking.</p>
<p>Batching also changes numerics. bf16 kernels pick different reduction orders at different batch sizes, so batched logits differ slightly from batch-1 ones. That is why correctness splits into G1a (fp32, exact) and G1b (bf16, within Hugging Face's own batching noise).</p>`,
formula:`bytes per decode step = weights + B × context × KV bytes/token
arithmetic intensity  ≈ B FLOP/byte (bf16)
roofline knee         ≈ B where intensity = balance point
                        (A100-80 ≈ 153, H100 ≈ 295, L4 ≈ 400)
static waste          = 1 − mean(len) / max(len) per batch`,
example:`<b>Worked example</b>8B, B = 32 at context 600: KV read = 32 × 600 × 128 KB = 2.5 GB on top of 16 GB of weights, so the step is ~15% slower than at B = 1 while producing 32× the tokens. With output lengths uniform 16–512 in batches of 16, the longest member averages ~483 tokens against a mean of 264: about 45% of decode rows are wasted.`,
diag:"batching",
predict:["Decode step time at B = 1 and B = 32: how much slower, and why?","At what B does arithmetic intensity reach your GPU's balance point? Is that inside the 1–64 sweep?","Throughput at B = 64 vs B = 1: ideal is 64×. What do you expect, and what limits it inside the sweep (KV reads, attention, Python per-step overhead)?","Lengths uniform 16–512 in batches of 16: what fraction of decode rows is wasted? How long do queued requests wait?","Which of these costs does continuous batching remove, and which remain?"],
lab:{name:"S3 · Batch-size sweep",why:"Measure what batching recovers from S2's idle GPU, and what static batching still wastes.",
spec:[["First pass",FIRST],["Official",OFFICIAL],["Part A","B = 1 → 64 workload-B requests decoding together, equal lengths"],["Part B","128 requests, outputs uniform 16–512, queued at once, static batches of 16"],["Command","<code>python benchmarks/s3_batch_sweep.py --model-dir models/Llama-3.1-8B --gpu a100-80</code>"],["Gates","G1a fp32 exact; G1b bf16 within 2× HF batch noise"]],
metrics:[["Step time vs B","Nearly flat while memory-bound"],["Tokens/s vs B","Near-linear, then the knee"],["Wasted rows","Share of decode rows on finished requests"],["Queue wait","How long later batches wait"]],
expected:"Throughput near-linear in B while memory-bound, ITL nearly flat. The roofline knee lies past B = 64 on A100, H100 and L4, so any flattening inside the sweep comes from KV reads and overhead. On mixed lengths, short requests wait for the longest."},
learn:"Batching uses the compute decode leaves idle. Static batching gives some of it back to padding and head-of-line blocking.",
next:"Let requests join and leave the batch at every step (M4).",
check:["Why does the knee sit at a larger B on an L4 than on an A100?","Why can't bf16 batched output be required to match batch-1 exactly?"]});

D({unit:"1a",n:"9",title:"Scheduler policy",status:"decided",
choice:"Iteration-level scheduling, FCFS admission with no skipping, slots reserving max_model_len, prefill-first with a token budget, 503 past max_waiting. From M5: preempt the newest request first, resume by recompute.",
why:"FCFS is the fairest policy that's easy to reason about and matches vLLM's default. Reserving full slots is the honest baseline for S5. Prefill-first gives the best TTFT, and its cost (decode stalls) is what S4 and S7 expose.",
rej:"Rejected: decode-first (static batching again), shortest-job-first (output length unknown, starves long requests), mixed steps before chunked prefill."});

M({unit:"1a",id:"m5",n:5,title:"Continuous batching and scheduling",tags:["M4","S4","R2"],status:"ready",
obj:["Describe iteration-level scheduling","Predict the goodput gain on mixed-length traffic","Identify the prefill stall it introduces"],
lesson:`<p><b>Continuous batching</b> (Orca's idea) makes the batch decision again at every step. A finished request leaves immediately; a waiting one joins the next step. Slots are busy only for their own request's lifetime, so the wasted rows and head-of-line blocking of static batching mostly disappear.</p>
<p>The scheduler now has policy choices. <b>Admission</b> is first-come-first-served with no skipping, so a long prompt can't be starved. A <b>token budget</b> caps how many prompt tokens one prefill step computes. <b>Backpressure</b> bounds in-flight requests and rejects the rest with 503, so overload degrades into rejections rather than out-of-memory crashes.</p>
<p>It creates a new problem. Each prefill step pauses every running decode. When an 8K prompt arrives, all decoding users wait ~0.85 s for their next token. Two metrics see this differently. <b>ITL</b> is each gap between tokens, so the stall shows up directly in its tail. <b>TPOT</b> is a request's decode time divided by its tokens, so a few long gaps are averaged away. A request can meet the 50 ms TPOT target while its p99 ITL is many times worse. Chunked prefill (M7) is the fix.</p>`,
formula:`M3 batch time     = time of its longest member
M4 slot busy time = its own request only
decode stall per A admission ≈ one 8K prefill ≈ 0.85 s (A100, 8B)
TPOT = decode time ÷ output tokens      ITL = each gap between tokens`,
example:`<b>Worked example</b>Mix at 0.67 req/s total, so A arrives at 0.2/s and prefill occupies the GPU ~17% of the time. A B request decodes 1,000 tokens over ~10 s and sees about 2 A prefills. Its TPOT rises by only 2 × 0.85 s ÷ 1,000 ≈ 1.7 ms, which still meets 50 ms. But 2 of its 1,000 gaps last ~850 ms, and across all B requests that tail decides p99 ITL as A's rate rises.`,
diag:"stall",
predict:["Service time of an A request and a B request; each engine's capacity in req/s on the 30/70 mix with batch 16.","Goodput at M3's knee vs M4's knee. Where does the factor come from?","p99 TTFT for B below the knee: M3 waits for a whole batch, M4 for one slot. Numbers for both.","How many A prefills does a B request see at A rate λ? What happens to its TPOT, and what happens to p99 ITL across all B gaps?","R2 at 2–3× the knee: is 64 waiting small enough that admitted requests still meet TTFT?"],
lab:{name:"S4 · Mixed traffic",why:"Measure goodput from iteration-level scheduling on the traffic that hurts static batching most, and expose the prefill stall.",
spec:[["First pass",FIRST],["Official",OFFICIAL],["Workload","Mix 30% A / 70% B, open-loop Poisson up to 2–3× M4's knee, ≥ 1,000 requests per rate, 30 s warm-up excluded"],["Servers","Both <code>--max-batch-size 16</code>; M4 also <code>--max-num-batched-tokens 8192 --max-waiting 64</code>; M3 from tag m3 on port 8001"],["Command","<code>python benchmarks/s4_mixed_traffic.py --base-url ... --tokenizer models/Llama-3.1-8B --tag m3|m4</code>, then <code>--compare m3,m4</code>"],["Gates","G1a under joins, leaves and slot reuse; G1b; R2 checked by the rates past the knee"]],
metrics:[["Goodput","Requests/s meeting both SLOs"],["p99 TTFT per class","A and B separately"],["p99 ITL for B","Shows the prefill stall directly"],["TPOT for B","Averages the stall away; compare with ITL"],["Rejection rate","503s past the knee"]],
expected:"Several× goodput on high length variance and much lower p99 TTFT. B's ITL spikes when a long prompt joins."},
learn:"Scheduling at every step recovers most of static batching's waste. Prefill and decode now interfere inside one step.",
next:"Memory, not compute, now caps how many requests run (M5).",
check:["Why does FCFS without skipping matter for long prompts?","Why reject at the frontend rather than in the engine?"]});

D({unit:"1a",n:"7",title:"Attention backend and KV layout",status:"decided",
choice:"Three backends on one layout: M5a torch reference (oracle), M5b FlashInfer, M5c vLLM's flash-attn fork. Block size 16; per-layer K and V tensors shaped [blocks, 16, Hkv, Dh] (NHD); ragged per-step metadata.",
why:"Both production kernels can read this layout, and whole blocks can be shipped layer by layer later (D3, N3). Choosing a layout only one kernel reads would mean reworking M5–M6."});

M({unit:"1a",id:"m6",n:6,title:"Paged KV memory and attention kernels",tags:["M5a","M5b","M5c","S5","R3",["memory","m"]],status:"ready",
obj:["Quantify the memory wasted by contiguous slots","Explain block tables and preemption by recompute","Compare a gather-based reference with paged kernels"],
lesson:`<p>Contiguous slots reserve <code>max_model_len</code> tokens of KV for every request, because the engine can't know how long it will run. Most requests are far shorter, so most reserved memory is empty, and that caps concurrency long before compute does.</p>
<p><b>Paged KV</b> (vLLM's idea) splits memory into fixed 16-token blocks handed out as tokens are generated. A per-request <b>block table</b> maps logical positions to physical blocks anywhere in the pool. Waste falls to the half-empty last block of each request. When the pool runs dry, the scheduler <b>preempts</b> the newest request and recomputes it later, which shows up as a sharp latency cliff.</p>
<p>Attention must now read KV through the block table. The torch reference gathers the blocks into a contiguous copy first: correct, but it copies batch × context every step. FlashInfer and flash-attn read pages in place. The layout is the contract that makes them interchangeable.</p>`,
formula:`contiguous slots in budget = budget ÷ (max_model_len × KV/token)
paged blocks                = budget ÷ (16 × KV/token)
blocks per request          = ceil(tokens / 16)
waste per request           ≈ 7.5 tokens (half a block)`,
example:`<b>Worked example</b>8 GiB budget, 8B (128 KB/token), max_model_len 8,192: contiguous fits <b>8</b> slots, and a 608-token request uses 7.4% of its slot. Paged: 4,096 blocks of 2 MB, 38 per request → about <b>107</b> concurrent requests, waste ~1.2%.`,
diag:"paged",
predict:["Part A: how many contiguous slots fit in 8 GiB at 8,192? What fraction of each holds real tokens? How many fit paged?","Block fill: what share of held block slots hold real tokens for ~600-token requests?","How much faster is decode with several× more requests running, given S3's batch curve? Where does it stop helping?","Part B: at which pool size do preemptions start? What does each cost? Why is the cliff sharp?","Part C: bytes the reference copies at 4,096 tokens × batch 32 vs the weights read. How close should FlashInfer and flash-attn get to contiguous?"],
lab:{name:"S5 · KV pressure",why:"Show that memory caps decode throughput, and that a paged kernel removes the cost of reading through a block table.",
spec:[["First pass",FIRST],["Official",OFFICIAL],["Workload","256 workload-B requests (100 in, outputs 16–1,000), in-process, max_model_len 8,192"],["Parts","A: 8 GiB, contiguous vs paged. B: pool at 1, ½, ¼, ⅛, 1/16. C: decode step at batch 1/8/32 × ctx 512/4,096 for slots, reference, FlashInfer, flash-attn"],["Command","<code>python benchmarks/s5_kv_pressure.py --model-dir models/Llama-3.1-8B --kv-gb 8 --gpu a100-80 --backends torch_ref,flashinfer,flash_attn</code>"],["Gates","G0/G1a on paged KV incl. preempted requests; R3 after every run; each kernel vs reference within G1b"]],
metrics:[["Concurrent requests","Average and max running"],["Block fill","Share of held slots with real tokens"],["Tokens/s","Paged vs contiguous"],["Preemptions + recomputed tokens","Where the cliff starts"],["Decode step time per backend","Reference vs FlashInfer vs flash-attn vs slots"]],
expected:"Paged waste under ~5%, several× concurrency. Preemption shows as TTFT/ITL cliffs. The reference falls behind with context; FlashInfer within ~10% of contiguous; the two kernels within 10–20% of each other."},
learn:"KV memory caps concurrency. Paging fixes the allocation, and a paged kernel fixes the read.",
next:"Many requests share the same prompt start. Store it once (M6).",
check:["Why is preemption by recompute cheaper to build than swapping?","What does R3 catch that output checks would miss?"]});

D({unit:"1a",n:"8",title:"Block-hash spec",status:"decided",
choice:"SHA-256 chain over full 16-token blocks (token ids as uint32 little-endian). The root is salted with model@revision, KV dtype, layout, block size and an optional per-request cache_salt. The last prompt token is always recomputed.",
why:"A specified hash lets a Go or Rust router compute identical keys (D2), and salting from day one turns tenant isolation (P2) into a policy change, not a migration."});

M({unit:"1a",id:"m7",n:7,title:"Prefix caching",tags:["M6","S6","R3",["compute","c"]],status:"ready",
obj:["Explain content-addressed KV blocks","Predict TTFT as a function of the cached fraction","Estimate the hit rate of an agent workload"],
lesson:`<p>Many requests start the same way: a system prompt, tool schemas, a document being questioned, an agent's whole conversation so far. <b>Prefix caching</b> keeps finished KV blocks after a request ends and reuses them when a new prompt begins with the same tokens.</p>
<p>Blocks are found by <b>hash</b>. Each full block's hash covers its parent's hash and its own 16 token ids, so equal hashes mean identical prefixes. Blocks carry <b>reference counts</b>: shared blocks are never freed while in use, and unreferenced ones stay cached until an <b>LRU</b> policy evicts them for space.</p>
<p>A hit skips prefill for the cached tokens, so TTFT should fall in proportion to the cached fraction. It can't reach zero: the new tokens still attend over the whole context, and each step has fixed overhead. Agents and RAG gain the most, which is why hit rate becomes a first-class metric.</p>`,
formula:`h_i = SHA-256(h_(i−1) ‖ token_ids[16·i : 16·(i+1)])
TTFT(f) ≈ (1 − f) × prefill + attention over cached context + fixed overhead
hit rate = cached prompt tokens ÷ prompt tokens`,
example:`<b>Worked example</b>An 8K prompt with f = 0.9 cached: the 8B prefills ~800 tokens instead of 8,192. Ideal speed-up is 10×; expect less because attention for those 800 tokens still reads 8K of context. Hashing 512 blocks costs tens of microseconds against a 0.85 s prefill.`,
diag:"hashchain",
predict:["Part A: with a fraction f cached, is TTFT ∝ (1 − f), or does it flatten at high f? What is the floor?","At f = 0, what does hashing cost against a 0.85 s prefill?","Part B: what share of a turn's prompt is cached on a first turn vs later turns? Predict the hit rate (prefix ~3.5K, 6 turns, ~300 new tokens per turn).","Part B TTFT with caching off vs on, against the 1 s SLO.","What breaks this: eviction from a small pool, and preemption. Which state would D4's tiering keep?"],
lab:{name:"S6 · Shared prefixes",why:"Measure how much prefill caching removes on agent and many-questions-one-document traffic, and what it costs when nothing is shared.",
spec:[["First pass",FIRST],["Official",OFFICIAL],["Part A","20 requests of 8,000 tokens sharing the first f × 8,000; f = 0, 0.25, 0.5, 0.75, 0.9; 8 output tokens; first request cold"],["Part B","24 workload-C sessions, ~3.5K shared prefix, growing history, turns interleaved round-robin"],["Command","<code>python benchmarks/s6_shared_prefixes.py --model-dir models/Llama-3.1-8B --gpu a100-80</code>"],["Gates","fp32 tokens identical with hits, sharing, eviction and resume; hash spec byte for byte; R3 with refcounts"]],
metrics:[["TTFT vs f","Against the ideal (1 − f) line"],["Hit rate","Cached ÷ prompt tokens"],["Prefilled tokens","Total off vs on"],["p99 TTFT per turn","Against the 1 s SLO"]],
expected:"TTFT drops roughly in proportion to the cached fraction, flattening at high f; near-zero cost on unique prompts."},
learn:"Agents and RAG are cache-friendly by construction; the hit rate decides their TTFT.",
next:"Ship v1, then attack the prefill stall that S4 exposed (M7).",
check:["Why is the last prompt token always recomputed?","Why must the hash be specified rather than Python's hash()?"]});

M({unit:"1a",id:"m8",n:8,title:"Ship v1",tags:["R4","v1"],status:"planned",measured:"pending. The R4 soak script and the scripted EC2 session in scripts/aws/ still need writing.",
obj:["List the release gate","Plan the one EC2 session that produces all v1 numbers","Know what goes in the write-up"],
lesson:`<p>v1 is done when every gate and robustness test passes, the <b>R4 soak</b> holds steady for 1–2 hours, and S0–S6 are measured on A100/H100 in one scripted session.</p>
<p><b>The EC2 session.</b> One g6.xlarge run (~2–3 hours) tests all CUDA-only code at once: M5b and M5c against the reference under G1b, then S5 part C. Then one A100/H100 session runs S0–S6 from their milestone tags and records the G2 baseline. Every script is ready before the instance starts.</p>
<p><b>Deliverables.</b> Tag <code>v1</code>, a README with the S0–S6 charts, and one public write-up. Applications for inference-platform roles open here.</p>`,
formula:`v1 gate = G0 ∧ G1a ∧ G1b ∧ G2 ∧ R1 ∧ R2 ∧ R3 ∧ R4 ∧ S0–S6 measured`,
example:`<b>R4 pass criteria</b>Steady Mix + C load for 1–2 h: no memory growth, no block leaks, p99 TTFT and ITL drift under 10% between the first and last 10 minutes.`,
predict:["Which study is most likely to need a rerun, and why?","What would memory growth during R4 point to?"],
lab:{name:"R4 · Soak + v1 release",why:"Prove the engine holds up for hours, not just for a benchmark run.",
spec:[["Kernel check","g6.xlarge: tests/test_kernels.py, M5b/M5c vs reference under G1b"],["Official runs",OFFICIAL+"; S0–S6 from tags m0–m6"],["Soak","Mix + workload C, 1–2 h. Script not written yet"],["Session script","scripts/aws/ has only a README; the end-to-end session (setup, download, runs, copy out, stop) is still to write"],["Gates","G0–G2, R1–R3 green"]],
metrics:[["Process memory over time","Flat"],["Free + used blocks","Equals total throughout"],["p99 drift","Under 10%"]],
expected:"All green; README charts and write-up published; tag v1."},
learn:"A release is a set of passed gates plus measured, explained numbers.",
next:"Unit 1b: chunked prefill, CUDA graphs, quantization, speculative decoding.",
check:["Which tests rerun at every later milestone?"]});
