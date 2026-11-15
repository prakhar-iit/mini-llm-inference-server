D({unit:"0",n:"25",title:"Measurement hardware per experiment",status:"decided",
choice:"One measurement GPU per chain of studies. L4 (g6.xlarge, $0.80/h) for development, tests and smoke runs (no study numbers); g6.12xlarge (4× L4) for multi-GPU correctness; 1× H100 (p5.4xlarge, $6.88/h) for Phase 1 official S0–S11; 8× H100 (p5.48xlarge) for Phases 2–4; p5en (8× H200) for Kimi K2. When the GPU changes, remeasure the baseline and report MBU/MFU next to the milliseconds.",
why:"Bandwidth sets decode time and compute sets prefill time, so a GPU change moves every number. Keeping one chip, the H100, from Phase 1 to Phase 4 means each gain comes from the code. The single-H100 p5.4xlarge is the cheapest official GPU and has FP8 for S9a."});

D({unit:"0",n:"20 (partial), 21",title:"Budget and AWS access before the first instance",status:"open",
choice:"Set a dollar cap per phase with an AWS Budgets alarm for each. Check the G-instance vCPU quota (g6 is needed first) and confirm the AWS account is personal. P-instance quota, Capacity Blocks and EFA wait until Phase 3.",
why:"Every lab runs on EC2, so an unbounded bill is the main project risk. A quota of 0 blocks the first lab."});

const PLAN8 = "p5.4xlarge (1× H100 80 GB)";

/* ---------------- Unit 1b ---------------- */
M({unit:"1b",id:"m9",n:9,title:"Chunked prefill",tags:["M7","S7",["compute","c"]],status:"planned",
obj:["Explain how a token budget bounds step time","Predict the ITL/TTFT trade-off per chunk size","See why this is the fair baseline for P/D disaggregation"],
lesson:`<p>S4 showed that a long prefill stalls every decoding user for the whole prefill. <b>Chunked prefill</b> (Sarathi-Serve) cuts a long prompt into chunks of at most the <b>token budget</b> and mixes each chunk into a step with the running decodes. No step is ever longer than one chunk plus a decode.</p>
<p>The cost moves to the long prompt: it now needs several steps to finish prefill, so its TTFT rises a little, and each chunk re-reads the KV of earlier chunks. Chunk size is the dial. Smaller chunks give smoother ITL for everyone else; larger ones give faster TTFT for the long prompt.</p>
<p>This milestone is required before D3. Comparing disaggregation against plain M4 would compare it against a strawman.</p>`,
formula:`step time ≈ chunk_tokens × 2 × params ÷ (MFU × peak) + decode step
p99 ITL   ≈ one chunk's step, not one whole prefill`,
example:`<b>Worked example</b>8B on an A100 at 50% MFU: a 2,048-token chunk takes ~0.21 s, 512 tokens ~53 ms, 256 tokens ~26 ms. Only chunks of ~256–512 keep a decoding user near the 50 ms TPOT target while an 8K prompt arrives; a 2,048 chunk still breaks it, but 4× less than a whole prefill.`,
diag:"chunked",
predict:["p99 ITL for decode users while 8K prompts arrive, at chunk 256 / 512 / 2,048.","How much does the long prompt's TTFT rise at each chunk size?","Which chunk size maximizes goodput on the Mix workload?"],
lab:{name:"S7 · Prefill/decode interference",why:"Measure how much of S4's ITL spike chunking removes, and what it costs in TTFT.",
spec:[["Instance",PLAN8],["Model","Llama-3.1-8B, bf16"],["Workload","Decode users on B while 8K A prompts arrive; Mix under Poisson load; chunk 256 / 512 / 2,048"],["Baseline","Tag m6 (no chunking)"],["Gates","G0, G1a/G1b with mixed steps"]],
metrics:[["p99 ITL of B","Should fall to about one chunk's time"],["TTFT of A","Rises slightly with smaller chunks"],["Goodput at the knee","Per chunk size"]],
expected:"p99 ITL falls from ~whole-prefill time to ~one chunk's time; TTFT rises slightly; chunk size trades one for the other."},
learn:"Colocated serving can bound interference but not remove it. That remaining trade-off is what disaggregation (D3) removes.",
next:"Look at the CPU side of a decode step (M8).",
check:["Why does a smaller chunk raise the long prompt's TTFT?","Why is M7 required before D3?"]});

M({unit:"1b",id:"m10",n:10,title:"CUDA graphs and launch overhead",tags:["M8","S8"],status:"planned",
obj:["Spot a CPU-bound decode step in a profile","Predict when CUDA graphs help","Read a Nsight Systems timeline"],
lesson:`<p>Each decode step launches hundreds of small GPU kernels from Python. Every launch costs a few microseconds of CPU time. When the GPU work per step is short, the GPU finishes each kernel and waits for the next launch: the step is bound by the CPU, not the GPU.</p>
<p>A <b>CUDA graph</b> records the whole sequence of kernels once and replays it with a single launch. It needs fixed shapes, so engines capture one graph per batch size and pad to the nearest.</p>
<p>The gain depends on the ratio of launch overhead to GPU time. Small models at small batch on fast GPUs gain most; a big model at big batch barely notices.</p>`,
formula:`step ≈ max(GPU work, kernels × launch cost) without graphs
launch overhead ≈ layers × kernels/layer × ~5–10 µs`,
example:`<b>Worked example</b>Llama-3.2-1B: 16 layers × ~12 kernels × ~7 µs ≈ 1.3 ms of launches, against a ~0.7 ms decode floor on the H100. Launches dominate, so graphs could cut the step by more than half. The 8B's ~4.8 ms floor makes the same overhead a much smaller share.`,
predict:["Decode step time for the 1B at batch 1 with and without graphs on the H100.","At what batch size does the 8B stop gaining?","What will the gaps between kernels look like in Nsight Systems before and after?"],
lab:{name:"S8 · CPU launch overhead",why:"Learn to see when the GPU waits on Python, and how much one launch per step recovers.",
spec:[["Instance",PLAN8],["Models","Llama-3.2-1B and Llama-3.1-8B, bf16"],["Workload","Decode step time at batch 1–32"],["Profile","PyTorch profiler and Nsight Systems"],["Gates","G0 identical with graphs on"]],
metrics:[["Decode step time","Graphs on vs off, per batch"],["GPU idle gaps","Between kernels in the timeline"],["Speed-up by model","1B vs 8B"]],
expected:"Large gain for the 1B at small batch; negligible for the 8B at large batch."},
learn:"Some latency lives on the CPU. A profile shows it as gaps between kernels.",
next:"Reduce the bytes each decode step reads (M9).",
check:["Why do graphs need padding to fixed batch sizes?"]});

M({unit:"1b",id:"m11",n:11,title:"Quantization",tags:["M9","S9a","S9b","G2",["memory","m"]],status:"planned",
obj:["Link bytes read to decode time","Separate weight quantization from KV quantization","Use G2 as a quality gate"],
lesson:`<p>Decode is bound by bytes read. Store weights in 8 bits instead of 16 and each step reads half as much, so ITL at small batch should improve by about the byte ratio. Prefill gains only where the GPU has FP8 tensor cores (L4, H100; not A100); otherwise the weights are expanded back to bf16 before the math.</p>
<p>Quantizing the <b>KV cache</b> to fp8 is a separate lever: it halves the bytes per token, so twice as many tokens fit in the same memory. That raises concurrency (S5) and cuts the KV read at long context.</p>
<p>Both lose some precision. <b>G2 is the hard gate</b>: perplexity and GSM8K must stay within noise of the bf16 baseline, or the speed-up doesn't count.</p>`,
formula:`decode floor ∝ weight bytes:  8B bf16 16 GB → fp8 8 GB
KV/token: 8B bf16 128 KB → fp8 64 KB  (2× tokens per GiB)`,
example:`<b>Worked example</b>8B on an A100 80 GB: the decode floor drops from 7.9 ms to ~3.9 ms with 8-bit weights. In an 8 GiB KV budget, fp8 KV holds ~131K tokens instead of ~65K.`,
predict:["S9a: ITL at batch 1 with 8-bit weights, and the change in G2 scores.","Will prefill get faster on the H100? Would it on an A100, which has no FP8 tensor cores?","S9b: max concurrency and long-context ITL with fp8 KV."],
lab:{name:"S9a/S9b · Quantization",why:"Show that bytes moved are the decode cost, and that precision is a capacity lever, without losing quality.",
spec:[["Instance",PLAN8+"; L4 for an FP8 tensor-core comparison"],["Model","Llama-3.1-8B: bf16 vs weight-only 8-bit/FP8 (borrowed kernel), bf16 KV vs fp8 KV"],["Workload","S9a: batch 1 ITL + G2. S9b: S5's sweep and long-context ITL"],["Gates","G2 within noise of the HF bf16 baseline (hard gate)"]],
metrics:[["ITL at batch 1","Against the byte ratio"],["Perplexity, GSM8K","G2 delta"],["Max concurrency","fp8 vs bf16 KV"],["Prefill time","A100 vs H100"]],
expected:"ITL improves by about the byte ratio at small batch with small quality loss. fp8 KV gives ~2× concurrent tokens."},
learn:"Decode cost is bytes. Precision trades a little quality for speed and capacity, and only a quality gate keeps that honest.",
next:"Spend idle compute guessing tokens ahead (M10).",
check:["Why doesn't the A100 speed up prefill with FP8 weights?"]});

M({unit:"1b",id:"m12",n:12,title:"Speculative decoding",tags:["M10","S10","G0",["compute","c"]],status:"planned",
obj:["Explain draft-then-verify","Predict speed-up from acceptance rate","See why the gain shrinks with batch size"],
lesson:`<p>Decode at low batch leaves compute idle (S1). <b>Speculative decoding</b> spends it: a cheap drafter guesses the next <i>k</i> tokens, and the target model checks all of them in one forward pass, which costs about the same as one normal decode step because it is memory-bound. It keeps the longest run of correct guesses plus one token of its own.</p>
<p>Drafters here are <b>n-gram prompt lookup</b> (copy from the prompt; great for summarization) and <b>Llama-3.2-1B</b> drafting for the 8B. With greedy decoding the output must be <i>exactly</i> the same as without speculation: any difference under G0 is a bug.</p>
<p>As batch size grows, the spare compute disappears, verification stops being free, and the speed-up shrinks and can turn negative.</p>`,
formula:`expected tokens per target pass = (1 − α^(k+1)) / (1 − α)
speed-up ≈ that ÷ (1 + k × draft cost / target cost)`,
example:`<b>Worked example</b>Acceptance α = 0.8, k = 4: 3.36 tokens per target pass. With a 1B draft at ~15% of the 8B's step cost, speed-up ≈ 3.36 ÷ 1.6 ≈ 2.1×. On creative text α may be 0.5: 1.94 tokens per pass, ≈ 1.2×.`,
diag:"spec",
predict:["Acceptance rate on A (copies a lot) vs B (creative), for n-gram and 1B drafts.","Speed-up at batch 1 on A and on B.","At what batch size does speculation stop paying?"],
lab:{name:"S10 · Acceptance vs speed-up",why:"Measure when idle compute is worth spending on guesses, and confirm it is lossless.",
spec:[["Instance",PLAN8],["Models","Target Llama-3.1-8B; drafts n-gram and Llama-3.2-1B"],["Workload","A and B, batch 1 → 32, k swept"],["Gates","G0: output identical to non-speculative greedy"]],
metrics:[["Acceptance rate α","By workload and drafter"],["Tokens per target pass",""],["Speed-up vs batch","Where it crosses 1"]],
expected:"1.5–2.5× at batch 1 on A; little on B. Gain shrinks and can go negative as batch grows."},
learn:"Speculation spends idle compute. It is a low-load optimization.",
next:"Compare the whole engine with vLLM and ship v1.1.",
check:["Why must speculative output match greedy output exactly?"]});

M({unit:"1b",id:"m13",n:13,title:"Own engine vs vLLM, ship v1.1",tags:["S11","R4","v1.1"],status:"planned",
obj:["Run a fair comparison","Explain a performance gap from profiles"],
lesson:`<p>The last study puts your engine and vLLM side by side on A, B, C and Mix with the same settings and the same benchmark client, so the metric definitions can't differ.</p>
<p>The goal is an honest <b>gap analysis</b>, not a close number. Expect gaps from kernels (fused ops, attention variants), scheduler details, and CPU overhead. Each one should be named and sized from a profile.</p>
<p>Then the v1.1 release gate: all gates green, the R4 soak rerun, and S0–S10 rerun on the H100.</p>`,
formula:`gap = vLLM goodput ÷ own goodput, per workload, explained term by term`,
predict:["On which workload will the gap be largest, and why?","Which single cause do you expect to dominate the gap?"],
lab:{name:"S11 · vs vLLM + v1.1",why:"Ground your engine against the production reference, then freeze v1.1.",
spec:[["Instance",PLAN8],["Model","Llama-3.1-8B"],["Workload","A, B, C, Mix; same client against both engines (decision 5)"],["Release","R4 soak rerun; S0–S10 rerun; tag v1.1"]],
metrics:[["Goodput per workload","Own vs vLLM"],["p99 TTFT / ITL",""],["Profile breakdown","Kernels, scheduler, CPU"]],
expected:"vLLM ahead, with every gap attributed to a cause."},
learn:"An explained gap is worth more than a close number.",
next:"Unit 2: take the engine across GPUs.",
check:["Why does using the same client for both engines matter?"]});

/* ---------------- Unit 2 ---------------- */
D({unit:"2",n:"16",title:"Multi-GPU dev hardware",status:"decided",
choice:"g6.12xlarge (4× L4, PCIe) for D1–D3 correctness; performance numbers only on the p5.48xlarge (8× H100, NVLink). Settled by decision 25.",why:"Develop multi-GPU code cheaply; PCIe without NVLink gives the wrong all-reduce cost, so it never produces performance numbers."});

M({unit:"2",id:"m14",n:14,title:"Tensor parallelism",tags:["D1","G1"],status:"planned",
obj:["Split attention and MLP across GPUs","Count the all-reduces per layer","Explain why TP buys latency, not efficiency"],
lesson:`<p><b>Tensor parallelism</b> (Megatron-LM) splits each layer's weight matrices across GPUs. The first matrix of a block is split by columns, so each GPU computes part of the output (half the attention heads, half the MLP width). The second is split by rows, so each GPU produces a partial sum, and an <b>all-reduce</b> adds them. That is two all-reduces per layer, on the critical path of every step.</p>
<p>Each GPU now reads 1/TP of the weights, so the decode floor drops, but the all-reduces add fixed latency and per-GPU throughput falls. Every run must match TP=1 within tolerance (G1).</p>`,
formula:`per-GPU weight read = weights ÷ TP
all-reduces per step = 2 × layers
all-reduce bytes ≈ 2 × hidden × batch × 2 bytes × (TP−1)/TP`,
example:`<b>Worked example</b>8B at TP=4 on NVLink: each GPU reads 4 GB, so the floor falls from 4.8 to ~1.2 ms on the H100, but 64 all-reduces per step at ~10–20 µs each add ~1 ms, nearly as much as it saved. On PCIe L4s the all-reduce share is far larger.`,
diag:"tp",
predict:["Decode latency at TP=1/2/4 on the p5.48xlarge.","Share of step time in all-reduce at TP=4, NVLink vs PCIe.","Per-GPU throughput at TP=4 relative to TP=1."],
lab:{name:"D1 · TP scaling",why:"Learn what splitting a model buys and costs before using it at scale.",
spec:[["Correctness","g6.12xlarge (4× L4), NCCL"],["Performance","p5.48xlarge (8× H100, NVLink)"],["Model","Llama-3.1-8B"],["Gates","G1b vs TP=1"]],
metrics:[["Latency vs TP",""],["Per-GPU throughput vs TP",""],["All-reduce share of step","From Nsight Systems"]],
expected:"Latency falls with TP; per-GPU efficiency falls too; all-reduce share grows with TP and is much larger without NVLink."},
learn:"TP buys latency with interconnect bandwidth. It belongs where bandwidth is high: inside a node.",
next:"Optionally, pipeline stages; then replicas and routing.",
check:["Why two all-reduces per layer rather than one?"]});

D({unit:"2",n:"13",title:"Pipeline-parallelism gap",status:"open",
choice:"Options: build D1b (2-stage PP) in the mini server, or run N2's TP+PP layout on vLLM as a measurement only.",why:"N2 compares TP=16 with TP=8 + PP=2; PP must exist somewhere for that study."});

M({unit:"2",id:"m15",n:15,title:"Pipeline parallelism (optional)",tags:["D1b"],status:"optional",
obj:["Split layers into stages","Predict the bubble fraction"],
lesson:`<p><b>Pipeline parallelism</b> gives each GPU a contiguous slice of layers. Activations pass from stage to stage, so communication is small and point-to-point, which suits slow links. The cost is the <b>bubble</b>: stages sit idle while the pipeline fills and drains. Splitting a batch into micro-batches keeps more stages busy.</p>`,
formula:`bubble fraction ≈ (stages − 1) ÷ (micro-batches + stages − 1)`,
example:`<b>Worked example</b>2 stages with 4 micro-batches: bubble ≈ 1/5 = 20%. With 8 micro-batches: ~11%.`,
predict:["Bubble fraction at 2, 4 and 8 micro-batches.","When would PP beat TP across a slow link?"],
lab:{name:"D1b · Bubble vs micro-batches",why:"Set up N2's comparison of TP vs TP+PP across nodes.",
spec:[["Instance","g6.12xlarge, then p5.48xlarge"],["Model","Llama-3.1-8B, 2 stages"],["Gates","G1b vs single GPU"]],
metrics:[["Bubble fraction","Measured vs formula"],["Throughput vs micro-batches",""]],
expected:"Bubble shrinks as micro-batches grow, roughly following the formula."},
learn:"PP trades idle time for cheap communication.",next:"Replicas and a router.",check:["Why does PP suit slow links better than TP?"]});

D({unit:"2",n:"10, 11",title:"Systems language and router state",status:"open",
choice:"Language: Go (llm-d, Kubernetes) or Rust (SGLang router, Dynamo core). Router state: recommendation is replicas publish KV block events (exact, and it becomes P1's global index unchanged), and the router tokenizes so hashes match decision 8.",why:"The router is the first component outside Python."});

M({unit:"2",id:"m16",n:16,title:"Replicas and cache-aware routing",tags:["D2","workload C"],status:"planned",
obj:["Compare routing policies","Explain why cache locality and load balance conflict","Use KV events as router state"],
lesson:`<p>With several engine replicas, a router decides where each request goes. <b>Round-robin</b> spreads load evenly but scatters a session's turns across replicas, so each turn misses the prefix cache. <b>Least-loaded</b> balances queues better. <b>Prefix- or session-aware</b> routing sends a request to the replica that already holds its prefix, which raises the hit rate but can pile load onto one replica.</p>
<p>The router knows what each replica holds through <b>KV block events</b> (block stored, block evicted) keyed by the hashes from decision 8. The best policy scores replicas on both cached prefix length and current load.</p>`,
formula:`score(replica) = cached_prefix_tokens − λ × queue_depth
TTFT saved per hit ≈ cached tokens × prefill time per token`,
example:`<b>Worked example</b>Workload C turn 4: ~4.4K tokens of history are cached on the replica that served turn 3. Routing there prefills ~300 new tokens; round-robin to another replica prefills all ~4.7K (~0.5 s on an A100).`,
predict:["Hit rate per policy on workload C at moderate load.","p99 TTFT per policy as load rises.","At what load does pure prefix-affinity lose to least-loaded?"],
lab:{name:"D2 · Routing policies",why:"Show cache-aware routing as the agent-serving differentiator, with numbers.",
spec:[["Instance","g6.12xlarge (4 replicas of the 8B) for logic; p5.48xlarge (8B replicas) for numbers"],["Workload","Workload C at rising load"],["Policies","Round-robin, least-loaded, prefix/session-aware"],["Tracing","OpenTelemetry added here (decision 6)"]],
metrics:[["Prefix hit rate","Per policy"],["p99 TTFT",""],["Goodput",""],["Load imbalance","Max ÷ mean queue depth"]],
expected:"Cache-aware routing wins on hit rate and TTFT until load imbalance takes over; a combined score wins overall."},
learn:"Routing decides the cache hit rate; the router is where agent serving is won.",
next:"Separate prefill and decode onto different GPUs (D3).",
check:["Why tokenize in the router?"]});

D({unit:"2",n:"14",title:"KV transfer mechanism",status:"open",
choice:"Intra-node: CUDA IPC or NCCL send/recv. Inter-node: NCCL over EFA, NIXL, UCX, or gRPC via CPU. Recommendation: NCCL send/recv for D3, then NIXL for N3 and P1 to match Dynamo and llm-d.",why:"D3 needs a way to move KV blocks between GPUs."});

M({unit:"2",id:"m17",n:17,title:"Prefill/decode disaggregation",tags:["D3",["compute","c"],["memory","m"]],status:"planned",
obj:["Explain why separating phases removes interference","Predict KV transfer cost","Know when transfer eats the gain"],
lesson:`<p>Even with chunked prefill, prefill and decode share each step, so one always waits on the other. <b>Disaggregation</b> (DistServe, Splitwise) runs them on different GPUs: prefill workers batch prompts for compute efficiency, decode workers batch many users for bandwidth efficiency, and the KV cache moves between them.</p>
<p>It pays when transfer time is small next to the interference it removes. Shipping whole blocks layer by layer lets transfer overlap the rest of prefill (decision 7's layout). The baseline is colocated serving <i>with</i> chunked prefill (M7).</p>`,
formula:`transfer time = prompt tokens × KV/token ÷ link bandwidth
8K prompt, 8B: 8,192 × 128 KB = 1 GiB`,
example:`<b>Worked example</b>1 GiB of KV over NVLink (~300 GB/s effective) ≈ 3.5 ms; over PCIe on L4s (~25 GB/s) ≈ 43 ms. Against a 0.26 s prefill on the H100, NVLink's cost is negligible and PCIe's is a sixth of it; for a 500-token prompt the fixed setup cost matters more.`,
diag:"pd",
predict:["SLO attainment colocated (with M7) vs disaggregated on Mix.","KV transfer time for 8K prompts on NVLink and on PCIe.","Prompt length below which disaggregation stops paying."],
lab:{name:"D3 · P/D vs colocated",why:"Find when separating the phases beats a well-tuned colocated engine.",
spec:[["Instance","g6.12xlarge for correctness; p5.48xlarge for numbers"],["Model","Llama-3.1-8B"],["Workload","Mix and A at rising load"],["Baseline","Colocated with chunked prefill (M7)"],["Gates","G1b: disaggregated output matches colocated"]],
metrics:[["SLO attainment / goodput",""],["KV transfer time","Per prompt length"],["Prefill:decode GPU ratio","Best split"]],
expected:"Disaggregation wins on ITL-sensitive mixes with long prompts; for short prompts transfer overhead can flip it."},
learn:"Separating the phases removes interference at the price of moving bytes.",
next:"Where does idle KV go between agent turns? (D4).",
check:["Why compare against M7 rather than M4?"]});

D({unit:"2",n:"12, 19",title:"Session API and tier eviction",status:"open",
choice:"Session API: add session_id plus tool-call pause/resume hints (and optional expected tool duration) via header or extra_body. Eviction: score = recompute cost (∝ prefix length) × reuse probability; write back to DRAM on eviction from HBM.",why:"D4 can't keep a session's KV warm without knowing the session is paused."});

M({unit:"2",id:"m18",n:18,title:"KV tiering on one node",tags:["D4",["memory","m"]],status:"planned",
obj:["Place KV across HBM, DRAM and NVMe","Find the fetch-vs-recompute crossover","Design eviction for paused agent sessions"],
lesson:`<p>Agents pause for tool calls lasting seconds to minutes, and their KV sits idle in scarce GPU memory. <b>KV tiering</b> moves idle KV down to CPU DRAM or local NVMe and brings it back on resume. Each tier holds more and moves slower.</p>
<p>Fetching beats recomputing when the bytes divided by the tier's bandwidth take less time than prefilling those tokens again. For long prefixes, prefill grows faster than linearly with length while bytes grow linearly, so fetching wins. For short prefixes, two fixed costs decide instead: each fetch pays a setup latency, and a prefill can never cost less than one full read of the weights (~5 ms for the 8B on the H100). Those set the crossover. This ties directly to patent US12547545B2 (HA via in-memory cache).</p>`,
formula:`fetch time     = setup latency + tokens × KV/token ÷ tier bandwidth
recompute time ≈ max(weights ÷ bandwidth, tokens × 2 × params ÷ (MFU × peak))
crossover: fetch time = recompute time`,
example:`<b>Worked example</b>4K-token agent prefix on the 8B: 512 MB of KV. From DRAM at ~25 GB/s ≈ 20 ms; from NVMe at ~3 GB/s ≈ 170 ms; recompute ≈ 0.42 s on an A100. Both tiers win at 4K; at short prefixes NVMe can lose.`,
diag:"tiers",
predict:["Crossover prompt length for DRAM and for NVMe.","Resume TTFT for a paused session: recompute vs DRAM vs NVMe.","How much more concurrency does offloading paused sessions free?"],
lab:{name:"D4 · Fetch vs recompute",why:"Measure the tier economics that Phase 3 takes cluster-wide.",
spec:[["Instance","p5.48xlarge (NVMe instance storage)"],["Model","Llama-3.1-8B"],["Workload","Workload C with tool pauses; prompt length sweep"],["Gates","Restored KV gives tokens identical to never-evicted (fp32)"]],
metrics:[["Resume TTFT per tier",""],["Crossover length",""],["Running sessions at fixed HBM",""]],
expected:"DRAM fetch beats recompute at almost every length; NVMe wins past a crossover of a few thousand tokens."},
learn:"Each memory tier has a length past which moving bytes beats doing math.",
next:"What happens when a replica dies mid-stream? (D5).",
check:["Why does the crossover move with prompt length?"]});

D({unit:"2",n:"15",title:"Failure model",status:"open",
choice:"Failures: process kill, GPU OOM, node loss. Recommendation: router replays prompt + already-emitted tokens to another replica (recompute); measure cold start by weight source (S3, local NVMe, peer GPU).",why:"D5 needs a defined set of failures and resume semantics."});

M({unit:"2",id:"m19",n:19,title:"Reliability and elasticity",tags:["D5","R4"],status:"planned",
obj:["Resume a stream after a replica fails","Measure cold-start time by weight source","Run failure injection during a soak"],
lesson:`<p>Replicas will die mid-generation. The router can resume a stream by sending the prompt plus the tokens already emitted to another replica, which recomputes and continues; the client sees a pause, not an error. With prefix caching or tiering, that recompute can become a fetch.</p>
<p>Scaling up needs new replicas fast. <b>Cold start</b> is dominated by loading weights: from S3, from local NVMe, or from a peer GPU over the interconnect.</p>`,
formula:`cold start ≈ weight bytes ÷ source bandwidth + init + warm-up
recovery TTFT ≈ detection + recompute(prompt + emitted)`,
example:`<b>Worked example</b>8B weights, 16 GB: from S3 at ~1 GB/s ≈ 16 s; NVMe at ~5 GB/s ≈ 3 s; a peer GPU over NVLink in well under a second.`,
predict:["Recovery time after a process kill mid-stream.","Requests lost with and without replay.","Cold start for each weight source."],
lab:{name:"D5 · Failure injection",why:"Apply distributed-systems fundamentals to serving and measure them.",
spec:[["Instance","p5.48xlarge"],["Workload","R4 soak (Mix + C) with injected kills, OOM, node loss"],["Gates","Resumed output matches an uninterrupted run (greedy)"]],
metrics:[["Recovery time",""],["Requests lost",""],["Cold-start time per source",""]],
expected:"Replay loses no requests at the cost of a recompute pause; peer-GPU loading is fastest."},
learn:"Reliability is a latency budget for failure cases.",
next:"Unit 3: cross the node boundary.",
check:["What does a resumed stream need from the router?"]});
