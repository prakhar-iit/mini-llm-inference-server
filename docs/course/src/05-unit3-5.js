/* ---------------- Unit 3 ---------------- */
D({unit:"3",n:"20 (full)",title:"AWS access for P-instances",status:"open",
choice:"Before booking: P-instance vCPU quota (often 0 by default), Capacity Blocks for 2 nodes, a cluster placement group in one AZ, an EFA-enabled AMI, and a Budgets alarm.",why:"Multi-node work fails silently without the second node actually being available."});

M({unit:"3",id:"m20",n:20,title:"Interconnects and multi-node mechanics",tags:["N1"],status:"planned",
obj:["Compare NVLink, PCIe and EFA bandwidth","Launch a 2-node job","Watch cross-node TP crawl"],
lesson:`<p>Inside a node, GPUs talk over NVLink at ~900 GB/s each. Between nodes, AWS EFA gives ~50 GB/s per node on p4d and ~400 GB/s on p5: an order of magnitude less. Design rules that hold inside a box break across boxes.</p>
<p>N1 learns the mechanics for a few dollars: two cheap GPU instances over plain TCP, a launcher (Ray or Kubernetes), NCCL all-reduce tests, and a deliberately bad cross-node TP run to see what the bandwidth drop does.</p>`,
formula:`all-reduce time ≈ 2 × bytes × (n−1)/n ÷ link bandwidth + latency × steps`,
example:`<b>Rule of thumb</b>Tensor parallelism inside a node; pipeline parallelism, replicas or P/D disaggregation across nodes.`,
predict:["NCCL all-reduce bandwidth over TCP between two cheap instances.","Slow-down of cross-node TP=2 vs single-node TP=2."],
lab:{name:"N1 · Mechanics on cheap hardware",why:"Learn launch, NCCL tuning and failure modes before paying p5 rates.",
spec:[["Instance","2× inexpensive GPU instances (e.g. g6), plain TCP"],["Launcher","Ray or Kubernetes"],["Tests","nccl-tests all-reduce; forced cross-node TP"]],
metrics:[["All-reduce bus bandwidth",""],["Cross-node TP step time",""]],
expected:"Cross-node TP is many times slower; the all-reduce dominates the step."},
learn:"Bandwidth sets the parallelism layout.",next:"The same question on EFA with a 70B model (N2).",check:["Why does TP stop at the node boundary?"]});

M({unit:"3",id:"m21",n:21,title:"Parallelism layout across nodes",tags:["N2","70B"],status:"planned",
obj:["Compare TP=16 with TP=8 + PP=2","Trade pipeline bubbles against all-reduce cost"],
lesson:`<p>A 70B model on two 8-GPU nodes can be split TP=16 (every layer's all-reduce crosses EFA) or TP=8 inside each node with PP=2 across them (only activations cross, once per stage). The second keeps the heavy traffic on NVLink and pays a pipeline bubble instead.</p>`,
formula:`TP=16: 2 × layers all-reduces per step over EFA
TP=8+PP=2: 1 activation transfer per micro-batch per step over EFA`,
example:`<b>Worked example</b>70B has 80 layers: TP=16 puts 160 cross-node all-reduces on every decode step. TP=8 + PP=2 sends one activation tensor (8,192 × batch × 2 bytes) between nodes.`,
predict:["Decode ITL for each layout on p4d and on p5.","Does p5's 8× more EFA bandwidth make TP=16 viable?"],
lab:{name:"N2 · TP vs TP+PP over EFA",why:"Confirm or break the rule of thumb with a real model.",
spec:[["Instance","2× p4d or 2× p5, EFA, placement group"],["Model","Llama-3.3-70B"],["Engine","Own engine with D1b, or vLLM as a measurement (decision 13)"]],
metrics:[["ITL and throughput per layout",""],["Bubble fraction vs all-reduce share",""]],
expected:"TP=8 + PP=2 wins on p4d; the gap narrows on p5."},
learn:"Keep the chatty parallelism on the fast link.",next:"Ship KV between nodes instead (N3).",check:["What crosses the node boundary under PP?"]});

M({unit:"3",id:"m22",n:22,title:"Cross-node P/D disaggregation",tags:["N3",["memory","m"]],status:"planned",
obj:["Predict KV transfer over EFA","Find the fetch-vs-recompute crossover across nodes"],
lesson:`<p>D3 moved KV between GPUs in one box. N3 moves it across EFA: prefill on node A, decode on node B. With fp8 KV the 70B stores 160 KB per token, so an 8K prompt is about 1.3 GB. Moving those bytes is still far cheaper than recomputing a long prompt; for short prompts, fixed transfer overhead can flip the result.</p>`,
formula:`8K × 160 KB ≈ 1.3 GB
p4d EFA ~50 GB/s → ~26 ms     p5 EFA ~400 GB/s → ~3 ms
recompute 8K on 70B → hundreds of ms`,
example:`<b>Worked example</b>bf16 KV doubles the transfer to ~2.6 GB: ~52 ms on p4d, still well under the recompute time for 8K tokens.`,
diag:"pd",
predict:["Transfer time for 1K, 4K, 8K, 32K prompts on p4d and p5.","Prompt length below which recompute wins."],
lab:{name:"N3 · KV over EFA",why:"Measure the crossover that every cluster-caching decision depends on.",
spec:[["Instance","2× p4d or 2× p5, EFA"],["Model","Llama-3.3-70B, fp8 and bf16 KV"],["Transport","NIXL (decision 14)"]],
metrics:[["Transfer time vs length",""],["End-to-end TTFT vs colocated",""],["Crossover length",""]],
expected:"Moving bytes beats recompute for long prompts; a crossover appears at short lengths."},
learn:"Over a fast fabric, KV is cheaper to move than to rebuild.",next:"Route across nodes with the cache in mind (N4).",check:["Why does fp8 KV matter more across nodes?"]});

M({unit:"3",id:"m23",n:23,title:"Cross-node cache-aware routing",tags:["N4"],status:"planned",
obj:["Extend D2's router across nodes","Price a cross-node cache miss"],
lesson:`<p>Replicas on both nodes sit behind the D2 router. Now a cache hit on the other node is reachable but costs an EFA transfer, and a miss costs a full prefill. The router's score gains a distance term.</p>`,
formula:`cost(replica) = recompute(uncached) + transfer(remote cached) + queue wait`,
predict:["Hit rate and p99 TTFT for local-only vs cross-node-aware routing on workload C."],
lab:{name:"N4 · Routing across nodes",why:"Quantify routing to where the prefix lives against load balance across nodes.",
spec:[["Instance","2× p4d or p5"],["Workload","Workload C at rising load"],["Router","D2 router with KV events from both nodes"]],
metrics:[["Hit rate",""],["p99 TTFT",""],["Cross-node bytes moved",""]],
expected:"Cross-node-aware routing keeps the hit rate of a single node while spreading load."},
learn:"In a cluster, a cache hit has a location.",next:"Build the cluster-wide KV store (P1).",check:["When is a remote hit worse than a local miss?"]});

D({unit:"3",n:"17, 18",title:"Cluster cache planes and KV compatibility",status:"open",
choice:"Planes: Redis/etcd for the index, RDMA for the bytes, with Redis-over-TCP as the explicit baseline. Compatibility: put layout (TP degree, KV dtype) in the cache key, or reshard on fetch and measure that cost.",why:"Redis can't do GPU-direct RDMA, and blocks written at TP=8 are sharded by head."});

M({unit:"3",id:"m24",n:24,title:"Cluster-wide KV store",tags:["P1"],status:"planned",
obj:["Separate the cache index from the cache bytes","Compare TCP, RDMA and LMCache data planes"],
lesson:`<p>Prompt caching at cluster scale has four layers: the engine's prefix cache (M6), node tiers (D4), a <b>cluster-wide store</b> (here), and the product layer (P2). P1 builds a global prefix index the router consults, plus a shared KV store that any replica can fetch from.</p>
<p>The index is small and needs consistency (Redis or etcd). The bytes are large and need bandwidth (RDMA via NIXL or Mooncake Transfer Engine). Mixing them in Redis measures TCP and serialization, not EFA.</p>`,
formula:`fetch time = bytes ÷ data-plane bandwidth + index lookup
hit rate rises with store size, with diminishing returns`,
predict:["Hit rate vs cache size on workload C.","Fetch time per transport (Redis/TCP, NIXL, LMCache) at 4K and 32K tokens."],
lab:{name:"P1 · Global prefix cache",why:"Measure cross-node fetch against local recompute, by prompt length and transport.",
spec:[["Instance","2× p4d or p5"],["Transports","Redis over TCP (baseline), NIXL or Mooncake TE, LMCache"],["Workload","Workload C; Mooncake trace replay (decision 2)"]],
metrics:[["Hit rate vs cache size",""],["Fetch vs recompute per transport",""]],
expected:"RDMA fetch wins well below the length where TCP does."},
learn:"The data plane decides whether a cluster cache is worth having.",next:"Turn the cache into a product (P2).",check:["Why keep the index and the bytes in different systems?"]});

M({unit:"3",id:"m25",n:25,title:"Cache as a product",tags:["P2"],status:"planned",
obj:["Set TTL and minimum cacheable length","Build a pricing model","Close the timing side channel with salting"],
lesson:`<p>Caching costs memory, so it has a price. Providers charge a premium to write the cache and a large discount to read it, with a short TTL. P2 sweeps TTL against memory cost, sets a minimum cacheable length, and builds a pricing model (cf. Anthropic: ~1.25× to write, ~0.1× to read, 5-minute TTL).</p>
<p>Sharing a cache across tenants leaks information: a fast TTFT reveals that someone else sent the same prefix. Salting the cache key per tenant (built into decision 8) closes that channel.</p>`,
formula:`break-even reads = write premium ÷ (1 − read price)
  1.25× write, 0.1× read → 0.25 ÷ 0.9 ≈ 0.28 extra reads`,
predict:["Memory cost per TTL setting on workload C.","Hit rate lost to per-tenant salting."],
lab:{name:"P2 · TTL, pricing, isolation",why:"See how cache policy becomes a price, and what isolation costs.",
spec:[["Instance","p4d or p5"],["Workload","Workload C with multiple tenants"],["Sweeps","TTL, minimum length, salt on/off"]],
metrics:[["Hit rate vs TTL",""],["Memory cost",""],["Timing leak with and without salt",""]],
expected:"Hit rate saturates past a few minutes of TTL; salting removes the timing signal at a small hit-rate cost."},
learn:"Cache policy is a pricing decision and a security boundary.",next:"Prefetch an agent's KV before it needs it (P3).",check:["Why does salting cost hit rate?"]});

M({unit:"3",id:"m26",n:26,title:"Agent KV prefetch",tags:["P3"],status:"planned",
obj:["Offload during a tool call, restore on resume","Prefetch before the tool returns"],
lesson:`<p>During a tool call the session's KV can move out of HBM; when the tool returns, it must come back. If the engine knows the expected tool duration (decision 12), it can start fetching <i>before</i> the tool returns, so resume TTFT approaches a warm hit. This is predictive pre-population from patent US12547545B2, at cluster scale.</p>`,
formula:`resume TTFT ≈ max(0, fetch time − (tool time − prefetch lead)) + new-token prefill`,
diag:"tiers",
predict:["Resume TTFT with offload and no prefetch vs with prefetch.","How early must prefetch start for an NVMe-tier session of 8K tokens?"],
lab:{name:"P3 · Resume with and without prefetch",why:"Show the agent-serving differentiator at cluster scale.",
spec:[["Instance","2× p4d or p5"],["Workload","Workload C with tool-duration hints"],["Gates","Restored sessions produce identical tokens"]],
metrics:[["Resume TTFT",""],["HBM freed during pauses",""],["Prefetch accuracy",""]],
expected:"Prefetch brings resume TTFT close to a warm hit while freeing HBM during pauses."},
learn:"Knowing the agent's schedule turns a cache miss into a hit.",next:"Probe a production system (P4).",check:["What happens if the tool returns earlier than predicted?"]});

M({unit:"3",id:"m27",n:27,title:"Probing production prompt caching",tags:["P4"],status:"planned",
obj:["Measure cached vs uncached TTFT from outside","Infer TTL from observed behaviour"],
lesson:`<p>A black-box probe of Anthropic or Bedrock prompt caching grounds your numbers: send the same long prefix at varying intervals, record TTFT and the billed cache reads and writes, and infer the TTL and the speed-up.</p>`,
formula:`speed-up = uncached TTFT ÷ cached TTFT
observed TTL = longest gap that still bills a cache read`,
predict:["Cached vs uncached TTFT for a 10K-token prefix.","Observed TTL."],
lab:{name:"P4 · External comparison",why:"Compare home-built caching with a production system.",
spec:[["Target","Anthropic API / Bedrock prompt caching"],["Workload","Fixed long prefix, interval sweep"],["Cost","Per-call billing recorded"]],
metrics:[["TTFT cached vs uncached",""],["Observed TTL",""],["Cost per call",""]],
expected:"Cached TTFT several times lower; TTL near the documented value."},
learn:"Production numbers calibrate your own.",next:"Unit 4: production stacks.",check:["Why measure cost per call alongside TTFT?"]});

/* ---------------- Unit 4 ---------------- */
M({unit:"4",id:"m28",n:28,title:"vLLM + llm-d / Dynamo on EKS",tags:["EKS"],status:"planned",
obj:["Deploy a production serving stack","Compare it with your own system and explain the gap"],
lesson:`<p>llm-d (Kubernetes-native, Go) and NVIDIA Dynamo (Rust core) package what Units 2–3 built: cache-aware routing, P/D disaggregation, KV transfer via NIXL, autoscaling. Deploying one on EKS and running your workloads through it shows which of your design choices match production and where they differ.</p>`,
formula:`gap per workload = production goodput ÷ own goodput, explained by component`,
predict:["Where will the production stack be clearly ahead: kernels, routing, or KV transfer?"],
lab:{name:"Production stack comparison",why:"Ground the whole distributed design against what teams deploy.",
spec:[["Platform","EKS with GPU node groups (p4d / p5)"],["Stack","vLLM + llm-d or NVIDIA Dynamo"],["Workload","A, B, C, Mix with the same client"]],
metrics:[["Goodput, p99 TTFT / ITL",""],["Hit rate",""],["Operational effort","Deploy and scale steps"]],
expected:"Production ahead on kernels and operations; routing and caching comparable if Units 2–3 went well."},
learn:"Production stacks are the same ideas with more engineering hours.",next:"Put a price on it (module 29).",check:["Which of your components maps to which llm-d / Dynamo component?"]});

M({unit:"4",id:"m29",n:29,title:"Self-host vs API cost",tags:["cost"],status:"planned",
obj:["Compute cost per job three ways","Find the break-even utilisation"],
lesson:`<p>Run the same 2,000-paper summarization job three ways: p4/p5 with vLLM, Bedrock real-time, and Bedrock batch. Self-hosting has a fixed hourly cost, so its price per paper depends on how busy you keep the GPUs. APIs charge per token. The <b>break-even utilisation</b> is where they cost the same.</p>`,
formula:`self-host cost/paper = instance $/h ÷ (papers/h at utilisation u)
break-even u: self-host cost/paper = API cost/paper`,
predict:["Cost per 1,000 papers each way.","Break-even utilisation for p4d vs Bedrock batch."],
lab:{name:"Cost study",why:"Turn throughput numbers into a business answer.",
spec:[["Self-host","p4/p5 + vLLM"],["API","Bedrock real-time and batch"],["Job","2,000 arXiv papers (workload A)"]],
metrics:[["Cost per 1,000 papers",""],["Wall-clock time",""],["Break-even utilisation",""]],
expected:"Batch API cheapest at low utilisation; self-host wins only when kept busy."},
learn:"Throughput is only useful once it has a price.",next:"Optional Neuron run and upstream work (module 30).",check:["Why does utilisation decide self-host cost?"]});

M({unit:"4",id:"m30",n:30,title:"Neuron and upstream contribution",tags:["optional"],status:"optional",
obj:["Run the same job on Trainium/Inferentia","Contribute to llm-d, SGLang or vLLM"],
lesson:`<p>Optionally, run the cost-study job with vLLM-on-Neuron to close the Trainium/Inferentia gap. Then, with D2–D4 and Phase 3 data in hand, write substantive comments or a small PR on the open agent-aware KV-cache issues in llm-d, SGLang and vLLM.</p>`,
formula:`same job, same metrics: cost per 1,000 papers on Neuron`,
predict:["Cost per 1,000 papers on Inferentia vs p4d."],
lab:{name:"Neuron run + upstream",why:"Cover non-NVIDIA hardware and turn findings into public work.",
spec:[["Instance","inf2 / trn1"],["Stack","vLLM on Neuron"],["Upstream","Comments or PR on agent-aware KV caching"]],
metrics:[["Cost per 1,000 papers",""],["Upstream outcome","Merged / discussed"]],
expected:"A cost data point on Neuron and at least one substantive upstream contribution."},
learn:"Public contributions turn private results into evidence.",next:"Unit 5: Kimi K2.",check:["What data from Units 2–3 would make an upstream comment substantive?"]});

/* ---------------- Unit 5 ---------------- */
M({unit:"5",id:"m31",n:31,title:"Multi-head Latent Attention",tags:["K1",["memory","m"]],status:"planned",
obj:["Explain the MLA latent KV","Compute KV bytes per token for K2","See why MLA needs data-parallel attention"],
lesson:`<p>MLA (DeepSeek-V2/V3, used by Kimi K2) stores one compressed latent per layer per token (512 dims plus 64 RoPE dims) instead of separate keys and values per head. At decode time, <b>weight absorption</b> folds the up-projection into the query and output matrices so attention runs directly on the latent.</p>
<p>The KV cache shrinks dramatically, which shifts every transfer and caching calculation in favour of moving KV. The catch: the latent isn't split by head, so tensor parallelism would replicate it on every GPU. Hence <b>data-parallel attention</b> in K3.</p>`,
formula:`K2 KV/token ≈ 61 layers × (512 + 64) × 2 bytes ≈ 69 KB
70B GQA: 320 KB   →  ~4.6× smaller`,
example:`<b>Worked example</b>An 8K prompt on K2 is ≈ 0.56 GB of KV, against ~2.6 GB for a 70B GQA model in bf16.`,
predict:["Max concurrency for Moonlight vs the 8B GQA model at the same KV budget.","Decode step time with and without weight absorption."],
lab:{name:"K1 · MLA on Moonlight",why:"See how MLA changes memory, caching and transfer math.",
spec:[["Instance","g6.12xlarge (4× L4) or one p-node"],["Model","Moonlight-16B-A3B (~32 GB bf16); tiny random DeepseekV3 config for G0/G1"],["Gates","G0/G1 vs HF on the tiny config; G2 on Moonlight"]],
metrics:[["KV bytes/token",""],["Max concurrency",""],["Decode step time",""]],
expected:"KV per token several× smaller than GQA, with concurrency rising to match."},
learn:"Attention design changes the whole memory budget.",next:"The MoE layer (K2).",check:["Why can't TP shard the MLA latent?"]});

M({unit:"5",id:"m32",n:32,title:"The MoE layer",tags:["K2",["compute","c"]],status:"planned",
obj:["Route tokens to experts","Use grouped GEMM","Find the batch where nearly all experts are hot"],
lesson:`<p>Kimi K2 has 384 experts per MoE layer; each token's router picks 8, plus 1 shared expert. At batch 1 a decode step reads only the ~32B active parameters. As batch grows, different tokens pick different experts, until almost every expert is touched and the step reads nearly all 1T parameters.</p>
<p>That makes MoE serving economics depend on large decode batches: the weight read is shared once enough tokens use each expert. Experts run as a <b>grouped GEMM</b> via a borrowed fused-MoE kernel.</p>`,
formula:`P(expert untouched at batch B) = (1 − 8/384)^B
B = 64 → 26% untouched   B = 256 → 0.5% untouched`,
example:`<b>Worked example</b>At batch 1, 8 of 384 experts are read. By batch ~256 nearly all are, so the bytes per step approach the full weights, but they are shared by 256 tokens.`,
predict:["Decode step time vs batch for Moonlight.","The batch at which ~90% of experts are hot."],
lab:{name:"K2 · MoE decode vs batch",why:"See why MoE needs big decode batches.",
spec:[["Instance","g6.12xlarge or one p-node"],["Model","Moonlight-16B-A3B"],["Kernel","Borrowed fused-MoE grouped GEMM"]],
metrics:[["Step time vs batch",""],["Experts touched per step",""]],
expected:"Step time rises steeply until most experts are hot, then flattens."},
learn:"MoE trades per-token cost for a dependence on batch size.",next:"Decide Phase 5 choices, then expert parallelism (K3).",check:["Why is MoE cheaper per token at large batch?"]});

D({unit:"5",n:"24",title:"Phase 5 choices",status:"open",
choice:"Checkpoint: newest text-only INT4 K2.x. Hardware: 1× p5en (simpler) vs 2× p5 (needed for cross-node EP). All-to-all on EFA: check DeepEP, pplx-kernels and NVSHMEM-over-libfabric; NCCL all-to-all as fallback. Weights: stage 0.6–1 TB in same-region S3, then local NVMe or FSx. Budget: K1–K5 on Moonlight, one short Capacity Block for K0 + K6.",why:"K3's process layout must support EP + DP attention."});

M({unit:"5",id:"m33",n:33,title:"Expert parallelism + data-parallel attention",tags:["K3"],status:"planned",
obj:["Place experts across GPUs","Dispatch and combine with all-to-all","Pair EP with data-parallel attention"],
lesson:`<p><b>Expert parallelism</b> puts different experts on different GPUs. Every MoE layer, each token is sent to the GPUs holding its 8 experts (<b>dispatch</b>) and the results come back (<b>combine</b>): two all-to-alls per layer. Attention runs <b>data-parallel</b>: each GPU handles its own requests with a full copy of the attention weights and its own MLA latents.</p>
<p>All-to-all is the hardest communication pattern in this roadmap; over EFA between nodes it is the main risk of Phase 5.</p>`,
formula:`per MoE layer: 2 all-to-alls of tokens × hidden × 2 bytes × experts per token
Moonlight, EP=8: 64 routed experts ÷ 8 = 8 per GPU
Kimi K2,   EP=8: 384 ÷ 8 = 48 per GPU   (EP=16: 24)`,
diag:"moe",
predict:["All-to-all share of step time at EP=8 on NVLink.","Per-expert load imbalance on workload C."],
lab:{name:"K3 · EP=8 on one node",why:"Learn the core MoE serving skill on a model that fits.",
spec:[["Instance","One 8-GPU p-node"],["Model","Moonlight-16B-A3B, EP=8"],["Gates","G1b vs single-GPU Moonlight"]],
metrics:[["All-to-all share of step",""],["Tokens per expert","Imbalance"],["Decode throughput",""]],
expected:"All-to-all a large share of step time; visible imbalance across experts."},
learn:"In EP, communication and imbalance set the step time.",next:"Balance the experts (K4).",check:["Why is attention data-parallel while experts are split?"]});

M({unit:"5",id:"m34",n:34,title:"Expert load balancing",tags:["K4"],status:"planned",
obj:["Replicate hot experts","Rebalance periodically (EPLB)"],
lesson:`<p>In EP every GPU waits for the slowest one each layer. Popular experts make their GPUs stragglers. <b>EPLB</b> keeps redundant copies of hot experts and periodically reshuffles placement based on observed load.</p>`,
formula:`step time ≈ max over GPUs (tokens routed to that GPU)
imbalance = max ÷ mean tokens per GPU`,
predict:["p99 ITL with and without balancing on skewed traffic.","How many redundant experts bring imbalance under 1.2?"],
lab:{name:"K4 · Balancing on skewed traffic",why:"Show that stragglers set the step time in EP.",
spec:[["Instance","One 8-GPU p-node"],["Model","Moonlight-16B-A3B, EP=8"],["Workload","Skewed traffic"]],
metrics:[["Imbalance ratio",""],["p99 ITL",""]],
expected:"Balancing lowers imbalance and p99 ITL noticeably."},
learn:"The slowest GPU sets everyone's pace.",next:"Load quantized MoE weights (K5).",check:["Why replicate experts rather than split them further?"]});

M({unit:"5",id:"m35",n:35,title:"Quantized MoE",tags:["K5","G2",["memory","m"]],status:"planned",
obj:["Load INT4 or block-FP8 MoE weights","Gate on quality"],
lesson:`<p>K2-Instruct ships block-FP8 (~1 TB); K2 Thinking and later ship native INT4 from quantization-aware training (~594 GB). bf16 would be ~2 TB. Borrowed kernels (Marlin-MoE, DeepGEMM) run these formats. As in M9, bytes per step drive decode time, and <b>G2 is the hard gate</b>.</p>`,
formula:`K2 weights: bf16 ~2 TB · FP8 ~1 TB · INT4 ~594 GB
8× H100 80 GB = 640 GB · 8× H200 = 1,128 GB`,
predict:["ITL and bytes per step for INT4 vs bf16 Moonlight.","G2 delta."],
lab:{name:"K5 · INT4 / FP8 MoE",why:"See why K2 ships INT4 natively.",
spec:[["Instance","One p-node"],["Model","Moonlight, INT4 (compressed-tensors) or block-FP8"],["Gates","G2 within noise (hard gate)"]],
metrics:[["Bytes per step",""],["ITL",""],["G2 scores",""]],
expected:"ITL falls with the byte ratio; quality within noise for QAT weights."},
learn:"For trillion-parameter models, precision decides whether it fits at all.",next:"The booked session: K0, K6, K7.",check:["Why does INT4 on 8× H100 leave almost no room for KV?"]});

M({unit:"5",id:"m36",n:36,title:"Kimi K2 across two nodes",tags:["K0","K6","K7"],status:"planned",
obj:["Run a production reference","Serve K2 on your engine across nodes","Explain the gap"],
lesson:`<p>One booked session does three things. <b>K0</b>: serve K2.x INT4 with SGLang or vLLM on 1× p5en or 2× p5, on workload C, as the reference. <b>K6</b>: serve it on your engine, with EP inside each node and, across nodes, either EP=16 over EFA or a prefill node plus a decode node. <b>K7</b>: explain the remaining gap (kernels, comms library, scheduler), with the same honesty rule as S11.</p>
<p>The reference design is LMSYS's K2 deployment on 128 H200s with P/D disaggregation, DeepEP and EPLB. This is a 1–2 node version of it.</p>`,
formula:`MLA KV for P/D: 8K tokens ≈ 0.56 GB per request
cross-node EP: 2 all-to-alls per MoE layer over EFA`,
diag:"moe",
predict:["Decode tok/s for K0 vs K6.","Which layout wins across nodes: EP=16 or P/D?","The largest single cause of the K7 gap."],
lab:{name:"K0 / K6 / K7 · Capstone",why:"Frontier-model serving on your own engine, measured against production.",
spec:[["Instance","1× p5en (8× H200) or 2× p5 over EFA, Capacity Block"],["Model","Kimi K2.x INT4 (~594 GB), staged in same-region S3 → NVMe"],["Reference","SGLang or vLLM"],["Workload","Workload C"]],
metrics:[["Decode tok/s",""],["TTFT / ITL",""],["All-to-all over EFA vs NVLink",""],["Cold-start time","A D5 data point"]],
expected:"Your engine runs K2 within a measured gap of SGLang, with each part of the gap explained."},
learn:"The same ideas from M0 onward scale to a trillion parameters.",next:"Done. Update the resume bullets with real numbers.",check:["Which earlier module's lesson matters most for K6?"]});

/* ---------------- Appendix ---------------- */
const APPENDIX = `
<div class="sec"><h2>Formula sheet</h2>
<div class="formula">FLOPs per token          ≈ 2 × params
prompt attention         ≈ 2 · n² · d · L
attention per decode tok ≈ 4 · ctx · d · L
KV bytes per token       = 2 × layers × kv_heads × head_dim × bytes
MFU                      = achieved FLOP/s ÷ peak
MBU                      = (weight bytes + KV bytes read) / step time ÷ peak bandwidth
decode floor (batch 1)   = weight bytes ÷ bandwidth
utilisation              ρ = λ × S;  M/D/1 wait = ρS / (2(1−ρ))
spec decoding            tokens/pass = (1 − α^(k+1)) / (1 − α)
transfer time            = bytes ÷ link bandwidth</div></div>
<div class="sec"><h2>GPUs</h2>
<div class="tbl"><table><thead><tr><th>GPU (instance)</th><th>bf16 dense</th><th>Bandwidth</th><th>Balance</th><th>8B floor</th><th>1B floor</th></tr></thead><tbody>
<tr><td>L4 (g6)</td><td class="n">~121 TFLOPS</td><td class="n">~300 GB/s</td><td class="n">~400 FLOP/B</td><td class="n">~54 ms</td><td class="n">~8 ms</td></tr>
<tr><td>A100 40 GB (p4d)</td><td class="n">312 TFLOPS</td><td class="n">1.56 TB/s</td><td class="n">~200 FLOP/B</td><td class="n">~10.3 ms</td><td class="n">~1.6 ms</td></tr>
<tr><td>A100 80 GB (p4de)</td><td class="n">312 TFLOPS</td><td class="n">2.04 TB/s</td><td class="n">~153 FLOP/B</td><td class="n">~7.9 ms</td><td class="n">~1.2 ms</td></tr>
<tr><td>H100 (p5)</td><td class="n">~989 TFLOPS</td><td class="n">3.35 TB/s</td><td class="n">~295 FLOP/B</td><td class="n">~4.8 ms</td><td class="n">~0.7 ms</td></tr>
</tbody></table></div>
<div class="tbl"><table><thead><tr><th>Link</th><th>Bandwidth</th></tr></thead><tbody>
<tr><td>NVLink, inside a node</td><td class="n">~900 GB/s per GPU</td></tr>
<tr><td>EFA, p4d</td><td class="n">~400 Gbps ≈ 50 GB/s per node</td></tr>
<tr><td>EFA, p5</td><td class="n">~3,200 Gbps ≈ 400 GB/s per node</td></tr>
</tbody></table></div></div>
<div class="sec"><h2>Decision log</h2><div class="tbl"><table><thead><tr><th>#</th><th>Decision</th><th>Needed by</th><th>Status</th></tr></thead><tbody>
<tr><td class="n">1</td><td>Engine process architecture</td><td>M2</td><td>Decided 2026-10-11</td></tr>
<tr><td class="n">2</td><td>Workload C spec</td><td>M2</td><td>Decided 2026-10-11</td></tr>
<tr><td class="n">3</td><td>SLOs and metrics</td><td>M2</td><td>Decided 2026-10-11</td></tr>
<tr><td class="n">4</td><td>Correctness method</td><td>M2</td><td>Decided 2026-10-11</td></tr>
<tr><td class="n">5</td><td>Benchmark client</td><td>M2</td><td>Decided 2026-10-11</td></tr>
<tr><td class="n">6</td><td>Observability</td><td>M2</td><td>Decided 2026-10-11</td></tr>
<tr><td class="n">7</td><td>Attention backend and KV layout</td><td>M5</td><td>Decided 2026-10-18</td></tr>
<tr><td class="n">8</td><td>Block-hash spec</td><td>M6</td><td>Decided 2026-10-18</td></tr>
<tr><td class="n">9</td><td>Scheduler policy</td><td>M4</td><td>Decided 2026-10-18</td></tr>
<tr><td class="n">10</td><td>Systems language</td><td>D2</td><td>Open</td></tr>
<tr><td class="n">11</td><td>Router state and tokenization</td><td>D2</td><td>Open</td></tr>
<tr><td class="n">12</td><td>Session / agent API</td><td>D4</td><td>Open</td></tr>
<tr><td class="n">13</td><td>Pipeline-parallelism gap</td><td>D1b / N2</td><td>Open</td></tr>
<tr><td class="n">14</td><td>KV transfer mechanism</td><td>D3</td><td>Open</td></tr>
<tr><td class="n">15</td><td>Failure model</td><td>D5</td><td>Open</td></tr>
<tr><td class="n">16</td><td>Multi-GPU dev hardware</td><td>D1</td><td>Open</td></tr>
<tr><td class="n">17</td><td>Cluster cache planes</td><td>P1</td><td>Open</td></tr>
<tr><td class="n">18</td><td>KV compatibility across engines</td><td>P1</td><td>Open</td></tr>
<tr><td class="n">19</td><td>Eviction and TTL across tiers</td><td>D4</td><td>Open</td></tr>
<tr><td class="n">20</td><td>AWS access and quotas</td><td>Phase 0 / N1</td><td>Open</td></tr>
<tr><td class="n">21</td><td>Budget per phase</td><td>Phase 0</td><td>Open</td></tr>
<tr><td class="n">22</td><td>Exit criteria and write-ups</td><td>All</td><td>Set in the plan</td></tr>
<tr><td class="n">23</td><td>MVP cut: Phase 1a + M7 + D2 + D3</td><td>All</td><td>Set in the plan</td></tr>
<tr><td class="n">24</td><td>Phase 5 choices</td><td>K3</td><td>Open</td></tr>
</tbody></table></div></div>
<div class="sec"><h2>Glossary</h2><dl class="gloss">
<dt>TTFT</dt><dd>Time to first token, as the client sees it: HTTP and tokenization, the hop to the engine, queue wait, prefill and first-token emission.</dd>
<dt>ITL / TPOT</dt><dd>Inter-token latency / time per output token during decode.</dd>
<dt>Goodput</dt><dd>Requests per second that meet both SLOs.</dd>
<dt>Prefill</dt><dd>The pass over the whole prompt; compute-bound.</dd>
<dt>Decode</dt><dd>One pass per output token; memory-bound.</dd>
<dt>MFU / MBU</dt><dd>Share of peak FLOP/s / peak memory bandwidth actually used.</dd>
<dt>Balance point</dt><dd>Peak FLOP/s ÷ bandwidth; the arithmetic intensity where a GPU switches from memory- to compute-bound.</dd>
<dt>KV cache</dt><dd>Stored keys and values for every earlier position, per layer.</dd>
<dt>Paged KV</dt><dd>KV in fixed-size blocks mapped through a per-request block table.</dd>
<dt>Preemption</dt><dd>Evicting a running request's KV to make room; it resumes by recompute.</dd>
<dt>Prefix caching</dt><dd>Reusing KV blocks of an identical prompt prefix, found by chained hash.</dd>
<dt>Chunked prefill</dt><dd>Splitting a long prefill into token-budgeted chunks mixed with decodes.</dd>
<dt>Speculative decoding</dt><dd>Drafting k tokens cheaply and verifying them in one target pass.</dd>
<dt>TP / PP / EP</dt><dd>Tensor, pipeline and expert parallelism.</dd>
<dt>All-reduce / all-to-all</dt><dd>Collective operations: sum across GPUs / every GPU sends to every other.</dd>
<dt>P/D disaggregation</dt><dd>Prefill and decode on separate workers, with KV transfer between them.</dd>
<dt>MLA</dt><dd>Multi-head Latent Attention: one compressed latent per token per layer.</dd>
<dt>MoE</dt><dd>Mixture of experts: a router sends each token to a few expert MLPs.</dd>
<dt>EPLB</dt><dd>Expert-parallel load balancing: replicate and re-place hot experts.</dd>
<dt>EFA</dt><dd>AWS Elastic Fabric Adapter, the inter-node RDMA network.</dd>
<dt>NIXL</dt><dd>NVIDIA's transfer library for moving KV between GPUs and nodes.</dd>
</dl></div>
<div class="sec"><h2>Deferred: kernel track</h2>
<div class="tbl"><table><thead><tr><th>#</th><th>Kernel</th><th>Notes</th></tr></thead><tbody>
<tr><td>KT1</td><td>Naive attention in PyTorch</td><td>Materializes the n×n matrix; baseline</td></tr>
<tr><td>KT2</td><td>Fused prefill attention in Triton</td><td>FlashAttention-style tiling, online softmax; ~1–2 weekends</td></tr>
<tr><td>KT3</td><td>Paged decode attention in Triton</td><td>Reads KV through the block table; drops into M5 in place of FlashInfer; measured by MBU</td></tr>
</tbody></table></div>
<p>Revisit after Phase 2, or if a target role requires it.</p></div>`;
