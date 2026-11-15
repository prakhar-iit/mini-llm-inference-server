/* ---------------- Further reading: 100 Days of Inference ----------------
   Elizabeth T's day-by-day series (github.com/elizabetht/100-days-of-inference), built around
   Philip Kiely's "Inference Engineering" (Baseten Books, 2026). Each day is a short write-up plus a notebook.
   READ maps a module id to [day, note]; the note says what to read the day for. Days 1–26 exist as of
   2026-11-14; modules with no matching day list none. Add days here as the series grows. */
const READ_REPO = "https://github.com/elizabetht/100-days-of-inference";
const DAYS = {
  1:"LLM inference mechanics: end-to-end text generation",
  2:"Inference from scratch: model internals and tokenization",
  3:"Embeddings: from integers to vectors",
  4:"Transformer blocks and attention",
  5:"KV cache",
  6:"Ops:byte ratio and the roofline model",
  7:"CUDA kernels, kernel selection and kernel fusion",
  8:"PyTorch, model file formats, ONNX and TensorRT",
  9:"vLLM: PagedAttention and continuous batching",
  10:"SGLang: RadixAttention and structured outputs",
  11:"TensorRT-LLM: compilation and plugins",
  12:"NVIDIA Dynamo: disaggregated serving",
  13:"Quantization: number formats (FP8, INT8, INT4, NVFP4)",
  14:"Quantization algorithms: GPTQ, AWQ, SmoothQuant",
  15:"Speculative decoding: draft-target, Medusa, EAGLE",
  16:"Prefix caching, KV memory tiers and cache-aware routing",
  17:"Model parallelism: tensor, expert, pipeline and data",
  18:"Disaggregation: prefill/decode split",
  19:"GPU architecture: SMs, memory hierarchy, HBM",
  20:"GPU generations: Hopper, Ada, Blackwell, Rubin",
  21:"Multi-GPU instances and MIG",
  22:"Containerization: Docker and NVIDIA NIMs",
  23:"Autoscaling: concurrency, batching and cold starts",
  24:"Routing, load balancing and queueing",
  25:"Multi-cloud capacity management",
  26:"Zero-downtime deployment and cost estimation"
};
const dayUrl = d => `${READ_REPO}/tree/main/day${String(d).padStart(2,"0")}`;

const READ = {
  m1:[[1,"The generate loop and why each step re-reads the whole sequence"],[2,"What is inside the weights; tokenization"],[3,"The embedding lookup, the first stage of the forward pass"],[4,"Attention and the transformer block"]],
  m2:[[5,"Why the cache removes the repeated work"],[6,"Roofline, balance point, why decode is memory-bound"],[19,"HBM and the memory hierarchy behind the decode floor"]],
  m3:[[24,"Queueing: Little's law and why wait explodes near full utilisation"]],
  m4:[[6,"Batching as the lever that moves decode up the roofline"]],
  m5:[[9,"Continuous batching vs static batching"],[24,"Queues, priorities and deadlines"]],
  m6:[[9,"PagedAttention: blocks, block tables, fragmentation"],[7,"Kernels and FlashAttention's tiling"],[19,"SRAM vs HBM, why paged kernels must avoid gathers"]],
  m7:[[16,"Prefix caching and why prompt layout decides the hit rate"],[10,"RadixAttention, SGLang's alternative to block hashes"]],
  m8:[],
  m9:[],
  m10:[[7,"Kernel launch overhead and fusion"],[8,"Compiled runtimes (TensorRT) that remove launches"]],
  m11:[[13,"FP8, INT8, INT4 and NVFP4 formats"],[14,"GPTQ, AWQ and SmoothQuant"]],
  m12:[[15,"Draft-target, Medusa and EAGLE; acceptance rate"]],
  m13:[[9,"vLLM, the engine you compare against"],[10,"SGLang"],[11,"TensorRT-LLM"]],
  m14:[[17,"Tensor parallelism and the per-layer sync"],[21,"NVLink vs InfiniBand cost of the all-reduce"]],
  m15:[[17,"Pipeline parallelism and bubbles"]],
  m16:[[24,"Token-aware and KV-cache-aware routing"],[16,"Cache-aware routing as part of prefix caching"]],
  m17:[[18,"Prefill/decode split and the KV transfer"],[12,"Dynamo's prefill and decode workers"]],
  m18:[[16,"KV memory tiers (VRAM, host RAM, SSD) and offload vs re-prefill"]],
  m19:[[23,"Autoscaling and cold starts"],[26,"Zero-downtime deployment and graceful shutdown"]],
  m20:[[21,"Multi-GPU instances, NVLink inside vs InfiniBand between nodes"],[20,"Interconnect bandwidth by GPU generation"]],
  m21:[[17,"Choosing tensor vs pipeline parallelism"],[21,"Why TP stops at the node boundary"]],
  m22:[[18,"KV transfer size and NIXL"],[12,"Dynamo's KV transfer layer"]],
  m23:[[24,"Routing across replicas"],[25,"Geo-aware load balancing across sites"]],
  m24:[[16,"Networked SSD tier and fleet-wide cache"],[12,"Dynamo's KV layer"]],
  m25:[[16,"Prefix caching as a prompt design problem"]],
  m26:[[16,"Offload vs re-prefill, the decision prefetch changes"]],
  m27:[[16,"How production prefix caching behaves"]],
  m28:[[12,"NVIDIA Dynamo"],[22,"Containers and NIMs"],[23,"Autoscaling on Kubernetes"]],
  m29:[[26,"$/1M tokens from GPU $/h and throughput"],[25,"Reserved, on-demand and spot supply"]],
  m30:[],
  m31:[],
  m32:[[17,"Expert parallelism"],[10,"SGLang for MoE models"]],
  m33:[[17,"Expert and data parallelism together"]],
  m34:[],
  m35:[[13,"Low-bit formats"],[14,"Weight quantization algorithms"]],
  m36:[[17,"Why a 671B-class model needs a whole node or more"],[18,"Disaggregation at scale"],[20,"Hopper H100 vs H200"]]
};
