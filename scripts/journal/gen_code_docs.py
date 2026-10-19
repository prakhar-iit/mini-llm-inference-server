"""Build one JSON doc per (milestone, module) for the journal page's "Code by milestone" tab.

Usage: python scripts/journal/gen_code_docs.py <out_dir>
Each milestone lists its modules with a role line and review questions; new files are taken whole at
the milestone's tag (or the working tree for the milestone in progress), changed files as a diff from the
previous tag. The JSON files are then written to the page's database (collection "code").
"""
import json, subprocess, pathlib, sys
REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(sys.argv[1])

def git(*a):
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout

def at_ref(ref, path):  # file contents at a git ref
    return git("show", f"{ref}:{path}")

def working(path):
    return (REPO / path).read_text()

def diff(ref, path):
    return git("diff", "-U4", ref, "--", path)

M = []  # (milestone, path, kind, fmt, code, role, review)

def add(ms, path, kind, fmt, code, role, review):
    M.append(dict(milestone=ms, path=path, kind=kind, format=fmt, code=code, role=role, review=review))

R = "m0"
# ---------------- M0 ----------------
add("M0", "src/minillm/model/llama.py", "new", "python", at_ref(R, "src/minillm/model/llama.py"),
    "The whole Llama forward pass: RMSNorm, RoPE, attention, SwiGLU MLP, decoder layers, model, LM head.",
    ["Trace one prompt through `LlamaForCausalLM.forward` and write the shape after every line, using B=1, T=6, D=2048.",
     "`RMSNorm`: why normalize in fp32 and multiply by `weight` only after casting back?",
     "`apply_rotary`: what pairing does `rotate_half` imply, and what breaks with interleaved pairs?",
     "`LlamaAttention`: what do `.view(b, t, -1, dh).transpose(1, 2)` and the reverse `transpose + reshape` do to the memory layout?",
     "`LlamaDecoderLayer`: rewrite it as post-norm and explain why pre-norm trains better in deep stacks.",
     "Why is `lm_head.weight` assigned to `embed_tokens.weight` when `tie_word_embeddings` is set?"])
add("M0", "src/minillm/model/rope.py", "new", "python", at_ref(R, "src/minillm/model/rope.py"),
    "RoPE frequencies (`inv_freq`), including Llama 3's long-context scaling.",
    ["What is the rotation speed of pair 0 and of the last pair for head_dim 64 and theta 500,000?",
     "Llama 3 scaling: which band of frequencies is divided by `factor`, which is untouched, and why interpolate between them?"])
add("M0", "src/minillm/model/config.py", "new", "python", at_ref(R, "src/minillm/model/config.py"),
    "Reads config.json into a frozen dataclass; accepts both the Hub format and transformers 5's `rope_parameters`.",
    ["Which config fields change between Llama-3.2-1B, 3.1-8B and 3.3-70B? Everything else is the same code.",
     "`num_kv_groups`: how many query heads share one KV head in the 1B model?"])
add("M0", "src/minillm/model/loader.py", "new", "python", at_ref(R, "src/minillm/model/loader.py"),
    "Builds the model on the meta device and assigns safetensors tensors as its parameters.",
    ["What is the meta device, and why does building on it avoid a second copy of the weights?",
     "Explain the two loader bugs: the untied `lm_head` (all `!` output) and the zeroed weights on MPS.",
     "Why must `inv_freq` be rebuilt after loading?"])
add("M0", "src/minillm/generate.py", "new", "python", at_ref(R, "src/minillm/generate.py"),
    "The no-cache greedy loop: full forward, argmax of the last row, append, repeat.",
    ["Count the FLOPs to generate 100 tokens from a 10-token prompt without a cache. Why quadratic?",
     "Why is the EOS token included in the output, and what would break if it weren't?"])
add("M0", "src/minillm/attention/torch_ref.py", "changed", "diff", git("diff", "-U4", "b344255", R, "--", "src/minillm/attention/torch_ref.py"),
    "The reference attention backend's prefill: one call to scaled_dot_product_attention.",
    ["Write the four steps SDPA performs (score, scale, mask, mix) with the tensor shape after each.",
     "What does `enable_gqa=True` save compared with repeating k and v for every query head?"])
add("M0", "tests/test_g0_correctness.py", "test", "python", at_ref(R, "tests/test_g0_correctness.py"),
    "G0 gate: logits and greedy tokens must match Hugging Face, on tiny configs and the real 1B model.",
    ["Why do the tiny tests run in fp32 on CPU but the real-model test in bf16 on MPS?",
     "Why does the real-model test run HF with `use_cache=False`?"])
add("M0", "tests/test_layers.py", "test", "python", at_ref(R, "tests/test_layers.py"),
    "Per-layer checks against HF: RMSNorm, MLP, RoPE and attention, several bit-identical.",
    ["`test_rope_score_depends_only_on_relative_position`: restate the property it proves in one sentence."])

# ---------------- M1 ----------------
add("M1", "src/minillm/model/kv_cache.py", "new", "python", at_ref("m1", "src/minillm/model/kv_cache.py"),
    "Contiguous per-layer K/V buffers: `append` writes this step's k/v, `advance` moves the write position.",
    ["Compute `bytes_per_token` for Llama-3.1-8B (32 layers, 8 KV heads, head_dim 128) in bf16.",
     "Why does `append` not advance `seq_len`, and who calls `advance`?",
     "What goes wrong in this design when two sequences in a batch have different lengths? (M5 fixes it.)"])
add("M1", "src/minillm/model/llama.py", "changed", "diff", git("diff", "-U4", "m0", "m1", "--", "src/minillm/model/llama.py"),
    "The cache threaded through attention, layers and model; positions continue from `seq_len`; `last_only` logits.",
    ["Why is k cached after RoPE instead of before?",
     "What positions does a decode step use, and where are they computed?",
     "How big would full logits be for a 16K-token prompt, and what does `last_only` change?"])
add("M1", "src/minillm/attention/torch_ref.py", "changed", "diff", git("diff", "-U4", "m0", "m1", "--", "src/minillm/attention/torch_ref.py"),
    "Attention for queries on top of a cache: three cases by Tq vs Tk.",
    ["Draw the mask from `tril(diagonal=tk - tq)` for Tq=3, Tk=6. Which query sees which keys?",
     "Why does a single decode query need no mask at all?"])
add("M1", "src/minillm/generate.py", "changed", "diff", git("diff", "-U4", "m0", "m1", "--", "src/minillm/generate.py"),
    "`greedy_generate`: one prefill call, then one-token decode calls against the cache.",
    ["Count the forward calls for a 100-token prompt and 50 new tokens, with and without the cache.",
     "Why does the loop return before running a forward pass for the last token?"])
add("M1", "src/minillm/evals.py", "new", "python", at_ref("m1", "src/minillm/evals.py"),
    "G2 quality eval: WikiText-2 perplexity and few-shot GSM8K.",
    ["Why is perplexity of uniform logits equal to the vocabulary size?",
     "Why are the window's first tokens scored with less context, and why is that fine for a regression check?",
     "Why compute log-softmax in fp32?"])
add("M1", "benchmarks/s1_prefill_decode.py", "tool", "python", at_ref("m1", "benchmarks/s1_prefill_decode.py"),
    "S1 study script: TTFT and prefill MFU vs prompt length; ITL and decode MBU vs context.",
    ["Check `CostModel.prefill_flops` against the plan's formulas. What does it leave out?",
     "Why does `decode_bytes` grow with context, and by how much per token for the 1B model?"])
add("M1", "tests/test_kv_cache.py", "test", "python", at_ref("m1", "tests/test_kv_cache.py"),
    "The cache must change cost, never output: decode, chunked prompts and generation all match a full pass and HF.",
    ["`test_prompt_in_chunks_matches_full_pass` exercises which backend branch, and which later milestones need it?"])

# ---------------- M2 (tag m2) ----------------
add("M2", "src/minillm/engine/protocol.py", "new", "python", at_ref("m2", "src/minillm/engine/protocol.py"),
    "Messages between frontend and engine: AddRequest, Abort, Ready, StepOutputs, as msgpack tagged maps.",
    ["Why msgpack instead of pickle? Name two reasons from decision 1.",
     "Why does the engine see token ids only, never text?",
     "What does `StepOutputs.stats` carry, and who reads it?"])
add("M2", "src/minillm/engine/core.py", "new", "python", at_ref("m2", "src/minillm/engine/core.py"),
    "The engine core: FIFO queue, one request at a time; each step is a prefill or one decode token.",
    ["Walk through `step()` for a new request, then for a running one. Where does the GPU work actually finish?",
     "Why serve one request at a time on purpose? What will S2 show as the arrival rate rises?",
     "What happens to a request's KV cache when it finishes or is aborted?",
     "Which timeline stamps does the engine add, and which come from the frontend?"])
add("M2", "src/minillm/engine/process.py", "new", "python", at_ref("m2", "src/minillm/engine/process.py"),
    "The engine process loop: drain messages, step, send outputs; idle-poll when there's no work.",
    ["Why is the poll timeout 0 while there is work and 50 ms when idle?",
     "Why is the step log written only when the engine is idle?",
     "Why does the frontend start this process with spawn instead of fork?"])
add("M2", "src/minillm/server/engine_client.py", "new", "python", at_ref("m2", "src/minillm/server/engine_client.py"),
    "The frontend's handle on the engine: one reader task routes outputs to per-request queues; disconnects abort.",
    ["Trace a request from `generate()` to the first yielded output.",
     "Why is the Abort sent with NOBLOCK instead of awaited?",
     "What was the R1 bug, and what does `_abandoned` fix?"])
add("M2", "src/minillm/server/api.py", "new", "python", at_ref("m2", "src/minillm/server/api.py"),
    "OpenAI-compatible endpoints, SSE streaming, incremental detokenization, timelines, Prometheus metrics.",
    ["Write the sequence of SSE events for a 3-token streamed completion with include_usage.",
     "Why does the detokenizer hold back text ending in U+FFFD?",
     "Which request fields are rejected with 400, and why reject instead of ignoring?",
     "How is server-side TTFT computed, and how does it differ from the client's TTFT?"])
add("M2", "src/minillm/server/__main__.py", "new", "python", at_ref("m2", "src/minillm/server/__main__.py"),
    "`python -m minillm.server`: builds the app (which starts the engine process) and runs uvicorn.",
    ["What do `--step-log` and `--timeline-log` write, and which studies read them?"])
add("M2", "tests/test_server.py", "test", "python", at_ref("m2", "tests/test_server.py"),
    "End to end on a tiny model: G0 through the API, streaming format, FIFO order, bad requests, R1.",
    ["How does `test_concurrent_requests_are_served_fifo` prove one-at-a-time serving from timelines alone?",
     "Why did the reference first disagree with the server (hint: token 2), and which side was right?"])
add("M2", "tests/test_engine_core.py", "test", "python", at_ref("m2", "tests/test_engine_core.py"),
    "The engine core in-process: FIFO, stop tokens, ignore_eos, aborts, timeline order.",
    ["Why does the stop-token test expect output up to the FIRST occurrence of the stop token?"])

add("M2", "benchmarks/workloads.py", "tool", "python", at_ref("m2", "benchmarks/workloads.py"),
    "Workloads as token ids: random, A (arXiv, 8K in), B (UltraChat, ~100 in / 1,000 out), C (agent sessions).",
    ["Why are prompts token ids rather than text, and why fix outputs with ignore_eos?",
     "Workload C: how is the turns distribution set so its mean is 6? Why lognormal for pauses?",
     "What does the generator do when the context cap cuts a session short, and why?"])
add("M2", "benchmarks/client.py", "tool", "python", at_ref("m2", "benchmarks/client.py"),
    "Open-loop client: gamma arrivals, per-token timestamps on the server's clock, sessions, SLO/goodput summary.",
    ["`arrival_offsets`: derive the gamma shape and scale that give mean 1/rate and coefficient of variation cv.",
     "Why does `run_requests` sleep until each scheduled time instead of awaiting the previous response?",
     "In `run_sessions`, why must the reply's exact token ids be appended to the history?",
     "Which requests count toward goodput, and how is session goodput different?"])
add("M2", "benchmarks/s2_overhead_queueing.py", "tool", "python", at_ref("m2", "benchmarks/s2_overhead_queueing.py"),
    "S2: TTFT split into six stages on an idle server; queue wait vs utilization against the M/D/1 formula.",
    ["Why is the M2 engine an M/D/1 queue? What do M, D and 1 stand for here?",
     "Derive W at rho = 0.9 in units of S. Why does it blow up as rho approaches 1?",
     "The Mac smoke run measured waits below M/D/1. Give two reasons."])
add("M2", "tests/test_client.py", "test", "python", at_ref("m2", "tests/test_client.py"),
    "Arrival statistics, summary math, workload C rules, and client runs against a real tiny server.",
    ["What does `test_open_loop_requests_against_server` prove by comparing client and server timestamps?"])

# ---------------- M3 (tag m3) ----------------
add("M3", "src/minillm/attention/torch_ref.py", "changed", "diff", git("diff", "-U4", "m2", "m3", "--", "src/minillm/attention/torch_ref.py"),
    "Padding-aware attention: causal AND real key, or the query's own position.",
    ["Draw the mask for rows with 2 and 4 real tokens padded to 4. Which entries does `own` add, and why?",
     "What would SDPA return for a padding query with no visible key?"])
add("M3", "src/minillm/model/kv_cache.py", "changed", "diff", git("diff", "-U4", "m2", "m3", "--", "src/minillm/model/kv_cache.py"),
    "The cache records which slots are real (`valid`) and how many real tokens each row has (`num_valid`).",
    ["Why track `has_padding` in Python instead of checking the tensor?",
     "After a padded prefill, why do decode steps still need the padding mask?"])
add("M3", "src/minillm/model/llama.py", "changed", "diff", git("diff", "-U4", "m2", "m3", "--", "src/minillm/model/llama.py"),
    "Per-row positions counting real tokens only; one padding mask built per forward and shared by all layers.",
    ["RoPE is shift-invariant, so shifted positions don't change the output. Why compute exact positions anyway?",
     "Trace positions for prompts of length 3 and 1, then two decode steps."])
add("M3", "src/minillm/generate.py", "changed", "diff", git("diff", "-U4", "m2", "m3", "--", "src/minillm/generate.py"),
    "`pad_left` and `greedy_generate_batch`: a static batch outside the engine, used by G1a and S3.",
    ["Why pad on the left rather than the right?"])
add("M3", "src/minillm/engine/core.py", "changed", "diff", git("diff", "-U4", "m2", "m3", "--", "src/minillm/engine/core.py"),
    "Static batching: admit up to max_batch_size, prefill together, decode in lockstep until the last finishes.",
    ["What happens to a row that finishes early? What does it cost in compute and memory?",
     "Why does a request arriving mid-batch wait for the whole batch even if a row is idle?",
     "How does the step log separate useful work from waste?"])
add("M3", "src/minillm/correctness.py", "new", "python", at_ref("m3", "src/minillm/correctness.py"),
    "G1b: teacher-forced bf16 drift, batch N vs batch 1, against 2× HF's own drift.",
    ["Why teacher-force instead of comparing generated tokens?",
     "Why is the tolerance measured from HF rather than fixed? What are the floors for?",
     "Why are near-ties (margin < 0.5) allowed to flip?"])
add("M3", "benchmarks/s3_batch_sweep.py", "tool", "python", at_ref("m3", "benchmarks/s3_batch_sweep.py"),
    "S3: decode step time and throughput vs batch size; static-batching waste on mixed lengths.",
    ["Derive arithmetic intensity ≈ B FLOP/byte for a bf16 decode step at batch B.",
     "What does `longest_share` measure?"])
add("M3", "tests/test_batching.py", "test", "python", at_ref("m3", "tests/test_batching.py"),
    "G1a (fp32 exact under padding), positions, NaN safety, G1b math and the real-model G1b.",
    ["Which injected bug did the output tests miss, and which test catches it?"])

# ---------------- M4 (tag m4) ----------------
add("M4", "src/minillm/engine/scheduler.py", "new", "python", at_ref("m4", "src/minillm/engine/scheduler.py"),
    "Iteration-level scheduler (decision 9): FCFS admission without skipping, slots, a padded token budget, compaction.",
    ["Walk `schedule()` for queued prompts of 10, 10, 30 tokens with 3 slots and a budget of 40. What runs each step?",
     "Why stop at the first request that doesn't fit instead of skipping to a smaller one behind it?",
     "Why is the head request admitted even when it alone exceeds the budget, and which milestone removes that exception?",
     "`release`: which requests move, and why does that keep every decode step a view rather than a gather?"])
add("M4", "src/minillm/model/kv_cache.py", "changed", "diff", git("diff", "-U4", "m3", "m4", "--", "src/minillm/model/kv_cache.py"),
    "`SlotKVCache`: one slot per running request, max_model_len reserved each; `SlotBatch` is one forward pass's view.",
    ["How much KV does the server reserve at startup for Llama-3.1-8B with 8 slots of 8,192 tokens in bf16?",
     "Why is prefill right-padded here when M3 left-padded? What would break with left padding?",
     "Decode rows have different lengths: how is each row's new k/v written, and which keys does the attention see?",
     "Why are slot lengths kept in Python rather than in a tensor?"])
add("M4", "src/minillm/attention/torch_ref.py", "changed", "diff", git("diff", "-U4", "m3", "m4", "--", "src/minillm/attention/torch_ref.py"),
    "A query may see its own slot only when it sees no real key at all.",
    ["With M3's rule (always allow the own position), which slot does a short decode row read in an M4 batch?",
     "Why is the new rule identical to M3's on left-padded batches?"])
add("M4", "src/minillm/model/llama.py", "changed", "diff", git("diff", "-U4", "m3", "m4", "--", "src/minillm/model/llama.py"),
    "Positions come from the cache; `last_index` picks each right-padded row's last real token.",
    ["Why move the position logic into the cache classes?"])
add("M4", "src/minillm/engine/core.py", "changed", "diff", git("diff", "-U4", "m3", "m4", "--", "src/minillm/engine/core.py"),
    "Continuous batching: each step is a prefill of newly admitted requests or a decode of all running ones.",
    ["Trace three requests (outputs 2, 5, 3 tokens, 2 slots) step by step: which slots, which moves, which outputs?",
     "Why do running requests pause during a prefill step, and what does S4 predict that costs?",
     "Why can an M4 server never run out of KV memory mid-request? What does it pay for that?"])
add("M4", "src/minillm/server/api.py", "changed", "diff", git("diff", "-U4", "m3", "m4", "--", "src/minillm/server/api.py"),
    "R2 admission control: at most max_batch_size + max_waiting in flight; the rest get 503 with Retry-After.",
    ["Why count in-flight requests in the frontend instead of asking the engine?",
     "What does `_ReleasingStream` protect against, and why isn't the stream generator's `finally` enough?"])
add("M4", "benchmarks/s4_mixed_traffic.py", "tool", "python", at_ref("m4", "benchmarks/s4_mixed_traffic.py"),
    "S4: goodput, per-class TTFT and B's inter-token latency vs arrival rate on the Mix workload, M3 vs M4.",
    ["Why is goodput judged per request against its own class's SLO?",
     "Which column shows the ITL spike when an A prompt joins, and why measure it on B requests only?"])
add("M4", "tests/test_continuous_batching.py", "test", "python", at_ref("m4", "tests/test_continuous_batching.py"),
    "M4 spec: slot cache vs batch 1 (all tiny variants), scheduler rules, engine churn, joins, aborts, KV stats.",
    ["Why fill the buffers with 1e4 instead of NaN in `test_stale_slot_contents_are_never_read`?"])

# ---------------- M5a (in progress: working tree vs m4) ----------------
add("M5a", "src/minillm/cache/paged.py", "new", "python", working("src/minillm/cache/paged.py"),
    "The paged pool: per layer, K and V tensors [num_blocks, 16, Hkv, Dh]; `PagedStep` is one step's view of it.",
    ["Where does token 37 of a request with block table [9, 4, 12] live? Give block and offset.",
     "How many blocks fit in 1 GiB for Llama-3.2-1B in bf16? How many tokens is that?",
     "Why does `PagedStep` have nothing to mark or advance, unlike M1's KVCache?"])
add("M5a", "src/minillm/attention/metadata.py", "new", "python", working("src/minillm/attention/metadata.py"),
    "Per-step metadata for a ragged batch: query_start_loc, seq_lens, positions, slot_mapping, block_tables.",
    ["Build the metadata by hand for a decode step of a 20-token request plus a fresh 3-token prompt (block size 16).",
     "Why is it built once per step on the host, and not per layer?"])
add("M5a", "src/minillm/attention/paged_torch_ref.py", "new", "python", working("src/minillm/attention/paged_torch_ref.py"),
    "Reference paged attention: write new k/v by slot, gather each sequence's blocks, run SDPA.",
    ["Why are decode rows attended together but multi-token rows one at a time?",
     "What does the gather cost per step, and what does a fused kernel (M5b/M5c) do instead?",
     "Which mask does a prompt continued on top of cached tokens need?"])
add("M5a", "src/minillm/engine/block_manager.py", "new", "python", working("src/minillm/engine/block_manager.py"),
    "The free list of block ids, and the R3 check: free + held = total, nothing held twice.",
    ["Why does the block manager not know which request holds which block?"])
add("M5a", "src/minillm/engine/scheduler.py", "changed", "diff", diff("m4", "src/minillm/engine/scheduler.py"),
    "Admission by blocks with a one-step lookahead; preemption of the newest request by recompute.",
    ["Why reserve a block only for requests at a block boundary, rather than one per running request?",
     "Walk a preemption: which request goes, what happens to its blocks and output, where does it re-enter the queue?",
     "Why does a preempted request recompute prompt + output rather than keeping its KV somewhere?"])
add("M5a", "src/minillm/engine/core.py", "changed", "diff", diff("m4", "src/minillm/engine/core.py"),
    "Each step: the scheduled requests' uncached tokens, flattened into one ragged row, plus its metadata.",
    ["What are a request's uncached tokens on a fresh prefill, a decode, and a resume after preemption?",
     "Why can the same code path later run M7's chunked prefill?"])
add("M5a", "src/minillm/model/llama.py", "changed", "diff", diff("m4", "src/minillm/model/llama.py"),
    "A ragged branch in attention: [1, heads, N, Dh] to [N, heads, Dh], then the backend's paged forward.",
    ["Why can RMSNorm, the MLP and RoPE run unchanged on a ragged row?"])
add("M5a", "tests/test_paged.py", "test", "python", working("tests/test_paged.py"),
    "Paged attention vs dense, model vs batch 1 on scattered blocks, block manager, scheduler, preemption, R3.",
    ["Which test proves a preempted request still produces exactly its batch-1 tokens?"])

add("M5a", "benchmarks/s5_kv_pressure.py", "tool", "python", working("benchmarks/s5_kv_pressure.py"),
    "S5: concurrency and block fill at fixed KV memory, the preemption cliff as the pool shrinks, decode by backend.",
    ["How many contiguous slots fit in 8 GiB for Llama-3.1-8B at max_model_len 8192? How many paged requests of ~600 tokens?",
     "Why does recompute after preemption show up as extra prefill tokens, and where does it hurt latency?"])

# ---------------- M5b / M5c (coded on the Mac; first run in the GPU session) ----------------
add("M5b", "src/minillm/attention/flashinfer_backend.py", "changed", "diff", diff("m4", "src/minillm/attention/flashinfer_backend.py"),
    "FlashInfer on decision 7's pages: the shared slot write, plan() once per step, decode or prefill wrapper.",
    ["Why is plan() keyed on the AttentionMetadata object, and what would planning per layer cost?",
     "Build kv_indptr, kv_indices and kv_last_page_len for sequences of 5 and 40 tokens with tables [3] and [8, 1, 6]."])
add("M5c", "src/minillm/attention/flash_attn_backend.py", "new", "python", working("src/minillm/attention/flash_attn_backend.py"),
    "vLLM's flash-attn on the same pages: one varlen call per layer with cu_seqlens_q, seqused_k and the block table.",
    ["Why can't upstream flash-attn 2 read these pages?",
     "flash-attn has no plan step. What does FlashInfer's plan buy it, and when does that matter (M8)?"])
add("M5b", "tests/test_kernels.py", "test", "python", working("tests/test_kernels.py"),
    "Kernels vs the reference on one GPU: a mixed ragged step, a decode-only step, and teacher-forced G1b on the 1B.",
    ["Why compare against the reference in bf16 with a tolerance instead of exact tokens?"])

# ---------------- M6 (in progress) ----------------
add("M6", "src/minillm/cache/prefix.py", "new", "python", working("src/minillm/cache/prefix.py"),
    "Decision 8's hash chain (salted SHA-256 over full blocks) and the hash -> block index.",
    ["Why must block 3's hash depend on blocks 0-2, not just its own 16 tokens?",
     "Two requests compute the same prefix in one step: which block becomes the cached copy, and what happens to the other?",
     "Write the bytes hashed for block 0 of a request with salt 'a', in order."])
add("M6", "src/minillm/engine/block_manager.py", "changed", "diff", diff("m4", "src/minillm/engine/block_manager.py"),
    "Free / held (refcounted) / cached (LRU) states; tables freed tail first; R3 extended to refcounts.",
    ["Why free a block table tail first? What breaks if block 0 is evicted before block 3?",
     "Why do cached blocks count as free capacity, and when is one actually evicted?"])
add("M6", "src/minillm/engine/scheduler.py", "changed", "diff", diff("m4", "src/minillm/engine/scheduler.py"),
    "Lookup at admission (hits shared, only the tail prefilled and budgeted); blocks named as they fill.",
    ["Why are hits counted only over the first num_tokens - 1 tokens?",
     "A cached block nobody holds is a hit: why does reviving it count against free capacity?",
     "Why are blocks named before finished requests free them?"])
add("M6", "benchmarks/s6_shared_prefixes.py", "tool", "python", working("benchmarks/s6_shared_prefixes.py"),
    "S6: TTFT vs shared-prefix fraction, caching on vs off; workload C turns with growing histories.",
    ["Why run part A one request at a time?", "What hit rate do you expect on workload C, and from which two sources?"])
add("M6", "tests/test_prefix_cache.py", "test", "python", working("tests/test_prefix_cache.py"),
    "Hash spec byte for byte, refcounts and LRU, sharing, salts, eviction, and a preempted request resuming from cache.",
    ["Which test failed when tables were freed head first, and why?"])

order = {"M0": 0, "M1": 100, "M2": 200, "M3": 300, "M4": 400, "M5a": 500, "M5b": 600, "M5c": 700, "M6": 800}
for i, m in enumerate(M):
    m["order"] = order[m["milestone"]] + i
    m["lines"] = m["code"].count("\n") + 1
    m["review"] = [{"text": t} for t in m["review"]]
    doc_id = (m["milestone"] + "--" + m["path"].replace("/", "--")).replace(".py", "")
    (OUT / f"{doc_id}.json").write_text(json.dumps(m))
    print(doc_id, m["kind"], m["lines"], "lines")
