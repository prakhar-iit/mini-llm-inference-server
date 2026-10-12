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

# ---------------- M2 (in progress: working tree vs m1) ----------------
add("M2", "src/minillm/engine/protocol.py", "new", "python", working("src/minillm/engine/protocol.py"),
    "Messages between frontend and engine: AddRequest, Abort, Ready, StepOutputs, as msgpack tagged maps.",
    ["Why msgpack instead of pickle? Name two reasons from decision 1.",
     "Why does the engine see token ids only, never text?",
     "What does `StepOutputs.stats` carry, and who reads it?"])
add("M2", "src/minillm/engine/core.py", "new", "python", working("src/minillm/engine/core.py"),
    "The engine core: FIFO queue, one request at a time; each step is a prefill or one decode token.",
    ["Walk through `step()` for a new request, then for a running one. Where does the GPU work actually finish?",
     "Why serve one request at a time on purpose? What will S2 show as the arrival rate rises?",
     "What happens to a request's KV cache when it finishes or is aborted?",
     "Which timeline stamps does the engine add, and which come from the frontend?"])
add("M2", "src/minillm/engine/process.py", "new", "python", working("src/minillm/engine/process.py"),
    "The engine process loop: drain messages, step, send outputs; idle-poll when there's no work.",
    ["Why is the poll timeout 0 while there is work and 50 ms when idle?",
     "Why is the step log written only when the engine is idle?",
     "Why does the frontend start this process with spawn instead of fork?"])
add("M2", "src/minillm/server/engine_client.py", "new", "python", working("src/minillm/server/engine_client.py"),
    "The frontend's handle on the engine: one reader task routes outputs to per-request queues; disconnects abort.",
    ["Trace a request from `generate()` to the first yielded output.",
     "Why is the Abort sent with NOBLOCK instead of awaited?",
     "What was the R1 bug, and what does `_abandoned` fix?"])
add("M2", "src/minillm/server/api.py", "new", "python", working("src/minillm/server/api.py"),
    "OpenAI-compatible endpoints, SSE streaming, incremental detokenization, timelines, Prometheus metrics.",
    ["Write the sequence of SSE events for a 3-token streamed completion with include_usage.",
     "Why does the detokenizer hold back text ending in U+FFFD?",
     "Which request fields are rejected with 400, and why reject instead of ignoring?",
     "How is server-side TTFT computed, and how does it differ from the client's TTFT?"])
add("M2", "src/minillm/server/__main__.py", "new", "python", working("src/minillm/server/__main__.py"),
    "`python -m minillm.server`: builds the app (which starts the engine process) and runs uvicorn.",
    ["What do `--step-log` and `--timeline-log` write, and which studies read them?"])
add("M2", "tests/test_server.py", "test", "python", working("tests/test_server.py"),
    "End to end on a tiny model: G0 through the API, streaming format, FIFO order, bad requests, R1.",
    ["How does `test_concurrent_requests_are_served_fifo` prove one-at-a-time serving from timelines alone?",
     "Why did the reference first disagree with the server (hint: token 2), and which side was right?"])
add("M2", "tests/test_engine_core.py", "test", "python", working("tests/test_engine_core.py"),
    "The engine core in-process: FIFO, stop tokens, ignore_eos, aborts, timeline order.",
    ["Why does the stop-token test expect output up to the FIRST occurrence of the stop token?"])

add("M2", "benchmarks/workloads.py", "tool", "python", working("benchmarks/workloads.py"),
    "Workloads as token ids: random, A (arXiv, 8K in), B (UltraChat, ~100 in / 1,000 out), C (agent sessions).",
    ["Why are prompts token ids rather than text, and why fix outputs with ignore_eos?",
     "Workload C: how is the turns distribution set so its mean is 6? Why lognormal for pauses?",
     "What does the generator do when the context cap cuts a session short, and why?"])
add("M2", "benchmarks/client.py", "tool", "python", working("benchmarks/client.py"),
    "Open-loop client: gamma arrivals, per-token timestamps on the server's clock, sessions, SLO/goodput summary.",
    ["`arrival_offsets`: derive the gamma shape and scale that give mean 1/rate and coefficient of variation cv.",
     "Why does `run_requests` sleep until each scheduled time instead of awaiting the previous response?",
     "In `run_sessions`, why must the reply's exact token ids be appended to the history?",
     "Which requests count toward goodput, and how is session goodput different?"])
add("M2", "benchmarks/s2_overhead_queueing.py", "tool", "python", working("benchmarks/s2_overhead_queueing.py"),
    "S2: TTFT split into six stages on an idle server; queue wait vs utilization against the M/D/1 formula.",
    ["Why is the M2 engine an M/D/1 queue? What do M, D and 1 stand for here?",
     "Derive W at rho = 0.9 in units of S. Why does it blow up as rho approaches 1?",
     "The Mac smoke run measured waits below M/D/1. Give two reasons."])
add("M2", "tests/test_client.py", "test", "python", working("tests/test_client.py"),
    "Arrival statistics, summary math, workload C rules, and client runs against a real tiny server.",
    ["What does `test_open_loop_requests_against_server` prove by comparing client and server timestamps?"])

order = {"M0": 0, "M1": 100, "M2": 200}
for i, m in enumerate(M):
    m["order"] = order[m["milestone"]] + i
    m["lines"] = m["code"].count("\n") + 1
    m["review"] = [{"text": t} for t in m["review"]]
    doc_id = (m["milestone"] + "--" + m["path"].replace("/", "--")).replace(".py", "")
    (OUT / f"{doc_id}.json").write_text(json.dumps(m))
    print(doc_id, m["kind"], m["lines"], "lines")
