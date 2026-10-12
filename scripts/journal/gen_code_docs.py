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
add("M1", "src/minillm/model/kv_cache.py", "new", "python", working("src/minillm/model/kv_cache.py"),
    "Contiguous per-layer K/V buffers: `append` writes this step's k/v, `advance` moves the write position.",
    ["Compute `bytes_per_token` for Llama-3.1-8B (32 layers, 8 KV heads, head_dim 128) in bf16.",
     "Why does `append` not advance `seq_len`, and who calls `advance`?",
     "What goes wrong in this design when two sequences in a batch have different lengths? (M5 fixes it.)"])
add("M1", "src/minillm/model/llama.py", "changed", "diff", diff(R, "src/minillm/model/llama.py"),
    "The cache threaded through attention, layers and model; positions continue from `seq_len`; `last_only` logits.",
    ["Why is k cached after RoPE instead of before?",
     "What positions does a decode step use, and where are they computed?",
     "How big would full logits be for a 16K-token prompt, and what does `last_only` change?"])
add("M1", "src/minillm/attention/torch_ref.py", "changed", "diff", diff(R, "src/minillm/attention/torch_ref.py"),
    "Attention for queries on top of a cache: three cases by Tq vs Tk.",
    ["Draw the mask from `tril(diagonal=tk - tq)` for Tq=3, Tk=6. Which query sees which keys?",
     "Why does a single decode query need no mask at all?"])
add("M1", "src/minillm/generate.py", "changed", "diff", diff(R, "src/minillm/generate.py"),
    "`greedy_generate`: one prefill call, then one-token decode calls against the cache.",
    ["Count the forward calls for a 100-token prompt and 50 new tokens, with and without the cache.",
     "Why does the loop return before running a forward pass for the last token?"])
add("M1", "src/minillm/evals.py", "new", "python", working("src/minillm/evals.py"),
    "G2 quality eval: WikiText-2 perplexity and few-shot GSM8K.",
    ["Why is perplexity of uniform logits equal to the vocabulary size?",
     "Why are the window's first tokens scored with less context, and why is that fine for a regression check?",
     "Why compute log-softmax in fp32?"])
add("M1", "benchmarks/s1_prefill_decode.py", "tool", "python", working("benchmarks/s1_prefill_decode.py"),
    "S1 study script: TTFT and prefill MFU vs prompt length; ITL and decode MBU vs context.",
    ["Check `CostModel.prefill_flops` against the plan's formulas. What does it leave out?",
     "Why does `decode_bytes` grow with context, and by how much per token for the 1B model?"])
add("M1", "tests/test_kv_cache.py", "test", "python", working("tests/test_kv_cache.py"),
    "The cache must change cost, never output: decode, chunked prompts and generation all match a full pass and HF.",
    ["`test_prompt_in_chunks_matches_full_pass` exercises which backend branch, and which later milestones need it?"])

order = {"M0": 0, "M1": 100}
for i, m in enumerate(M):
    m["order"] = order[m["milestone"]] + i
    m["lines"] = m["code"].count("\n") + 1
    m["review"] = [{"text": t} for t in m["review"]]
    doc_id = (m["milestone"] + "--" + m["path"].replace("/", "--")).replace(".py", "")
    (OUT / f"{doc_id}.json").write_text(json.dumps(m))
    print(doc_id, m["kind"], m["lines"], "lines")
