# mini-llm-inference-server

An LLM inference server built from scratch, one optimization at a time. Every milestone comes with a
study that predicts, measures and explains how that step changes latency and throughput. The single-GPU
engine is then taken distributed: tensor parallelism, cache-aware routing, prefill/decode
disaggregation, multi-node serving, and finally a 1T-parameter MoE model (Kimi K2).

Full roadmap, in build order: [`docs/distributed_inference_plan.md`](docs/distributed_inference_plan.md)
(see "Implementation order" for the step-by-step checklist and "Testing program" for the gates).

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Setup: budget, AWS G-quota, Llama access, dev env | In progress: budget and quotas (decisions 20, 21) open |
| 1a | Core engine M0–M6 (KV cache, server, batching, continuous batching, paged KV, prefix caching) → **v1** | M0–M6 built and tagged `m0`–`m6`; S0–S6 scripts ready, EC2 runs pending; R4 soak and AWS session scripts not written |
| 1b | M7–M10 (chunked prefill, CUDA graphs, quantization, speculative decoding) → v1.1 | — |
| 2 | Distributed on one node (TP, router, P/D disaggregation, KV tiering, reliability) | — |
| 3 | Multi-node cluster and cluster-scale prompt caching | — |
| 4 | Production stacks (vLLM + llm-d / Dynamo on EKS), self-host vs API cost study | — |
| 5 | Capstone: Kimi K2 (MLA + MoE, expert parallelism) | — |

## Layout

```
src/minillm/
  device.py        # device + dtype selection (cuda / cpu)
  model/           # model code: Llama first (M0), DeepSeek-V3 / Kimi K2 later (Phase 5)
  cache/           # every KV cache: contiguous (M1/M3), slots (M4), paged pool (M5), prefix cache (M6)
  attention/       # attention backends: torch reference (all devices) + FlashInfer (CUDA); paged metadata
  engine/          # scheduler, block manager, engine core (M3–M6)
  server/          # OpenAI-compatible API + streaming (M2)
tests/             # one suite, runs on every tier
benchmarks/        # workload generators (A summarize, B generate, C agent) and runners
docs/
  distributed_inference_plan.md
  course/          # the learning course (HTML) built from src/ by build.py; mirrors the artifact
  studies/         # one note per study: S0, S1, ...  (copy _TEMPLATE.md)
  decisions/       # design decision log
scripts/aws/       # instance setup and scripted benchmark sessions
```

## Development tiers

Everything runs on EC2. The MacBook is only an editor: no tests, smoke runs or studies run on it (decided 2026-11-14).

| Tier | Machine | Use it for |
|---|---|---|
| 1 | AWS g6.xlarge (1× L4) / g6.12xlarge (4× L4) | Development, `pytest`, smoke runs, CUDA path, study dry runs, multi-GPU correctness |
| 2 | AWS p4/p5 (p5en for Kimi K2) | Scripted runs only: official study numbers, multi-GPU and multi-node performance |

## Quick start (g6.xlarge)

```bash
uv venv && source .venv/bin/activate     # Python 3.12, pinned in .python-version
uv pip install -e ".[dev,cuda]"
pytest -q -rs                    # skips list unfinished milestone work ("not implemented yet: M0 ...")
python -m minillm.device        # prints the device and dtype this machine will use
hf auth login && python scripts/download_model.py   # Llama-3.2-1B into models/ (gated: request access first)
```

Run the server (OpenAI-compatible; the engine runs in its own process):

```bash
python -m minillm.server --model-dir models/Llama-3.2-1B --port 8000
curl -N localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 20, "stream": true}'
```

M5c (flash-attn) installs separately; see `pyproject.toml` (decision 7).

## Rules

- Predict before measuring; write the gap down in `docs/studies/`.
- Correctness gate at every milestone: greedy output matches Hugging Face at batch 1, and logits stay
  within tolerance under batching. Compare on the same device only.
- First-pass L4 numbers are labelled as such and never mixed with official A100/H100 numbers.
- Every change updates every doc it affects, including the course. See `CLAUDE.md`.

## References

[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and
[mini-sglang](https://github.com/sgl-project/mini-sglang) are used for comparison, not copied.
