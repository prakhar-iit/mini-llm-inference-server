"""S6 — shared prefixes: prefix caching on vs off, in-process (no server) to isolate the engine.

Part A, shared-prefix fraction: requests of --prompt-len tokens whose first f × prompt-len tokens are the same
for every request (one system prompt, or one paper with many questions about it), the rest unique. Requests run
one at a time, so TTFT is queue-free prefill time. With caching, a request recomputes only its uncached tokens:
TTFT should fall roughly in proportion to 1 − f, and match "off" at f = 0 (hashing is the only overhead).
Part B, workload C: agent sessions (shared per-agent-type prefix, history growing every turn), turns interleaved
round-robin across sessions, each turn appending the real reply. Every turn after the first should hit its whole
previous context, and turns of a new session hit its agent type's prefix.

    python benchmarks/s6_shared_prefixes.py --model-dir models/Llama-3.1-8B --gpu a100-80
    python benchmarks/s6_shared_prefixes.py --model-dir models/Llama-3.2-1B --prompt-len 1024 --requests 8 \\
        --sessions 6 --agent-scale 0.2 --kv-gb 0.5  # Mac smoke run
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workloads import AgentParams, TokenSource, agent_sessions  # noqa: E402

from minillm.cache.paged import PagedKVCache  # noqa: E402
from minillm.device import select_device  # noqa: E402
from minillm.engine.core import EngineCore  # noqa: E402
from minillm.engine.protocol import AddRequest  # noqa: E402
from minillm.model.loader import load_model  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"


def one(core: EngineCore, rid: str, prompt: list[int], max_tokens: int) -> tuple[float, int, list[int]]:
    """Run one request alone: (TTFT seconds from queueing, prompt tokens served from cache, output tokens)."""
    core.add_request(AddRequest(rid, prompt, max_tokens, ignore_eos=True))
    out_tokens, last = [], None
    while core.has_work():
        for o in core.step().outputs:
            out_tokens += o.new_token_ids
            last = o
    tl = last.timeline
    return (tl["first_token"] - tl["queued"]) / 1e9, last.num_cached_tokens, out_tokens


def engine(model, args, caching: bool) -> EngineCore:
    blocks = PagedKVCache.blocks_for_bytes(model.config, next(model.parameters()).dtype, int(args.kv_gb * 2**30))
    return EngineCore(model, args.max_model_len, max_batch_size=1, num_kv_blocks=blocks,
                      enable_prefix_caching=caching, model_id=args.model_dir.name)


def main() -> None:
    ap = argparse.ArgumentParser(description="S6: prefix caching on shared prefixes")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--gpu", default=None, help="label only")
    ap.add_argument("--kv-gb", type=float, default=8.0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--prompt-len", type=int, default=8000, help="part A: tokens per request (workload A shape)")
    ap.add_argument("--fractions", default="0,0.25,0.5,0.75,0.9")
    ap.add_argument("--requests", type=int, default=20, help="part A: requests per fraction (the first is cold)")
    ap.add_argument("--output-len", type=int, default=8)
    ap.add_argument("--sessions", type=int, default=24, help="part B: workload C sessions")
    ap.add_argument("--agent-scale", type=float, default=1.0, help="part B: scale prefix and turn sizes (Mac smoke)")
    ap.add_argument("--skip", default="", help="comma-separated parts to skip: a,b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    info = select_device(args.device)
    model = load_model(args.model_dir, info.device, info.dtype)
    vocab = model.config.vocab_size
    tag = args.tag or f"{args.model_dir.name}_{args.gpu or info.device.type}"
    RESULTS.mkdir(exist_ok=True)
    skip = set(args.skip.split(","))
    one(engine(model, args, False), "warmup", [5] * 64, 2)

    rows_a = []
    if "a" not in skip:
        print(f"S6 part A on {info.name}: {args.prompt_len}-token prompts, {args.requests} requests per point, one at a time")
        rng = random.Random(args.seed)
        for f in (float(x) for x in args.fractions.split(",")):
            shared = [rng.randrange(3, vocab) for _ in range(int(f * args.prompt_len))]
            prompts = [shared + [rng.randrange(3, vocab) for _ in range(args.prompt_len - len(shared))]
                       for _ in range(args.requests)]
            row = {"fraction": f}
            for mode, caching in (("off", False), ("on", True)):
                core = engine(model, args, caching)
                runs = [one(core, f"a{i}", p, args.output_len) for i, p in enumerate(prompts)]
                warm = runs[1:]  # the first request fills the cache
                row[f"ttft_{mode}_ms"] = statistics.fmean(t for t, _, _ in warm) * 1e3
                row[f"cached_{mode}"] = statistics.fmean(c for _, c, _ in warm)
                if caching:
                    s = core.stats()
                    row["hit_rate"] = s.prefix_cache_hits / s.prefix_cache_queries
            row["speedup"] = row["ttft_off_ms"] / row["ttft_on_ms"]
            rows_a.append(row)
            print(f"  shared {f:4.0%}  TTFT off {row['ttft_off_ms']:9.1f} ms  on {row['ttft_on_ms']:9.1f} ms  "
                  f"({row['speedup']:.2f}×; ideal ≈ {1 / max(1e-9, 1 - f):.2f}×)  cached {row['cached_on']:.0f} tok, "
                  f"hit rate {row['hit_rate']:.1%}")
        with open(RESULTS / f"s6_{tag}_fraction.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_a[0]))
            w.writeheader()
            w.writerows(rows_a)

    rows_b = []
    if "b" not in skip:
        sc = args.agent_scale
        base = AgentParams()
        params = replace(base, prefix_mean=int(base.prefix_mean * sc), prefix_min=int(base.prefix_min * sc),
                         prefix_max=int(base.prefix_max * sc), new_tokens_median=max(8, int(base.new_tokens_median * sc)),
                         large_tool_output=(int(1000 * sc), int(2000 * sc)), tool_call_tokens=max(4, int(80 * sc)),
                         final_answer_tokens=max(4, int(300 * sc)), max_context=int(base.max_context * sc))
        sessions = agent_sessions(args.sessions, TokenSource(vocab_size=vocab, seed=args.seed), params, args.seed)
        print(f"S6 part B: {len(sessions)} workload C sessions, {sum(len(s.turns) for s in sessions)} turns, "
              f"round-robin, one turn at a time")
        for mode, caching in (("off", False), ("on", True)):
            core = engine(model, args, caching)
            history = {s.session_id: list(s.shared_prefix_ids) for s in sessions}
            k = 0
            while any(k < len(s.turns) for s in sessions):
                for s in sessions:
                    if k >= len(s.turns):
                        continue
                    turn = s.turns[k]
                    history[s.session_id] += turn.new_token_ids
                    ttft, cached, out = one(core, f"{s.session_id}-{k}", history[s.session_id], turn.max_tokens)
                    rows_b.append({"mode": mode, "session": s.session_id, "turn": k,
                                   "prompt_tokens": len(history[s.session_id]), "cached_tokens": cached,
                                   "ttft_ms": ttft * 1e3})
                    history[s.session_id] += out
                k += 1
            mine = [r for r in rows_b if r["mode"] == mode]
            st = core.stats()
            rate = st.prefix_cache_hits / st.prefix_cache_queries if caching and st.prefix_cache_queries else 0.0
            print(f"  caching {mode:<3}  TTFT mean {statistics.fmean(r['ttft_ms'] for r in mine):8.1f} ms  "
                  f"p99 {sorted(r['ttft_ms'] for r in mine)[int(0.99 * (len(mine) - 1))]:8.1f} ms  "
                  f"prefilled {sum(r['prompt_tokens'] - r['cached_tokens'] for r in mine):8d} tok  hit rate {rate:.1%}")
        with open(RESULTS / f"s6_{tag}_agent.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_b[0]))
            w.writeheader()
            w.writerows(rows_b)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    if rows_a:
        fr = [r["fraction"] for r in rows_a]
        a.plot(fr, [r["ttft_off_ms"] for r in rows_a], marker="o", label="caching off")
        a.plot(fr, [r["ttft_on_ms"] for r in rows_a], marker="o", label="caching on")
        a.plot(fr, [rows_a[0]["ttft_off_ms"] * (1 - f) for f in fr], linestyle="--", color="gray", label="∝ 1 − f")
        a.set(xlabel="shared-prefix fraction", ylabel="TTFT (ms)", title="Part A: TTFT vs shared fraction")
        a.legend()
    if rows_b:
        for mode in ("off", "on"):
            mine = [r for r in rows_b if r["mode"] == mode]
            b.scatter([r["prompt_tokens"] for r in mine], [r["ttft_ms"] for r in mine], s=8, label=f"caching {mode}")
        b.set(xlabel="turn prompt tokens (history)", ylabel="TTFT (ms)", title="Part B: workload C turns")
        b.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / f"s6_{tag}.png", dpi=120)
    print(f"wrote {RESULTS / f's6_{tag}.png'}")


if __name__ == "__main__":
    main()
