"""S4 — mixed traffic: static batching (M3) vs continuous batching (M4) on the Mix workload under Poisson load.

Mix = workload A (long prompt, short output) and B (short prompt, long output) interleaved, 30/70 by request.
Static batching makes every request wait for the longest one in its batch, and a long B output holds a batch
for a long time; continuous batching admits each request as soon as a slot frees up. Its new cost: when an A
prompt is prefilled, every running B request pauses for that whole prefill, so B's inter-token latency spikes.

Run once per engine against a running server (same model, hardware and flags), then compare:

    git worktree add ../minillm-m3 m3          # M3's code; PYTHONPATH wins over the editable install of this checkout
    PYTHONPATH=../minillm-m3/src python -m minillm.server --model-dir models/Llama-3.1-8B --port 8001
    python -m minillm.server --model-dir models/Llama-3.1-8B --port 8000
    python benchmarks/s4_mixed_traffic.py --base-url http://127.0.0.1:8001 --tokenizer models/Llama-3.1-8B --tag m3
    python benchmarks/s4_mixed_traffic.py --base-url http://127.0.0.1:8000 --tokenizer models/Llama-3.1-8B --tag m4
    python benchmarks/s4_mixed_traffic.py --compare m3,m4

Mac smoke run (random token ids, shapes scaled down):
    python benchmarks/s4_mixed_traffic.py --base-url http://127.0.0.1:8000 --vocab-size 128256 \\
        --a-shape 1024,16 --b-shape 64,128 --rates 0.25,0.5,1 --per-rate 40 --warmup 0 --tag smoke-m4

Rates should go past the knee (2-3× it) so the run doubles as R2's load: the M4 server rejects what doesn't
fit (HTTP 503) and the goodput of admitted requests should hold.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import SLOS, RequestRecord, pct, run_requests  # noqa: E402
from workloads import TokenSource, mix, random_requests, workload_a, workload_b  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"


def shape(s: str) -> tuple[int, int]:
    prompt, output = (int(x) for x in s.split(","))
    return prompt, output


def stats(recs: list[RequestRecord], classes: list[str], warmup_s: float) -> dict:
    """One rate point: per-class latency, pooled B inter-token latency, goodput by each request's own SLO."""
    measured = [(c, r) for c, r in zip(classes, recs) if r.scheduled_s >= warmup_s]
    ok = [(c, r) for c, r in measured if r.error is None and r.done_ns is not None]
    rejected = sum(1 for _, r in measured if r.error and r.error.startswith("HTTP 503"))
    row = {"requests": len(measured), "rejected": rejected, "errors": len(measured) - len(ok) - rejected}
    if not ok:
        return row
    window = (max(r.done_ns for _, r in ok) - min(r.sent_ns for _, r in ok)) / 1e9
    meets = [r for c, r in ok if r.ttft_s <= SLOS[c][0] and (r.tpot_s is None or r.tpot_s <= SLOS[c][1])]
    row.update(window_s=window, requests_s=len(ok) / window, goodput_req_s=len(meets) / window,
               output_tok_s=sum(r.output_tokens for _, r in ok) / window)
    for cls in ("a", "b"):
        mine = [r for c, r in ok if c == cls]
        ttft = [r.ttft_s for r in mine]
        tpot = [r.tpot_s for r in mine if r.tpot_s is not None]
        row[f"{cls}_ttft_p50_ms"] = None if not ttft else pct(ttft, 50) * 1e3
        row[f"{cls}_ttft_p99_ms"] = None if not ttft else pct(ttft, 99) * 1e3
        row[f"{cls}_tpot_p99_ms"] = None if not tpot else pct(tpot, 99) * 1e3
    # B requests are the decode users; their pooled token gaps show the stall when an A prompt is prefilled.
    itl = [(y - x) / 1e9 for c, r in ok if c == "b" for x, y in zip(r.token_ns, r.token_ns[1:])]
    row["b_itl_p50_ms"] = None if not itl else pct(itl, 50) * 1e3
    row["b_itl_p99_ms"] = None if not itl else pct(itl, 99) * 1e3
    row["b_itl_max_ms"] = None if not itl else max(itl) * 1e3
    return row


def measure(args) -> None:
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        pool_a = workload_a(tok, args.pool, *args.a_shape)
        pool_b = workload_b(tok, args.pool, output_tokens=args.b_shape[1])
    else:
        src = TokenSource(vocab_size=args.vocab_size, seed=args.seed)
        pool_a = random_requests(args.pool, *args.a_shape, src)
        pool_b = random_requests(args.pool, *args.b_shape, src)

    rows = []
    for k, rate in enumerate(float(x) for x in args.rates.split(",")):
        labelled = mix(pool_a, pool_b, args.per_rate, args.frac_a, random.Random(args.seed + k))
        classes = [c for c, _ in labelled]
        recs = asyncio.run(run_requests(args.base_url, [s for _, s in labelled], rate, args.cv, args.seed + k))
        row = {"rate": rate, **stats(recs, classes, args.warmup)}
        rows.append(row)
        f = lambda key: "    -" if row.get(key) is None else f"{row[key]:7.0f}"  # noqa: E731
        print(f"  rate {rate:5.2f}/s  goodput {row.get('goodput_req_s', 0):5.2f}/s  "
              f"TTFT p99 A {f('a_ttft_p99_ms')} B {f('b_ttft_p99_ms')} ms  "
              f"B ITL p99 {f('b_itl_p99_ms')} max {f('b_itl_max_ms')} ms  rejected {row['rejected']}")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"s4_{args.tag}.csv"
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "rate", k))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def compare(tags: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def load(tag):
        with open(RESULTS / f"s4_{tag}.csv") as fh:
            return [{k: (float(v) if v not in ("", None) else None) for k, v in r.items()} for r in csv.DictReader(fh)]

    panels = [("goodput_req_s", "goodput (req/s meeting SLO)"), ("b_ttft_p99_ms", "B: p99 TTFT (ms)"),
              ("a_ttft_p99_ms", "A: p99 TTFT (ms)"), ("b_itl_p99_ms", "B: p99 inter-token latency (ms)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(20, 4.2))
    for tag in tags:
        rows = load(tag)
        for ax, (key, label) in zip(axes, panels):
            pts = [(r["rate"], r[key]) for r in rows if r.get(key) is not None]
            if pts:
                ax.plot(*zip(*pts), marker="o", label=tag)
            ax.set(xlabel="arrival rate (req/s)", ylabel=label)
    for ax, (key, _) in zip(axes, panels):
        if key.endswith("_ms"):
            ax.set_yscale("log")
        ax.legend()
    axes[1].axhline(SLOS["b"][0] * 1e3, color="gray", linestyle=":")
    axes[2].axhline(SLOS["a"][0] * 1e3, color="gray", linestyle=":")
    axes[3].axhline(SLOS["b"][1] * 1e3, color="gray", linestyle=":")
    fig.tight_layout()
    out = RESULTS / f"s4_{'_vs_'.join(tags)}.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="S4: static vs continuous batching on the Mix workload")
    p.add_argument("--base-url")
    p.add_argument("--tokenizer", help="model dir: real A (arXiv) and B (UltraChat) prompts")
    p.add_argument("--vocab-size", type=int, help="random token ids below this when no tokenizer is given")
    p.add_argument("--a-shape", type=shape, default=(8000, 150), help="A prompt,output tokens")
    p.add_argument("--b-shape", type=shape, default=(100, 1000), help="B prompt,output tokens (prompt: random only)")
    p.add_argument("--frac-a", type=float, default=0.3)
    p.add_argument("--rates", default="0.5,1,2,3,4,6,8")
    p.add_argument("--per-rate", type=int, default=1000, help="requests per rate point (decision 3: >= 1,000)")
    p.add_argument("--pool", type=int, default=200, help="distinct prompts per class")
    p.add_argument("--cv", type=float, default=1.0)
    p.add_argument("--warmup", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="run")
    p.add_argument("--compare", help="comma-separated tags of earlier runs to plot together")
    args = p.parse_args()
    if args.compare:
        compare(args.compare.split(","))
        return
    if not args.base_url or not (args.tokenizer or args.vocab_size):
        p.error("--base-url and one of --tokenizer / --vocab-size are needed to measure")
    measure(args)


if __name__ == "__main__":
    main()
