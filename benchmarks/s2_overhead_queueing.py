"""S2 — overhead and queueing, against a running server (`python -m minillm.server ...`).

Part 1, overhead: one request at a time (idle server), 1 output token. Client and server share the monotonic
clock, so each request's TTFT splits into stages:
    client→server | HTTP + tokenize | frontend→engine | queue | prefill | engine→client
Part 2, queueing: measure the service time S of a fixed request, then send Poisson load at utilization
rho = rate × S. The M2 engine serves one request at a time, so it is a single-server queue with near-constant
service time (M/D/1): mean wait W = rho·S / (2(1 − rho)), unbounded as rho → 1. Measured wait is plotted against
that prediction.

    python benchmarks/s2_overhead_queueing.py --base-url http://127.0.0.1:8000 --vocab-size 128256
    python benchmarks/s2_overhead_queueing.py --base-url ... --vocab-size 128256 --per-rate 30 --warmup 0 \\
        --output-len 8 --rhos 0.3,0.6,0.9  # quick Mac smoke run
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import SLOS, RequestRecord, _http, run_requests, send_request, summarize  # noqa: E402
from workloads import TokenSource, random_requests  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
STAGES = [  # (label, start event, end event); "client_sent"/"client_first" are the client's own stamps
    ("client→server", "client_sent", "received"),
    ("HTTP + tokenize", "received", "tokenized"),
    ("frontend→engine", "tokenized", "queued"),
    ("queue", "queued", "scheduled"),
    ("prefill", "scheduled", "first_token"),
    ("engine→client", "first_token", "client_first"),
]


async def sequential(base_url: str, specs) -> list[RequestRecord]:
    async with _http() as http:
        return [await send_request(http, base_url, s.prompt_token_ids, s.max_tokens, RequestRecord(0.0))
                for s in specs]


def stage_ms(r: RequestRecord) -> dict[str, float]:
    t = {**r.timeline, "client_sent": r.sent_ns, "client_first": r.first_token_ns}
    return {label: (t[b] - t[a]) / 1e6 for label, a, b in STAGES}


def main() -> None:
    p = argparse.ArgumentParser(description="S2: overhead breakdown and queueing vs utilization")
    p.add_argument("--base-url", required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--overhead-n", type=int, default=100)
    p.add_argument("--overhead-prompt", type=int, default=32)
    p.add_argument("--prompt-len", type=int, default=128, help="queueing part: prompt length")
    p.add_argument("--output-len", type=int, default=32, help="queueing part: output length")
    p.add_argument("--service-n", type=int, default=20)
    p.add_argument("--rhos", default="0.2,0.4,0.6,0.8,0.9,1.0,1.2")
    p.add_argument("--per-rate", type=int, default=300, help="requests per utilization point")
    p.add_argument("--warmup", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="run")
    args = p.parse_args()
    src = TokenSource(vocab_size=args.vocab_size, seed=args.seed)

    # ---- Part 1: overhead ----
    recs = asyncio.run(sequential(args.base_url, random_requests(args.overhead_n + 3, args.overhead_prompt, 1, src)))
    recs = [r for r in recs[3:] if r.error is None and r.timeline]  # first few warm up the path
    stages = {label: statistics.median(stage_ms(r)[label] for r in recs) for label, _, _ in STAGES}
    ttft = statistics.median(r.ttft_s * 1e3 for r in recs)
    print(f"Part 1: TTFT breakdown, median of {len(recs)} requests ({args.overhead_prompt}-token prompt, 1 token out)")
    for label, ms in stages.items():
        print(f"  {label:<17}{ms:9.3f} ms")
    print(f"  {'client TTFT':<17}{ttft:9.3f} ms  (non-GPU share: {(ttft - stages['prefill']) / ttft:.0%})")

    # ---- Part 2: queueing ----
    svc = asyncio.run(sequential(args.base_url, random_requests(args.service_n, args.prompt_len, args.output_len, src)))
    service_s = statistics.fmean((r.timeline["finished"] - r.timeline["scheduled"]) / 1e9 for r in svc if r.timeline)
    print(f"Part 2: service time S = {service_s * 1e3:.1f} ms ({args.prompt_len} in, {args.output_len} out); "
          f"capacity 1/S = {1 / service_s:.2f} req/s")
    rows = []
    for rho in (float(x) for x in args.rhos.split(",")):
        rate = rho / service_s
        specs = random_requests(args.per_rate, args.prompt_len, args.output_len, src)
        recs = asyncio.run(run_requests(args.base_url, specs, rate, 1.0, args.seed))
        measured = [r for r in recs if r.scheduled_s >= args.warmup and r.error is None and r.timeline]
        wait = statistics.fmean((r.timeline["scheduled"] - r.timeline["queued"]) / 1e9 for r in measured)
        predicted = rho * service_s / (2 * (1 - rho)) if rho < 1 else None
        summ = summarize(recs, SLOS["random"], args.warmup)
        rows.append({"rho": rho, "rate": rate, "mean_wait_ms": wait * 1e3,
                     "mdl_predicted_wait_ms": None if predicted is None else predicted * 1e3,
                     "ttft_p50_ms": summ["ttft_ms"]["p50"], "ttft_p99_ms": summ["ttft_ms"]["p99"],
                     "throughput_req_s": summ["requests_s"], "goodput_req_s": summ["goodput_req_s"]})
        pred = "   (unbounded)" if predicted is None else f"  M/D/1 {predicted * 1e3:8.0f} ms"
        print(f"  rho={rho:4.2f} rate={rate:5.2f}/s  mean wait {wait * 1e3:8.0f} ms{pred}  "
              f"TTFT p99 {summ['ttft_ms']['p99']:8.0f} ms  served {summ['requests_s']:.2f}/s")

    out = REPO / "results"
    out.mkdir(exist_ok=True)
    with open(out / f"s2_{args.tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "key", "value"])
        w.writerows(("overhead_ms", k, v) for k, v in stages.items())
        w.writerow(("overhead_ms", "client_ttft", ttft))
        w.writerow(("service", "service_s", service_s))
        for r in rows:
            w.writerows(("queueing", f"rho={r['rho']}:{k}", v) for k, v in r.items())
    print(f"wrote {out / f's2_{args.tag}.csv'}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 4.2))
    left = 0.0
    for label, ms in stages.items():
        a.barh([0], [ms], left=left, label=f"{label} ({ms:.2f} ms)")
        left += ms
    a.set(yticks=[], xlabel="ms", title="Part 1: where TTFT goes (idle server)")
    a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    rhos = [r["rho"] for r in rows]
    b.plot(rhos, [r["mean_wait_ms"] for r in rows], marker="o", label="measured")
    pr = [(r["rho"], r["mdl_predicted_wait_ms"]) for r in rows if r["mdl_predicted_wait_ms"] is not None]
    if pr:
        b.plot(*zip(*pr), linestyle="--", label="M/D/1 prediction")
    b.set(xlabel="utilization rho = rate × S", ylabel="mean queue wait (ms)", title="Part 2: queueing", yscale="log")
    b.legend()
    c.plot([r["rate"] for r in rows], [r["ttft_p99_ms"] for r in rows], marker="o")
    c.axhline(SLOS["random"][0] * 1e3, color="gray", linestyle=":", label="TTFT SLO")
    c.axvline(1 / service_s, color="gray", linestyle="--", label="capacity 1/S")
    c.set(xlabel="arrival rate (req/s)", ylabel="p99 TTFT (ms)", title="p99 TTFT vs load", yscale="log")
    c.legend()
    fig.tight_layout()
    fig.savefig(out / f"s2_{args.tag}.png", dpi=120)
    print(f"wrote {out / f's2_{args.tag}.png'}")


if __name__ == "__main__":
    main()
