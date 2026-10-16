"""S3 — batch-size sweep and the cost of static batching. Runs in-process (no server) to isolate the engine.

Part A, batch sweep: B requests of workload-B shape (100-token prompt) decode together. A decode step reads
all weights once whatever B is, so while memory-bound the step time stays nearly flat and throughput grows
~linearly with B. The knee comes where arithmetic intensity (FLOPs per byte read) reaches the GPU's balance
point (plan's GPU table: ~400 FLOP/B on the L4, ~150-300 on A100/H100).

Part B, mixed lengths: requests with output lengths from 16 to 512 tokens go through the M3 engine core
(static batches). A batch runs until its longest request finishes; rows that finished early are still
computed. Reported: the fraction of computed rows that produced a kept token, and how long short requests
wait for long ones.

    python benchmarks/s3_batch_sweep.py --model-dir models/Llama-3.1-8B --gpu a100-80
    python benchmarks/s3_batch_sweep.py --model-dir models/Llama-3.2-1B --batches 1,2,4,8,16 --decode-steps 32 \\
        --mixed-requests 32  # Mac smoke run
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_prefill_decode import GPUS, CostModel, sync  # noqa: E402

from minillm.device import select_device  # noqa: E402
from minillm.engine.core import EngineCore  # noqa: E402
from minillm.engine.protocol import AddRequest  # noqa: E402
from minillm.model.kv_cache import KVCache  # noqa: E402
from minillm.model.loader import load_model  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


@torch.inference_mode()
def decode_step_time(model, batch: int, prompt_len: int, steps: int, device) -> tuple[float, float]:
    """(prefill seconds for the batch, median decode-step seconds) with equal-length random prompts."""
    p = next(model.parameters())
    cache = KVCache(model.config, batch, prompt_len + steps + 1, device, p.dtype)
    ids = torch.randint(0, model.config.vocab_size, (batch, prompt_len), device=device)
    sync(device)
    t0 = time.perf_counter()
    nxt = model(ids, kv_cache=cache, last_only=True)[:, -1].argmax(-1, keepdim=True)
    sync(device)
    prefill = time.perf_counter() - t0
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        nxt = model(nxt, kv_cache=cache, last_only=True)[:, -1].argmax(-1, keepdim=True)
        sync(device)
        times.append(time.perf_counter() - t0)
    return prefill, statistics.median(times[2:] if len(times) > 4 else times)


def main() -> None:
    ap = argparse.ArgumentParser(description="S3: batch-size sweep and static-batching waste")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--gpu", choices=sorted(GPUS), help="peak numbers for MBU/MFU and the balance point")
    ap.add_argument("--batches", default="1,2,4,8,16,32,64")
    ap.add_argument("--prompt-len", type=int, default=100)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--mixed-batch", type=int, default=16, help="part B: max_batch_size of the engine")
    ap.add_argument("--mixed-requests", type=int, default=128)
    ap.add_argument("--mixed-min", type=int, default=16)
    ap.add_argument("--mixed-max", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    info = select_device(args.device)
    model = load_model(args.model_dir, info.device, info.dtype)
    cost = CostModel(model)
    peak_flops, peak_bw = GPUS[args.gpu] if args.gpu else (None, None)
    tag = args.tag or f"{args.model_dir.name}_{args.gpu or info.device.type}"
    decode_step_time(model, 1, 16, 4, info.device)  # warm-up

    # ---- Part A ----
    print(f"S3 part A on {info.name}: decode step vs batch size ({args.prompt_len}-token prompts)")
    rows = []
    for b in (int(x) for x in args.batches.split(",")):
        prefill, step = decode_step_time(model, b, args.prompt_len, args.decode_steps, info.device)
        ctx = args.prompt_len + args.decode_steps // 2
        flops = 2 * (cost.layer_params + cost.head_params) * b + 4 * ctx * cost.d * cost.layers * b
        bytes_read = cost.weight_bytes + b * ctx * cost.kv_bytes_per_token
        row = {"batch": b, "step_ms": step * 1e3, "tok_s": b / step, "prefill_ms": prefill * 1e3,
               "intensity_flop_per_byte": flops / bytes_read,
               "mbu": bytes_read / step / peak_bw if peak_bw else None,
               "mfu": flops / step / peak_flops if peak_flops else None}
        rows.append(row)
        extra = f"  MBU {row['mbu']:5.1%}  MFU {row['mfu']:5.1%}" if peak_bw else ""
        print(f"  B={b:>3}  step {step * 1e3:8.2f} ms  {b / step:9.1f} tok/s  "
              f"intensity {row['intensity_flop_per_byte']:7.1f} FLOP/B{extra}")
    if peak_bw:
        print(f"  balance point of this GPU: {peak_flops / peak_bw:.0f} FLOP/B")

    # ---- Part B ----
    rng = random.Random(args.seed)
    lens = [rng.randint(args.mixed_min, args.mixed_max) for _ in range(args.mixed_requests)]
    core = EngineCore(model, max_batch_size=args.mixed_batch, step_log=True)
    for i, n in enumerate(lens):
        core.add_request(AddRequest(f"m{i}", [rng.randrange(3, model.config.vocab_size)
                                              for _ in range(args.prompt_len)], n, ignore_eos=True))
    finished = {}
    t0 = time.perf_counter()
    while core.has_work():
        for o in core.step().outputs:
            if o.finished:
                finished[o.req_id] = o
    wall = time.perf_counter() - t0
    log = [r for r in core.step_log if r.prefill_tokens == 0]
    computed, kept = sum(r.num_seqs for r in log), sum(r.decode_tokens for r in log)
    # All requests are queued at t=0, so a request's wait is the time every batch before it took to finish,
    # and each of those batches lasted as long as its LONGEST request.
    e2e = {rid: (o.timeline["finished"] - o.timeline["queued"]) / 1e9 for rid, o in finished.items()}
    wait = {rid: (o.timeline["scheduled"] - o.timeline["queued"]) / 1e9 for rid, o in finished.items()}
    batches = [lens[i:i + args.mixed_batch] for i in range(0, len(lens), args.mixed_batch)]
    longest_share = sum(max(b) for b in batches) / sum(sum(b) / len(b) for b in batches)
    print(f"S3 part B: {args.mixed_requests} requests, outputs {args.mixed_min}-{args.mixed_max} tokens, "
          f"static batches of {args.mixed_batch}")
    print(f"  decode rows computed {computed}, tokens kept {kept}: {kept / computed:.1%} useful "
          f"({1 - kept / computed:.1%} spent on rows that already finished)")
    print(f"  throughput {sum(lens) / wall:.1f} tok/s over {wall:.1f} s")
    waits = [w for w in wait.values() if w > 0]
    if waits:
        print(f"  queued requests waited median {statistics.median(waits):.2f} s, max {max(waits):.2f} s; each batch "
              f"lasted {longest_share:.2f}× its average request (it runs until its longest finishes)")

    out = REPO / "results"
    out.mkdir(exist_ok=True)
    with open(out / f"s3_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / f"s3_{tag}_mixed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["req", "output_tokens", "queue_wait_s", "queued_to_finished_s"])
        w.writerows((f"m{i}", n, wait[f"m{i}"], e2e[f"m{i}"]) for i, n in enumerate(lens))
    print(f"wrote {out / f's3_{tag}.csv'} and {out / f's3_{tag}_mixed.csv'}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 4.2))
    bs = [r["batch"] for r in rows]
    a.plot(bs, [r["tok_s"] for r in rows], marker="o", label="measured")
    a.plot(bs, [rows[0]["tok_s"] * x for x in bs], linestyle="--", color="gray", label="linear from B=1")
    a.set(xscale="log", yscale="log", xlabel="batch size", ylabel="decode tokens/s", title="Throughput vs batch")
    a.legend()
    b.plot(bs, [r["step_ms"] for r in rows], marker="o")
    b.set(xscale="log", xlabel="batch size", ylabel="decode step (ms)", title="Step time (ITL) vs batch")
    c.scatter(lens, [e2e[f"m{i}"] for i in range(len(lens))], s=10)
    c.set(xlabel="output tokens requested", ylabel="queued → finished (s)",
          title=f"Static batches of {args.mixed_batch}: everyone waits for the longest")
    fig.tight_layout()
    fig.savefig(out / f"s3_{tag}.png", dpi=120)
    print(f"wrote {out / f's3_{tag}.png'}")


if __name__ == "__main__":
    main()
