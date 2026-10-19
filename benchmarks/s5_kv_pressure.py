"""S5 — KV pressure: contiguous slots (M4) vs a paged pool (M5), in-process (no server) to isolate the engine.

Part A, memory at a fixed KV budget: workload-B-shaped requests (100-token prompts, outputs uniform between
--out-min and --out-max), all queued at once.
  - Contiguous (M4) reserves max_model_len per request, so it runs budget ÷ (max_model_len × KV bytes/token)
    requests at a time, whatever their real length. That count is exact arithmetic; the run below with the
    batch capped at that count reproduces its concurrency on the same engine.
  - Paged (M5) holds ceil(tokens / 16) blocks per request. Measured: running requests per decode step, the
    share of held block slots that hold a real token (the rest is the half-filled last block), throughput.
Part B, pressure: the same requests with the pool shrunk step by step. Once a decode step finds no free block,
the newest request is preempted and later recomputed: preemptions, recomputed tokens, TTFT and inter-token
latency show the cliff.
Part C, kernels (M5b/M5c, GPU): decode step time at batch × context for the torch reference, FlashInfer and
flash-attn on the same pages, and M4's contiguous slots. Backends that aren't built yet are reported as skipped.

    python benchmarks/s5_kv_pressure.py --model-dir models/Llama-3.1-8B --kv-gb 8 --gpu a100-80
    python benchmarks/s5_kv_pressure.py --model-dir models/Llama-3.2-1B --kv-gb 0.25 --max-model-len 2048 \\
        --requests 32 --out-max 256 --fractions 1,0.5,0.25,0.125 --batches 1,8 --contexts 256,1024  # Mac smoke run
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

from minillm.attention import get_backend  # noqa: E402
from minillm.cache.contiguous import KVCache  # noqa: E402
from minillm.cache.paged import BLOCK_SIZE, PagedKVCache  # noqa: E402
from minillm.cache.slot import SlotKVCache  # noqa: E402
from minillm.device import select_device  # noqa: E402
from minillm.engine.core import EngineCore  # noqa: E402
from minillm.engine.protocol import AddRequest  # noqa: E402
from minillm.model.loader import load_model  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(p / 100 * len(xs)) - 1))]


def run(model, specs, max_model_len: int, max_seqs: int, num_blocks: int) -> dict:
    """All requests queued at t=0, run to completion. Token times are taken as each step returns."""
    core = EngineCore(model, max_model_len, step_log=True, max_batch_size=max_seqs, num_kv_blocks=num_blocks,
                      max_num_batched_tokens=max(8192, max_model_len))
    for i, (prompt, n) in enumerate(specs):
        core.add_request(AddRequest(f"r{i}", prompt, n, ignore_eos=True))
    times: dict[str, list[int]] = {}
    first: dict[str, int] = {}
    t0 = time.perf_counter_ns()
    while core.has_work():
        out = core.step()
        now = time.perf_counter_ns()
        for o in out.outputs:
            if o.new_token_ids:
                times.setdefault(o.req_id, []).append(now)
                first.setdefault(o.req_id, now - t0)
    wall = (time.perf_counter_ns() - t0) / 1e9
    core.check_blocks()  # R3 after every run
    log = core.step_log
    decode = [r for r in log if r.prefill_tokens == 0]
    itl = [(b - a) / 1e9 for ts in times.values() for a, b in zip(ts, ts[1:])]
    out_tokens = sum(n for _, n in specs)
    prompt_tokens = sum(len(p) for p, _ in specs)
    held = [r.kv_tokens_used / (r.kv_blocks_used * BLOCK_SIZE) for r in decode if r.kv_blocks_used]
    return {
        "max_seqs": max_seqs, "num_blocks": num_blocks, "wall_s": wall, "output_tok_s": out_tokens / wall,
        "mean_running": statistics.fmean(r.num_seqs for r in decode), "max_running": max(r.num_seqs for r in decode),
        "block_fill": statistics.fmean(held) if held else None,
        "peak_blocks_used": max(r.kv_blocks_used for r in log),
        "preemptions": sum(r.num_preempted for r in log),
        "recomputed_tokens": sum(r.prefill_tokens for r in log) - prompt_tokens,
        "ttft_p50_s": pct(list(first.values()), 50) / 1e9, "ttft_p99_s": pct(list(first.values()), 99) / 1e9,
        "itl_p50_ms": pct(itl, 50) * 1e3, "itl_p99_ms": pct(itl, 99) * 1e3, "itl_max_ms": max(itl) * 1e3,
    }


@torch.inference_mode()
def decode_step_paged(model, batch: int, ctx: int, steps: int, device) -> float:
    """Median decode step (s) with `batch` sequences of `ctx` cached tokens each, through the model's backend."""
    core = EngineCore(model, ctx + steps + 1, max_batch_size=batch, step_log=True,
                      num_kv_blocks=batch * -(-(ctx + steps + 1) // BLOCK_SIZE), max_num_batched_tokens=batch * ctx)
    for i in range(batch):
        core.add_request(AddRequest(f"k{i}", [random.randrange(3, 1000) for _ in range(ctx)], steps + 1,
                                    ignore_eos=True))
    while core.has_work():
        core.step()
    d = [r.duration_ns / 1e9 for r in core.step_log if r.prefill_tokens == 0]
    return statistics.median(d[2:] if len(d) > 4 else d)


@torch.inference_mode()
def decode_step_contiguous(model, batch: int, ctx: int, steps: int, device) -> float:
    """The same through M4's slots: one contiguous [batch, Hkv, max_len, Dh] buffer per layer."""
    p = next(model.parameters())
    cache = SlotKVCache(model.config, batch, ctx + steps + 1, device, p.dtype)
    ids = torch.randint(3, 1000, (batch, ctx), device=device)
    nxt = model(ids, kv_cache=cache.batch(0, batch, [ctx] * batch), last_only=True)[:, -1].argmax(-1, keepdim=True)
    times = []
    for _ in range(steps):
        sync(device)
        t0 = time.perf_counter()
        nxt = model(nxt, kv_cache=cache.batch(0, batch, [1] * batch), last_only=True)[:, -1].argmax(-1, keepdim=True)
        sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times[2:] if len(times) > 4 else times)


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="S5: KV pressure, contiguous vs paged")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--gpu", choices=sorted(GPUS), help="for the bandwidth line in part C")
    ap.add_argument("--kv-gb", type=float, default=8.0, help="part A's KV budget")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--requests", type=int, default=256)
    ap.add_argument("--prompt-len", type=int, default=100)
    ap.add_argument("--out-min", type=int, default=16)
    ap.add_argument("--out-max", type=int, default=1000)
    ap.add_argument("--max-seqs", type=int, default=256, help="paged: cap on running requests")
    ap.add_argument("--fractions", default="1,0.5,0.25,0.125,0.0625", help="part B: pool sizes, share of part A's")
    ap.add_argument("--backends", default="torch_ref,flashinfer,flash_attn")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--contexts", default="512,4096")
    ap.add_argument("--decode-steps", type=int, default=16)
    ap.add_argument("--skip", default="", help="comma-separated parts to skip: a,b,c")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    info = select_device(args.device)
    model = load_model(args.model_dir, info.device, info.dtype)
    cfg = model.config
    tag = args.tag or f"{args.model_dir.name}_{args.gpu or info.device.type}"
    RESULTS.mkdir(exist_ok=True)
    skip = set(args.skip.split(","))
    bpt = KVCache.bytes_per_token(cfg, info.dtype)
    budget = int(args.kv_gb * 2**30)
    blocks = PagedKVCache.blocks_for_bytes(cfg, info.dtype, budget)
    rng = random.Random(args.seed)
    specs = [([rng.randrange(3, cfg.vocab_size) for _ in range(args.prompt_len)], rng.randint(args.out_min, args.out_max))
             for _ in range(args.requests)]
    random.seed(args.seed)

    if "a" not in skip:
        slots = budget // (args.max_model_len * bpt)
        mean_len = args.prompt_len + (args.out_min + args.out_max) / 2
        print(f"S5 part A on {info.name}: {args.kv_gb} GiB of KV = {blocks} blocks = {blocks * BLOCK_SIZE} tokens "
              f"({bpt // 1024} KB/token)")
        print(f"  contiguous: {slots} slots of {args.max_model_len} tokens; a request averages ~{mean_len:.0f} tokens, "
              f"so ~{mean_len / args.max_model_len:.1%} of the reservation holds real tokens")
        if slots < 1:
            raise SystemExit("budget too small for one contiguous slot: raise --kv-gb or lower --max-model-len")
        rows = []
        for mode, max_seqs in (("contiguous-equivalent", slots), ("paged", args.max_seqs)):
            r = {"mode": mode, **run(model, specs, args.max_model_len, max_seqs, blocks)}
            rows.append(r)
            print(f"  {mode:<22} running mean {r['mean_running']:6.1f} max {r['max_running']:4d}  "
                  f"{r['output_tok_s']:8.1f} tok/s  preemptions {r['preemptions']}"
                  + (f"  block fill {r['block_fill']:.1%}" if r["block_fill"] else ""))
        print(f"  paged / contiguous throughput: {rows[1]['output_tok_s'] / rows[0]['output_tok_s']:.2f}×")
        write_csv(RESULTS / f"s5_{tag}_memory.csv", rows)

    pressure = []
    if "b" not in skip:
        print(f"S5 part B: shrink the pool ({args.requests} requests, cap {args.max_seqs} running)")
        need = -(-(args.prompt_len + args.out_max) // BLOCK_SIZE)
        for frac in (float(x) for x in args.fractions.split(",")):
            n = max(need, int(blocks * frac))
            r = {"fraction": frac, **run(model, specs, args.max_model_len, args.max_seqs, n)}
            pressure.append(r)
            print(f"  pool {frac:6.3f} ({n:6d} blocks)  running mean {r['mean_running']:6.1f}  "
                  f"preemptions {r['preemptions']:5d}  recomputed {r['recomputed_tokens']:7d} tok  "
                  f"{r['output_tok_s']:8.1f} tok/s  TTFT p99 {r['ttft_p99_s']:7.2f} s  ITL p99 {r['itl_p99_ms']:7.1f} "
                  f"max {r['itl_max_ms']:7.1f} ms")
        write_csv(RESULTS / f"s5_{tag}_pressure.csv", pressure)

    kernels = []
    if "c" not in skip:
        print("S5 part C: decode step time by backend (ms)")
        cost = CostModel(model)
        layers = [layer.self_attn for layer in model.model.layers]
        original = layers[0].backend
        for name in ["contiguous"] + args.backends.split(","):
            for b in (int(x) for x in args.batches.split(",")):
                for ctx in (int(x) for x in args.contexts.split(",")):
                    try:
                        if name == "contiguous":
                            step = decode_step_contiguous(model, b, ctx, args.decode_steps, info.device)
                        else:
                            backend = get_backend(name)
                            for a in layers:
                                a.backend = backend
                            step = decode_step_paged(model, b, ctx, args.decode_steps, info.device)
                    except (NotImplementedError, RuntimeError, ValueError, ImportError) as e:
                        print(f"  {name:<12} skipped: {e}")
                        break
                    finally:
                        for a in layers:
                            a.backend = original
                    bytes_read = cost.weight_bytes + b * ctx * cost.kv_bytes_per_token
                    mbu = bytes_read / step / GPUS[args.gpu][1] if args.gpu else None
                    kernels.append({"backend": name, "batch": b, "context": ctx, "step_ms": step * 1e3, "mbu": mbu})
                    print(f"  {name:<12} B={b:<3} ctx={ctx:<6} {step * 1e3:8.2f} ms"
                          + (f"  MBU {mbu:5.1%}" if mbu else ""))
                else:
                    continue
                break
        if kernels:
            write_csv(RESULTS / f"s5_{tag}_kernels.csv", kernels)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if pressure:
        fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 4.2))
        fr = [r["fraction"] for r in pressure]
        a.plot(fr, [r["output_tok_s"] for r in pressure], marker="o")
        a.set(xscale="log", xlabel="pool size (share of part A's)", ylabel="output tok/s", title="Throughput")
        b.plot(fr, [r["preemptions"] for r in pressure], marker="o", label="preemptions")
        b.plot(fr, [r["recomputed_tokens"] / max(1, args.prompt_len) for r in pressure], marker="s",
               label="recomputed tokens ÷ prompt length")
        b.set(xscale="log", xlabel="pool size", title="Preemption")
        b.legend()
        c.plot(fr, [r["ttft_p99_s"] * 1e3 for r in pressure], marker="o", label="TTFT p99 (ms)")
        c.plot(fr, [r["itl_p99_ms"] for r in pressure], marker="s", label="ITL p99 (ms)")
        c.set(xscale="log", yscale="log", xlabel="pool size", title="Latency cliff")
        c.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / f"s5_{tag}.png", dpi=120)
        print(f"wrote {RESULTS / f's5_{tag}.png'}")


if __name__ == "__main__":
    main()
