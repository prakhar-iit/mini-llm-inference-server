"""S1 — prefill vs decode.

With a KV cache, a request has two phases with different bottlenecks:
- prefill: one pass over the whole prompt -> time to first token (TTFT). Many tokens per weight read,
  so it should be compute-bound (judge by MFU).
- decode: one token per step -> inter-token latency (ITL). Every step re-reads all weights for a
  single token, so it should be memory-bandwidth-bound (judge by MBU).

Two sweeps, batch 1, random token ids:
  A. prompt length 128 -> 16K, 16 output tokens: TTFT and prefill MFU vs prompt length.
  B. 128-token prompt, 2,000 output tokens: ITL and decode MBU vs context length.

Writes results/s1_<tag>.csv and .png. Pass --gpu for MFU/MBU (needs the GPU's peak numbers); on the
Mac the run only checks shapes, and its numbers never go in a write-up.

    python benchmarks/s1_prefill_decode.py --model-dir models/Llama-3.1-8B --gpu l4
    python benchmarks/s1_prefill_decode.py --model-dir models/Llama-3.2-1B --prompt-lengths 128,512,2048 --decode-tokens 300
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

from minillm.device import select_device
from minillm.cache.contiguous import KVCache
from minillm.model.loader import load_model

REPO = Path(__file__).resolve().parent.parent

# bf16 dense peak FLOP/s and memory bandwidth (bytes/s), as in the plan's GPU table.
GPUS = {
    "l4": (121e12, 300e9),
    "a100-40": (312e12, 1.56e12),
    "a100-80": (312e12, 2.04e12),
    "h100": (989e12, 3.35e12),
}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class CostModel:
    """FLOPs and bytes per step from the plan's formulas (matmuls and attention; norms etc. ignored)."""

    def __init__(self, model) -> None:
        c = model.config
        self.layers, self.d = c.num_hidden_layers, c.hidden_size
        # Weights touched by matmuls: every Linear in the decoder layers, plus lm_head (V × D).
        self.layer_params = sum(p.numel() for n, p in model.model.layers.named_parameters() if p.dim() == 2)
        self.head_params = c.vocab_size * c.hidden_size
        p = next(model.parameters())
        self.weight_bytes = (self.layer_params + self.head_params) * p.element_size()
        self.kv_bytes_per_token = KVCache.bytes_per_token(c, p.dtype)

    def prefill_flops(self, n: int) -> float:
        # 2 FLOPs per weight per token; lm_head runs on the last position only (last_only);
        # attention ≈ 2·n²·d·L.
        return 2 * self.layer_params * n + 2 * self.head_params + 2 * n * n * self.d * self.layers

    def decode_bytes(self, ctx: int) -> float:
        # Every decode step reads all weights once, plus the cached K/V of every earlier token.
        return self.weight_bytes + ctx * self.kv_bytes_per_token


@torch.inference_mode()
def timed_generation(model, prompt_len: int, new_tokens: int, device: torch.device) -> tuple[float, list[float]]:
    """Greedy generation from a random prompt. Returns (TTFT seconds, seconds per decode step)."""
    p = next(model.parameters())
    cache = KVCache(model.config, 1, prompt_len + new_tokens, device, p.dtype)
    ids = torch.randint(0, model.config.vocab_size, (1, prompt_len), device=device)

    sync(device)
    t0 = time.perf_counter()
    logits = model(ids, kv_cache=cache, last_only=True)
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    sync(device)
    ttft = time.perf_counter() - t0

    steps = []
    for _ in range(new_tokens - 1):
        t0 = time.perf_counter()
        logits = model(nxt, kv_cache=cache, last_only=True)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        sync(device)
        steps.append(time.perf_counter() - t0)
    return ttft, steps


def main() -> None:
    parser = argparse.ArgumentParser(description="S1: prefill vs decode with a KV cache")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--gpu", choices=sorted(GPUS), help="peak numbers for MFU/MBU")
    parser.add_argument("--prompt-lengths", default="128,256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--prefill-reps", type=int, default=3)
    parser.add_argument("--decode-prompt", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=2000)
    parser.add_argument("--tag", default=None, help="output name suffix (default: model + device)")
    args = parser.parse_args()

    info = select_device(args.device)
    model = load_model(args.model_dir, info.device, info.dtype)
    cost = CostModel(model)
    peak_flops, peak_bw = GPUS[args.gpu] if args.gpu else (None, None)
    tag = args.tag or f"{args.model_dir.name}_{args.gpu or info.device.type}"
    print(f"S1 on {info.name} ({info.device}, {info.dtype}), model={args.model_dir.name}")
    print(f"  weights read per decode step: {cost.weight_bytes / 1e9:.2f} GB; "
          f"KV cache: {cost.kv_bytes_per_token / 1024:.0f} KB/token")
    if peak_bw:
        print(f"  decode floor (weights / bandwidth): {cost.weight_bytes / peak_bw * 1e3:.2f} ms")

    timed_generation(model, 64, 4, info.device)  # warm-up

    rows = []
    print("Sweep A: prefill")
    for n in (int(x) for x in args.prompt_lengths.split(",")):
        ttfts, itls = [], []
        for _ in range(args.prefill_reps):
            ttft, steps = timed_generation(model, n, 16, info.device)
            ttfts.append(ttft)
            itls.extend(steps)
        ttft, itl = statistics.median(ttfts), statistics.median(itls)
        tflops = cost.prefill_flops(n) / ttft / 1e12
        mfu = tflops * 1e12 / peak_flops if peak_flops else None
        rows.append({"sweep": "prefill", "tokens": n, "ttft_ms": ttft * 1e3, "itl_ms": itl * 1e3,
                     "achieved_tflops": tflops, "mfu": mfu, "achieved_gbps": None, "mbu": None})
        print(f"  prompt={n:>6}  TTFT={ttft * 1e3:9.1f} ms  ITL={itl * 1e3:7.2f} ms  "
              f"{tflops:6.2f} TFLOP/s" + (f"  MFU={mfu:5.1%}" if mfu is not None else ""))

    print("Sweep B: decode")
    _, steps = timed_generation(model, args.decode_prompt, args.decode_tokens, info.device)
    for i, sec in enumerate(steps):
        ctx = args.decode_prompt + 1 + i  # tokens already in the cache when this step runs
        gbps = cost.decode_bytes(ctx) / sec / 1e9
        rows.append({"sweep": "decode", "tokens": ctx, "ttft_ms": None, "itl_ms": sec * 1e3,
                     "achieved_tflops": None, "mfu": None, "achieved_gbps": gbps,
                     "mbu": gbps * 1e9 / peak_bw if peak_bw else None})
    for ctx_row in rows[-len(steps)::max(1, len(steps) // 6)]:
        print(f"  context={ctx_row['tokens']:>6}  ITL={ctx_row['itl_ms']:7.2f} ms  "
              f"{ctx_row['achieved_gbps']:7.1f} GB/s" + (f"  MBU={ctx_row['mbu']:5.1%}" if ctx_row["mbu"] else ""))

    out_dir = REPO / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"s1_{tag}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pre = [r for r in rows if r["sweep"] == "prefill"]
    dec = [r for r in rows if r["sweep"] == "decode"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].loglog([r["tokens"] for r in pre], [r["ttft_ms"] for r in pre], marker="o")
    axes[0].set(xlabel="prompt length (tokens)", ylabel="TTFT (ms)", title="Prefill: TTFT vs prompt length")
    axes[1].plot([r["tokens"] for r in dec], [r["itl_ms"] for r in dec], linewidth=0.8)
    if peak_bw:
        axes[1].axhline(cost.weight_bytes / peak_bw * 1e3, linestyle="--", color="gray", label="decode floor")
        axes[1].legend()
    axes[1].set(xlabel="context length (tokens)", ylabel="ITL (ms)", title="Decode: ITL vs context")
    if peak_flops:
        axes[2].semilogx([r["tokens"] for r in pre], [r["mfu"] * 100 for r in pre], marker="o", label="prefill MFU")
        axes[2].semilogx([r["tokens"] for r in dec], [r["mbu"] * 100 for r in dec], linewidth=0.8, label="decode MBU")
        axes[2].set(xlabel="tokens", ylabel="% of peak", title="Utilization")
        axes[2].legend()
    else:
        axes[2].semilogx([r["tokens"] for r in pre], [r["achieved_tflops"] for r in pre], marker="o")
        axes[2].set(xlabel="prompt length", ylabel="TFLOP/s", title="Prefill throughput (no --gpu peak)")
    fig.suptitle(f"S1 prefill vs decode — {args.model_dir.name}, {info.name}")
    fig.tight_layout()
    png_path = out_dir / f"s1_{tag}.png"
    fig.savefig(png_path, dpi=120)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
