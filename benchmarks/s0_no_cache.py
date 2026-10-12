"""S0 — cost of no cache.

Without a KV cache, generating the token at position L means a full forward pass over all L tokens.
This script measures:

1. step time vs sequence length: one forward over L tokens, for L in --lengths;
2. cumulative time to generate N tokens from a short prompt, to show the total is quadratic.

Writes results/s0_<tag>.csv and results/s0_<tag>.png. Numbers from the Mac are for checking the shape
only; the study's numbers come from the L4 (see the plan's dev tiers).

    python benchmarks/s0_no_cache.py --model-dir models/Llama-3.2-1B
    python benchmarks/s0_no_cache.py --tiny            # random tiny model, smoke test
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

from minillm.device import select_device
from minillm.model.config import LlamaConfig
from minillm.model.llama import LlamaForCausalLM
from minillm.model.loader import load_model

REPO = Path(__file__).resolve().parent.parent


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def time_forward(model, length: int, vocab: int, device: torch.device, reps: int) -> float:
    """Median seconds for one forward pass over `length` tokens."""
    ids = torch.randint(0, vocab, (1, length), device=device)
    model(ids)  # warm-up (kernel selection, allocator)
    sync(device)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(ids)
        sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


@torch.inference_mode()
def time_generation(model, prompt_len: int, new_tokens: int, vocab: int, device: torch.device) -> list[float]:
    """Cumulative seconds after each generated token (greedy, no cache)."""
    ids = torch.randint(0, vocab, (1, prompt_len), device=device)
    cumulative, t0 = [], time.perf_counter()
    for _ in range(new_tokens):
        nxt = model(ids)[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        sync(device)
        cumulative.append(time.perf_counter() - t0)
    return cumulative


def tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
                         rope_theta=10000.0)
    return LlamaForCausalLM(config).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="S0: step time vs sequence length without a KV cache")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-dir", type=Path)
    src.add_argument("--tiny", action="store_true", help="random tiny model (smoke test)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--lengths", default="128,256,512,1024,2048,4096")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--gen-prompt", type=int, default=16, help="prompt length for the cumulative run")
    parser.add_argument("--gen-tokens", type=int, default=256, help="tokens to generate in the cumulative run")
    parser.add_argument("--tag", default=None, help="output name suffix (default: model + device)")
    args = parser.parse_args()

    info = select_device(args.device)
    if args.tiny:
        model, name = tiny_model().to(info.device, info.dtype), "tiny"
    else:
        model, name = load_model(args.model_dir, info.device, info.dtype), args.model_dir.name
    vocab = model.config.vocab_size
    tag = args.tag or f"{name}_{info.device.type}"
    print(f"S0 on {info.name} ({info.device}, {info.dtype}), model={name}")

    out_dir = REPO / "results"
    out_dir.mkdir(exist_ok=True)

    step_rows = []
    for length in (int(x) for x in args.lengths.split(",")):
        sec = time_forward(model, length, vocab, info.device, args.reps)
        step_rows.append((length, sec))
        print(f"  L={length:>6}  step={sec * 1e3:9.2f} ms")

    cumulative = time_generation(model, args.gen_prompt, args.gen_tokens, vocab, info.device)
    print(f"  generated {args.gen_tokens} tokens in {cumulative[-1]:.2f} s")

    csv_path = out_dir / f"s0_{tag}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "x", "seconds"])
        w.writerows(("step_time", length, sec) for length, sec in step_rows)
        w.writerows(("cumulative", i + 1, sec) for i, sec in enumerate(cumulative))
    print(f"wrote {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot([r[0] for r in step_rows], [r[1] * 1e3 for r in step_rows], marker="o")
    a.set(xlabel="sequence length L (tokens)", ylabel="step time (ms)", title="Time per token vs length")
    b.plot(range(1, len(cumulative) + 1), cumulative)
    b.set(xlabel="tokens generated", ylabel="cumulative time (s)", title="Total generation time")
    fig.suptitle(f"S0 no KV cache — {name}, {info.name}")
    fig.tight_layout()
    png_path = out_dir / f"s0_{tag}.png"
    fig.savefig(png_path, dpi=120)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
