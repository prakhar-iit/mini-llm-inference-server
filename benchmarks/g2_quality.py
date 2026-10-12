"""G2 quality gate: WikiText-2 perplexity + GSM8K accuracy for our engine, against an HF baseline.

    # Record the Hugging Face baseline once per model (writes benchmarks/baselines/g2_<model>.json):
    python benchmarks/g2_quality.py --model-dir models/Llama-3.2-1B --engine hf --save-baseline
    # Score our engine and check it against the baseline (exit code 1 if the gate fails):
    python benchmarks/g2_quality.py --model-dir models/Llama-3.2-1B --check

Pass criterion: perplexity within 1% of the baseline, and GSM8K accuracy within two standard errors
(binomial, at the baseline's accuracy and this run's question count). Quality numbers don't depend on
the GPU, so a Mac baseline is valid; record which device produced it all the same.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from minillm.device import select_device
from minillm.evals import gsm8k_accuracy, load_gsm8k, perplexity, wikitext_token_ids
from minillm.generate import greedy_generate
from minillm.model.loader import load_model

REPO = Path(__file__).resolve().parent.parent
BASELINES = REPO / "benchmarks" / "baselines"


def paragraph_break_ids(tokenizer) -> tuple[int, ...]:
    """Every vocab token containing a blank line ("\\n\\n"; byte-level BPE spells \\n as "Ċ")."""
    tokens = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    return tuple(i for i, t in enumerate(tokens) if t and "ĊĊ" in t)


def build_engine(name: str, model_dir: Path, device: torch.device, dtype: torch.dtype, tok, max_new_tokens: int):
    """(logits_fn for perplexity, generate_fn for GSM8K) for our engine or HF's."""
    stop_ids = paragraph_break_ids(tok) + tuple(
        [tok.eos_token_id] if isinstance(tok.eos_token_id, int) else tok.eos_token_id or [])

    if name == "ours":
        model = load_model(model_dir, device, dtype)

        def logits_fn(ids):
            return model(ids.to(device))

        def generate_fn(prompt):
            ids = tok(prompt, return_tensors="pt").input_ids.to(device)
            return tok.decode(greedy_generate(model, ids, max_new_tokens, eos_token_ids=stop_ids))
    else:
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device).eval()

        @torch.inference_mode()
        def logits_fn(ids):
            return model(ids.to(device)).logits

        @torch.inference_mode()
        def generate_fn(prompt):
            ids = tok(prompt, return_tensors="pt").input_ids.to(device)
            out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                 do_sample=False, eos_token_id=list(stop_ids), pad_token_id=tok.eos_token_id)
            return tok.decode(out[0, ids.shape[1]:])

    return logits_fn, generate_fn


def git_rev() -> str:
    try:
        return subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="G2 quality gate: perplexity + GSM8K")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--engine", choices=["ours", "hf"], default="ours")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--ppl-tokens", type=int, default=32768)
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--gsm8k", type=int, default=200, help="number of GSM8K test questions (0 to skip)")
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--save-baseline", action="store_true", help="write the result as this model's baseline")
    parser.add_argument("--check", action="store_true", help="compare with the saved baseline; exit 1 on failure")
    args = parser.parse_args()

    info = select_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    logits_fn, generate_fn = build_engine(args.engine, args.model_dir, info.device, info.dtype, tok,
                                          args.max_new_tokens)
    print(f"G2 · engine={args.engine} · {args.model_dir.name} · {info.name} ({info.dtype}) · {git_rev()}")

    t0 = time.perf_counter()
    ppl = perplexity(logits_fn, wikitext_token_ids(tok, args.ppl_tokens), window=args.window)
    print(f"  WikiText-2 perplexity ({args.ppl_tokens} tokens, window {args.window}): {ppl:.4f}  "
          f"[{time.perf_counter() - t0:.0f} s]")

    result = {"model": args.model_dir.name, "engine": args.engine, "device": info.name, "dtype": str(info.dtype),
              "git": git_rev(), "ppl": ppl, "ppl_tokens": args.ppl_tokens, "window": args.window}

    if args.gsm8k:
        questions, shots = load_gsm8k(args.gsm8k, args.shots)
        t0 = time.perf_counter()

        def progress(done, correct):
            if done % 10 == 0 or done == len(questions):
                print(f"  GSM8K {done}/{len(questions)}: {correct} correct  [{time.perf_counter() - t0:.0f} s]",
                      flush=True)

        acc, records = gsm8k_accuracy(generate_fn, questions, shots, progress)
        print(f"  GSM8K accuracy ({args.gsm8k} questions, {args.shots}-shot): {acc:.1%}")
        result.update({"gsm8k_acc": acc, "gsm8k_n": args.gsm8k, "gsm8k_shots": args.shots,
                       "gsm8k_preds": [r["pred"] for r in records]})

    out_dir = REPO / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"g2_{args.model_dir.name}_{args.engine}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")

    baseline_path = BASELINES / f"g2_{args.model_dir.name}.json"
    if args.save_baseline:
        BASELINES.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(result, indent=2))
        print(f"saved baseline {baseline_path}")

    if args.check:
        base = json.loads(baseline_path.read_text())
        failures = []
        if (base["ppl_tokens"], base["window"]) != (args.ppl_tokens, args.window):
            failures.append("perplexity settings differ from the baseline's")
        elif abs(ppl / base["ppl"] - 1) > 0.01:
            failures.append(f"perplexity {ppl:.4f} vs baseline {base['ppl']:.4f} (more than 1% apart)")
        if args.gsm8k and "gsm8k_acc" in base:
            p, n = base["gsm8k_acc"], args.gsm8k
            tol = 2 * math.sqrt(max(p * (1 - p), 1 / n) / n)
            if abs(result["gsm8k_acc"] - p) > tol:
                failures.append(f"GSM8K {result['gsm8k_acc']:.1%} vs baseline {p:.1%} (tolerance ±{tol:.1%})")
            same = sum(a == b for a, b in zip(result["gsm8k_preds"], base["gsm8k_preds"]))
            print(f"  GSM8K answers identical to baseline: {same}/{min(n, base['gsm8k_n'])}")
        print("G2 PASS" if not failures else "G2 FAIL: " + "; ".join(failures))
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
