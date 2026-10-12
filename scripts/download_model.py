"""Download a Llama checkpoint into models/ (git-ignored).

The Llama repos are gated: request access on the model page first, then `hf auth login`.

    python scripts/download_model.py                      # meta-llama/Llama-3.2-1B
    python scripts/download_model.py meta-llama/Llama-3.1-8B
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo_id", nargs="?", default="meta-llama/Llama-3.2-1B")
    args = parser.parse_args()

    target = REPO / "models" / args.repo_id.split("/")[-1]
    snapshot_download(
        args.repo_id,
        local_dir=target,
        # Weights, config and tokenizer only; skip the original/ consolidated .pth copy.
        allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
    )
    print(target)


if __name__ == "__main__":
    main()
