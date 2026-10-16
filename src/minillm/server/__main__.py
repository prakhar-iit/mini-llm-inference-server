"""Start the server: the API frontend in this process, the engine core in a child process.

    python -m minillm.server --model-dir models/Llama-3.2-1B --port 8000
"""

from __future__ import annotations

import argparse

import uvicorn

from .api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="minillm OpenAI-compatible server")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--max-model-len", type=int, default=None, help="default: min(model limit, 8192)")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--step-log", default=None, help="write the engine step log (JSONL) here")
    parser.add_argument("--timeline-log", default=None, help="write per-request timelines (JSONL) here")
    parser.add_argument("--max-batch-size", type=int, default=8,
                        help="static batching (M3): requests prefilled and decoded together; 1 = M2 behaviour")
    args = parser.parse_args()

    app = create_app(args.model_dir, args.device, args.max_model_len, args.served_model_name,
                     args.step_log, args.timeline_log, args.max_batch_size)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
