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
                        help="requests running at once, each reserving max-model-len tokens of KV; 1 = M2 behaviour")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        help="token budget of one prefill step, padding included (default 8192; decision 9)")
    parser.add_argument("--max-waiting", type=int, default=64,
                        help="requests queued beyond the running ones before new ones get HTTP 503 (R2)")
    args = parser.parse_args()

    app = create_app(args.model_dir, args.device, args.max_model_len, args.served_model_name,
                     args.step_log, args.timeline_log, args.max_batch_size, args.max_num_batched_tokens,
                     args.max_waiting)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
