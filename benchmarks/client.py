"""Open-loop benchmark client (decision 5) for any OpenAI-compatible server: ours or vLLM's.

- Arrivals follow a fixed schedule (gamma inter-arrivals: CV 1 = Poisson, CV > 1 = burstier), however slow
  the server gets. Waiting for each response before sending the next would hide queueing (decision 3).
- Every streamed token is timestamped with time.monotonic_ns, the same clock the server's timeline uses, so
  on one machine client and server stages line up directly (decision 6).
- Workload C sessions send their whole history each turn, append the reply's exact token ids, then wait out
  the tool pause (decision 2).

    python benchmarks/client.py --base-url http://127.0.0.1:8000 --workload random --vocab-size 128256 \\
        --prompt-len 128 --output-len 64 --num 200 --rate 2
    python benchmarks/client.py --base-url ... --workload c --tokenizer models/Llama-3.2-1B --num 50 --rate 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workloads import (  # noqa: E402
    AgentParams, RequestSpec, Session, TokenSource, agent_sessions, load_sessions, random_requests, save_sessions,
    workload_a, workload_b)

# Decision 3: (TTFT p99 seconds, TPOT p99 seconds) per workload.
SLOS = {"a": (3.0, 0.050), "b": (1.0, 0.050), "c": (1.0, 0.050), "random": (1.0, 0.050)}


@dataclass
class RequestRecord:
    scheduled_s: float  # offset from run start at which the schedule said to send
    sent_ns: int = 0
    first_token_ns: int | None = None
    token_ns: list[int] = field(default_factory=list)  # arrival time of every output token
    output_token_ids: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    output_tokens: int = 0
    done_ns: int | None = None
    error: str | None = None
    req_id: str | None = None
    timeline: dict | None = None  # the server's per-request timeline, when it sends one
    session_id: str | None = None
    turn: int | None = None

    @property
    def ttft_s(self) -> float | None:
        return None if self.first_token_ns is None else (self.first_token_ns - self.sent_ns) / 1e9

    @property
    def tpot_s(self) -> float | None:
        if self.done_ns is None or self.first_token_ns is None or self.output_tokens < 2:
            return None
        return (self.token_ns[-1] - self.first_token_ns) / 1e9 / (self.output_tokens - 1)


def arrival_offsets(n: int, rate: float, cv: float, rng: random.Random) -> list[float]:
    """Send times (seconds from start) with mean gap 1/rate and coefficient of variation cv (gamma gaps)."""
    if math.isinf(rate):
        return [0.0] * n
    shape = 1.0 / (cv * cv)
    t, out = 0.0, []
    for _ in range(n):
        out.append(t)
        t += rng.gammavariate(shape, 1.0 / (rate * shape))
    return out


async def send_request(http: httpx.AsyncClient, base_url: str, prompt_ids: list[int], max_tokens: int,
                       rec: RequestRecord, model: str | None = None) -> RequestRecord:
    body = {"prompt": prompt_ids, "max_tokens": max_tokens, "temperature": 0, "stream": True, "ignore_eos": True,
            "return_token_ids": True, "stream_options": {"include_usage": True}}
    if model:
        body["model"] = model
    rec.prompt_tokens = len(prompt_ids)
    rec.sent_ns = time.monotonic_ns()
    try:
        async with http.stream("POST", base_url + "/v1/completions", json=body) as r:
            if r.status_code != 200:
                rec.error = f"HTTP {r.status_code}: {(await r.aread())[:200]!r}"
                return rec
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                now = time.monotonic_ns()
                msg = json.loads(data)
                if "error" in msg:
                    rec.error = str(msg["error"])
                    return rec
                rec.req_id = msg.get("id", rec.req_id)
                if msg.get("usage"):
                    rec.output_tokens = msg["usage"]["completion_tokens"]
                for choice in msg.get("choices", []):
                    ids = choice.get("token_ids")
                    n = len(ids) if ids is not None else (1 if choice.get("text") else 0)
                    if ids:
                        rec.output_token_ids.extend(ids)
                    if n and rec.first_token_ns is None:
                        rec.first_token_ns = now
                    rec.token_ns.extend([now] * n)
                if "minillm_timeline" in msg:
                    rec.timeline = msg["minillm_timeline"]
        rec.done_ns = time.monotonic_ns()
        rec.output_tokens = rec.output_tokens or len(rec.token_ns)
    except httpx.HTTPError as e:
        rec.error = f"{type(e).__name__}: {e}"
    return rec


def _http(max_conn: int = 2048) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30),
                             limits=httpx.Limits(max_connections=max_conn, max_keepalive_connections=max_conn))


async def run_requests(base_url: str, specs: list[RequestSpec], rate: float, cv: float = 1.0, seed: int = 0,
                       model: str | None = None) -> list[RequestRecord]:
    """Open loop: request i is sent at its scheduled time whether or not earlier ones have finished."""
    offsets = arrival_offsets(len(specs), rate, cv, random.Random(seed))
    async with _http() as http:
        start = time.monotonic()
        tasks = []
        for spec, off in zip(specs, offsets):
            delay = start + off - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            rec = RequestRecord(scheduled_s=off)
            tasks.append(asyncio.create_task(send_request(http, base_url, spec.prompt_token_ids, spec.max_tokens,
                                                          rec, model)))
        return list(await asyncio.gather(*tasks))


async def run_sessions(base_url: str, sessions: list[Session], rate: float, cv: float = 1.0, seed: int = 0,
                       model: str | None = None) -> list[RequestRecord]:
    """Workload C: sessions start on an open-loop schedule; within a session, turns follow their replies."""
    offsets = arrival_offsets(len(sessions), rate, cv, random.Random(seed))
    records: list[RequestRecord] = []

    async def one(session: Session, off: float, http: httpx.AsyncClient, start: float) -> None:
        await asyncio.sleep(max(0.0, start + off - time.monotonic()))
        history = list(session.shared_prefix_ids)
        for k, turn in enumerate(session.turns):
            history += turn.new_token_ids
            rec = RequestRecord(scheduled_s=time.monotonic() - start, session_id=session.session_id, turn=k)
            records.append(rec)
            await send_request(http, base_url, history, turn.max_tokens, rec, model)
            if rec.error or len(rec.output_token_ids) != rec.output_tokens:
                rec.error = rec.error or "server did not return token ids; can't continue the session exactly"
                return
            history += rec.output_token_ids  # the agent's real reply becomes part of its context
            await asyncio.sleep(turn.pause_s)

    async with _http() as http:
        start = time.monotonic()
        await asyncio.gather(*(one(s, off, http, start) for s, off in zip(sessions, offsets)))
    return records


# ---------- summary (decision 3) ----------


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(p / 100 * len(xs)) - 1))]


def summarize(records: list[RequestRecord], slo: tuple[float, float], warmup_s: float = 0.0) -> dict:
    measured = [r for r in records if r.scheduled_s >= warmup_s]
    ok = [r for r in measured if r.error is None and r.done_ns is not None]
    if not ok:
        return {"requests": len(measured), "errors": len(measured), "note": "no successful requests"}
    ttft = [r.ttft_s for r in ok]
    tpot = [r.tpot_s for r in ok if r.tpot_s is not None]
    itl = [(b - a) / 1e9 for r in ok for a, b in zip(r.token_ns, r.token_ns[1:])]
    e2e = [(r.done_ns - r.sent_ns) / 1e9 for r in ok]
    window = (max(r.done_ns for r in ok) - min(r.sent_ns for r in ok)) / 1e9
    meets = [r for r in ok if r.ttft_s <= slo[0] and (r.tpot_s is None or r.tpot_s <= slo[1])]
    ms = lambda x: None if x is None else round(x * 1e3, 2)  # noqa: E731
    out = {
        "requests": len(measured), "errors": len(measured) - len(ok), "window_s": round(window, 2),
        "ttft_ms": {"p50": ms(pct(ttft, 50)), "p90": ms(pct(ttft, 90)), "p99": ms(pct(ttft, 99))},
        "tpot_ms": {"p50": ms(pct(tpot, 50)), "p90": ms(pct(tpot, 90)), "p99": ms(pct(tpot, 99))},
        "itl_ms": {"p50": ms(pct(itl, 50)), "p99": ms(pct(itl, 99))},
        "e2e_s": {"p50": round(pct(e2e, 50), 3), "p99": round(pct(e2e, 99), 3)},
        "output_tok_s": round(sum(r.output_tokens for r in ok) / window, 1),
        "requests_s": round(len(ok) / window, 3),
        "goodput_req_s": round(len(meets) / window, 3),
        "slo_attainment": round(len(meets) / len(ok), 3),
        "slo": {"ttft_s": slo[0], "tpot_s": slo[1]},
    }
    if any(r.timeline for r in ok):
        q = [(r.timeline["scheduled"] - r.timeline["queued"]) / 1e9 for r in ok
             if r.timeline and "scheduled" in r.timeline]
        out["server_queue_ms"] = {"p50": ms(pct(q, 50)), "p99": ms(pct(q, 99))}
    sessions: dict[str, list[RequestRecord]] = {}
    for r in measured:
        if r.session_id is not None:
            sessions.setdefault(r.session_id, []).append(r)
    if sessions:
        met = {id(r) for r in meets}
        good = [s for s, turns in sessions.items() if all(id(t) in met for t in turns)]
        out["sessions"] = len(sessions)
        out["session_goodput"] = round(len(good) / len(sessions), 3)
    if len(ok) < 1000:
        out["note"] = f"{len(ok)} requests: p99 is unreliable below 1,000 (decision 3)"
    return out


def save_records(records: list[RequestRecord], path: str | Path) -> None:
    with open(path, "w") as f:
        for r in records:
            d = asdict(r)
            d.pop("output_token_ids")  # large and only needed while a session runs
            f.write(json.dumps(d) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Open-loop benchmark client for OpenAI-compatible servers")
    p.add_argument("--base-url", required=True)
    p.add_argument("--workload", choices=["random", "a", "b", "c"], required=True)
    p.add_argument("--num", type=int, required=True, help="requests (random, a, b) or sessions (c)")
    p.add_argument("--rate", type=float, required=True, help="arrivals per second (requests, or sessions for c)")
    p.add_argument("--cv", type=float, default=1.0, help="arrival burstiness: 1 = Poisson, >1 = bursty")
    p.add_argument("--warmup", type=float, default=30.0, help="seconds of schedule excluded from the summary")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer", help="model dir whose tokenizer builds A/B/C prompts")
    p.add_argument("--vocab-size", type=int, help="random ids below this when no tokenizer is given")
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--sessions-file", help="replay workload C sessions from this JSONL")
    p.add_argument("--save-sessions", help="write the generated workload C sessions here")
    p.add_argument("--model", default=None, help="model name to send (vLLM requires it)")
    p.add_argument("--out", help="write per-request records (JSONL) here")
    args = p.parse_args()

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    source = lambda: TokenSource(tok, args.vocab_size, args.seed)  # noqa: E731

    if args.workload == "c":
        sessions = load_sessions(args.sessions_file) if args.sessions_file else agent_sessions(
            args.num, source(), AgentParams(), args.seed)
        if args.save_sessions:
            save_sessions(sessions, args.save_sessions)
        records = asyncio.run(run_sessions(args.base_url, sessions[:args.num], args.rate, args.cv, args.seed,
                                           args.model))
    else:
        if args.workload == "random":
            specs = random_requests(args.num, args.prompt_len, args.output_len, source())
        elif args.workload == "a":
            specs = workload_a(tok, args.num)
        else:
            specs = workload_b(tok, args.num)
        records = asyncio.run(run_requests(args.base_url, specs, args.rate, args.cv, args.seed, args.model))

    if args.out:
        save_records(records, args.out)
    print(json.dumps(summarize(records, SLOS[args.workload], args.warmup), indent=2))


if __name__ == "__main__":
    main()
