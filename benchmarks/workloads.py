"""Workload generators (benchmarks/README.md, decision 2). Every request is token ids, so lengths are exact.

- random: fixed prompt/output lengths from random ids. No downloads; used for sweeps and smoke tests.
- A (summarize): an arXiv paper truncated to ~8K tokens -> 150 tokens out. Prefill-heavy.
- B (generate): a real user prompt of ~40-250 tokens (UltraChat-200k) -> 1,000 tokens out. Decode-heavy.
- C (agent): multi-turn sessions with shared per-agent-type prefixes, growing history and tool pauses.

Outputs are fixed with max_tokens + ignore_eos so every engine and milestone does the same work.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RequestSpec:
    prompt_token_ids: list[int]
    max_tokens: int


@dataclass
class Turn:
    new_token_ids: list[int]  # user message or tool result appended before this turn's request
    max_tokens: int
    pause_s: float  # tool time after this turn's reply, before the next turn is sent


@dataclass
class Session:
    session_id: str
    agent_type: int
    shared_prefix_ids: list[int]
    turns: list[Turn] = field(default_factory=list)


# ---------- token sources ----------


class TokenSource:
    """Draws realistic token ids: random spans of tokenized WikiText-2, or uniform ids for tiny test models."""

    def __init__(self, tokenizer=None, vocab_size: int | None = None, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        if tokenizer is not None:
            from datasets import load_dataset

            text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"])
            self.corpus = tokenizer(text[:4_000_000], add_special_tokens=False).input_ids
        elif vocab_size:
            self.corpus = None
            self.vocab_size = vocab_size
        else:
            raise ValueError("need a tokenizer or a vocab_size")

    def take(self, n: int) -> list[int]:
        if self.corpus is None:
            return [self.rng.randrange(3, self.vocab_size) for _ in range(n)]
        start = self.rng.randrange(0, len(self.corpus) - n)
        return self.corpus[start:start + n]


def random_requests(n: int, prompt_len: int, output_len: int, source: TokenSource) -> list[RequestSpec]:
    return [RequestSpec(source.take(prompt_len), output_len) for _ in range(n)]


# ---------- workloads A and B ----------


def workload_a(tokenizer, n: int, prompt_tokens: int = 8000, output_tokens: int = 150) -> list[RequestSpec]:
    """Summarize: the first `n` arXiv test papers, each truncated to `prompt_tokens` including the instruction."""
    from datasets import load_dataset

    head = tokenizer("Summarize the following paper.\n\n").input_ids
    tail = tokenizer("\n\nSummary:", add_special_tokens=False).input_ids
    body_len = prompt_tokens - len(head) - len(tail)
    out = []
    for row in load_dataset("ccdv/arxiv-summarization", split="test", streaming=True):
        body = tokenizer(row["article"], add_special_tokens=False).input_ids
        if len(body) < body_len:
            continue  # keep the prompt length fixed: skip papers shorter than the target
        out.append(RequestSpec(head + body[:body_len] + tail, output_tokens))
        if len(out) == n:
            return out
    raise ValueError(f"only {len(out)} arXiv papers have {prompt_tokens} tokens")


def workload_b(tokenizer, n: int, output_tokens: int = 1000, min_prompt: int = 40,
               max_prompt: int = 250) -> list[RequestSpec]:
    """Generate: real first-turn user prompts (UltraChat-200k test) of 40-250 tokens, fixed-length output.

    The plan names ShareGPT; its only public copy is one 670 MB JSON file. UltraChat's prompts are the same
    shape (real user requests, ~100 tokens) and stream, so only the rows used are downloaded.
    """
    from datasets import load_dataset

    out = []
    for row in load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft", streaming=True):
        ids = tokenizer(row["prompt"]).input_ids
        if min_prompt <= len(ids) <= max_prompt:
            out.append(RequestSpec(ids, output_tokens))
            if len(out) == n:
                return out
    raise ValueError(f"only {len(out)} prompts in range")


# ---------- workload C (decision 2) ----------


@dataclass
class AgentParams:
    agent_types: int = 4
    prefix_mean: int = 3500
    prefix_min: int = 2000
    prefix_max: int = 5000
    turns_mean: float = 6.0
    turns_min: int = 2
    turns_max: int = 20
    new_tokens_median: int = 300
    large_tool_output_p: float = 0.10
    large_tool_output: tuple[int, int] = (1000, 2000)
    tool_call_tokens: int = 80
    final_answer_tokens: int = 300
    pause_median_s: float = 2.0
    pause_min_s: float = 0.2
    pause_max_s: float = 30.0
    max_context: int = 8000  # turns that would exceed this are dropped (server max_model_len is 8192)


def _lognormal(rng: random.Random, median: float, sigma: float) -> float:
    return rng.lognormvariate(math.log(median), sigma)


def agent_sessions(n_sessions: int, source: TokenSource, params: AgentParams = AgentParams(),
                   seed: int = 0) -> list[Session]:
    rng = random.Random(seed)
    prefixes = []
    for _ in range(params.agent_types):
        size = int(min(params.prefix_max, max(params.prefix_min, rng.gauss(params.prefix_mean, 800))))
        prefixes.append(source.take(size))

    sigma_turns = 0.5
    mu_turns = math.log(params.turns_mean) - sigma_turns ** 2 / 2  # so the mean is turns_mean
    sessions = []
    for i in range(n_sessions):
        agent = rng.randrange(params.agent_types)
        n_turns = int(round(min(params.turns_max, max(params.turns_min, rng.lognormvariate(mu_turns, sigma_turns)))))
        context = len(prefixes[agent])
        turns: list[Turn] = []
        for t in range(n_turns):
            if t > 0 and rng.random() < params.large_tool_output_p:
                n_new = rng.randint(*params.large_tool_output)
            else:
                n_new = int(_lognormal(rng, params.new_tokens_median, 0.6))
            final = t == n_turns - 1
            out_len = params.final_answer_tokens if final else params.tool_call_tokens
            if context + n_new + out_len > params.max_context:
                break
            context += n_new + out_len
            pause = 0.0 if final else min(params.pause_max_s, max(params.pause_min_s,
                                                                   _lognormal(rng, params.pause_median_s, 1.0)))
            turns.append(Turn(source.take(n_new), out_len, round(pause, 3)))
        # A session always ends with the final answer. If the context cap cut it short, turn the last kept
        # tool call into the final answer, dropping turns until the longer answer fits.
        while turns and turns[-1].max_tokens != params.final_answer_tokens:
            if context - turns[-1].max_tokens + params.final_answer_tokens <= params.max_context:
                context += params.final_answer_tokens - turns[-1].max_tokens
                turns[-1].max_tokens = params.final_answer_tokens
            else:
                dropped = turns.pop()
                context -= len(dropped.new_token_ids) + dropped.max_tokens
        if turns:
            turns[-1].pause_s = 0.0
            sessions.append(Session(f"s{i}", agent, prefixes[agent], turns))
    return sessions


def save_sessions(sessions: list[Session], path: str | Path) -> None:
    with open(path, "w") as f:
        for s in sessions:
            f.write(json.dumps(asdict(s)) + "\n")


def load_sessions(path: str | Path) -> list[Session]:
    out = []
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        d["turns"] = [Turn(**t) for t in d["turns"]]
        out.append(Session(**d))
    return out
