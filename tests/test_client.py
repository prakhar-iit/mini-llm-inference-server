"""The benchmark client and workload C generator (decisions 2, 3, 5): pure logic first, then against a real server."""

import asyncio
import random
import statistics

import pytest
import torch

from client import RequestRecord, arrival_offsets, pct, run_requests, run_sessions, summarize
from minillm.generate import greedy_generate
from minillm.model.loader import load_model
from test_g0_correctness import tiny_hf_model
from test_server import start_server
from workloads import AgentParams, TokenSource, agent_sessions, load_sessions, random_requests, save_sessions

# ---------- arrivals ----------


@pytest.mark.parametrize("cv", [1.0, 2.0])
def test_arrivals_have_requested_rate_and_burstiness(cv):
    t = arrival_offsets(20000, rate=5.0, cv=cv, rng=random.Random(0))
    gaps = [b - a for a, b in zip(t, t[1:])]
    mean = statistics.fmean(gaps)
    assert mean == pytest.approx(0.2, rel=0.05)
    assert statistics.pstdev(gaps) / mean == pytest.approx(cv, rel=0.08)


# ---------- summary ----------


def fake(sched, ttft_ms, n_tokens, gap_ms, error=None):
    r = RequestRecord(scheduled_s=sched, sent_ns=int(sched * 1e9))
    if error:
        r.error = error
        return r
    r.first_token_ns = r.sent_ns + int(ttft_ms * 1e6)
    r.token_ns = [r.first_token_ns + int(i * gap_ms * 1e6) for i in range(n_tokens)]
    r.output_tokens, r.done_ns = n_tokens, r.token_ns[-1]
    return r


def test_percentile_is_nearest_rank():
    xs = list(range(1, 101))
    assert (pct(xs, 50), pct(xs, 99), pct(xs, 100)) == (50, 99, 100)


def test_summary_goodput_and_warmup():
    recs = [fake(0.5, 5000, 10, 10)]  # warm-up: excluded even though it is slow
    recs += [fake(10 + i, 200, 11, 20) for i in range(8)]  # meet TTFT 1 s and TPOT 50 ms
    recs += [fake(20, 1500, 11, 20), fake(21, 200, 11, 80), fake(22, 0, 0, 0, error="HTTP 500")]
    s = summarize(recs, slo=(1.0, 0.050), warmup_s=5)
    assert s["requests"] == 11 and s["errors"] == 1
    assert s["slo_attainment"] == pytest.approx(8 / 10)
    assert s["tpot_ms"]["p50"] == pytest.approx(20, abs=0.01)
    assert s["ttft_ms"]["p99"] == pytest.approx(1500, abs=0.01)


# ---------- workload C generator ----------


def test_agent_sessions_follow_decision_2():
    src = TokenSource(vocab_size=1000, seed=1)
    sessions = agent_sessions(400, src, AgentParams(), seed=1)
    prefixes = {}
    for s in sessions:
        assert prefixes.setdefault(s.agent_type, s.shared_prefix_ids) == s.shared_prefix_ids  # shared per type
        context = len(s.shared_prefix_ids) + sum(len(t.new_token_ids) + t.max_tokens for t in s.turns)
        assert context <= 8000
        assert s.turns[-1].pause_s == 0 and s.turns[-1].max_tokens == 300
        assert all(0.2 <= t.pause_s <= 30 for t in s.turns[:-1])
    assert len(prefixes) == 4
    assert all(2000 <= len(p) <= 5000 for p in prefixes.values())
    assert 4 <= statistics.fmean(len(s.turns) for s in sessions) <= 7  # mean 6 before the context cap
    pauses = [t.pause_s for s in sessions for t in s.turns[:-1]]
    assert 1.5 <= statistics.median(pauses) <= 2.7


def test_sessions_are_reproducible_and_roundtrip(tmp_path):
    make = lambda: agent_sessions(20, TokenSource(vocab_size=500, seed=3), seed=3)  # noqa: E731
    a, b = make(), make()
    assert a == b
    save_sessions(a, tmp_path / "c.jsonl")
    assert load_sessions(tmp_path / "c.jsonl") == a


# ---------- against a real server (tiny model, CPU) ----------


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny-client")
    tiny_hf_model(max_position_embeddings=4096).save_pretrained(d)
    proc, base = start_server(d, "cpu")
    model = load_model(d, device="cpu", dtype=torch.float32)
    yield base, model
    proc.terminate()
    proc.wait(timeout=20)


def test_open_loop_requests_against_server(tiny):
    base, model = tiny
    specs = random_requests(6, prompt_len=12, output_len=9, source=TokenSource(vocab_size=256, seed=5))
    recs = asyncio.run(run_requests(base, specs, rate=50.0))
    for spec, r in zip(specs, recs):
        assert r.error is None
        assert r.output_tokens == 9 == len(r.token_ns)  # ignore_eos fixes the length
        assert r.output_token_ids == greedy_generate(model, torch.tensor([spec.prompt_token_ids]), 9)
        # Client and server share the monotonic clock: client send < server received < first token.
        assert r.sent_ns <= r.timeline["received"] <= r.timeline["first_token"] <= r.first_token_ns
    s = summarize(recs, slo=(10.0, 1.0))
    assert s["errors"] == 0 and s["slo_attainment"] == 1.0
    assert "server_queue_ms" in s


def test_sessions_carry_exact_history(tiny):
    base, _ = tiny
    params = AgentParams(agent_types=2, prefix_mean=60, prefix_min=40, prefix_max=80, turns_mean=3,
                         new_tokens_median=10, large_tool_output_p=0, tool_call_tokens=4, final_answer_tokens=6,
                         pause_median_s=0.02, pause_min_s=0.01, pause_max_s=0.05, max_context=4000)
    sessions = agent_sessions(3, TokenSource(vocab_size=256, seed=7), params, seed=7)
    recs = asyncio.run(run_sessions(base, sessions, rate=20.0))
    assert all(r.error is None for r in recs)
    for s in sessions:
        turns = sorted((r for r in recs if r.session_id == s.session_id), key=lambda r: r.turn)
        assert len(turns) == len(s.turns)
        expected = len(s.shared_prefix_ids)
        for r, t in zip(turns, s.turns):
            expected += len(t.new_token_ids)
            assert r.prompt_tokens == expected  # whole history, including earlier replies
            expected += r.output_tokens
    assert summarize(recs, slo=(10.0, 1.0))["session_goodput"] == 1.0
