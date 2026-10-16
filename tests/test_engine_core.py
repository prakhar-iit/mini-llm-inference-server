"""M2 engine core, in-process: FIFO, one request at a time, same tokens as greedy_generate."""

import pytest
import torch

from minillm.engine.core import EngineCore
from minillm.engine.protocol import Abort, AddRequest, EngineStats, Ready, RequestOutput, StepOutputs, decode, encode
from minillm.generate import greedy_generate
from minillm.model.loader import from_hf_model
from test_g0_correctness import tiny_hf_model


@pytest.fixture(scope="module")
def model():
    return from_hf_model(tiny_hf_model())


def prompt(seed, n=10):
    return torch.randint(0, 256, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def run_to_completion(core):
    outs = {}
    while core.has_work():
        for o in core.step().outputs:
            outs.setdefault(o.req_id, []).append(o)
    return outs


def test_protocol_roundtrip():
    msgs = [AddRequest("r1", [1, 2, 3], 5, [9], timeline={"received": 1}), Abort("r1"), Ready("m", "cpu", "fp32"),
            StepOutputs([RequestOutput("r1", [7], True, "stop", {"queued": 3}, 3, 1)], EngineStats(1, 0, 4))]
    for m in msgs:
        assert decode(encode(m)) == m


def test_single_request_matches_greedy_generate(model):
    core = EngineCore(model)
    p = prompt(1)
    core.add_request(AddRequest("a", p, 20))
    outs = run_to_completion(core)["a"]
    tokens = [t for o in outs for t in o.new_token_ids]
    assert tokens == greedy_generate(model, torch.tensor([p]), 20)
    assert outs[-1].finished and outs[-1].finish_reason == "length"
    assert outs[-1].num_prompt_tokens == 10 and outs[-1].num_output_tokens == 20
    tl = outs[-1].timeline
    assert tl["queued"] <= tl["scheduled"] <= tl["first_token"] <= tl["finished"]


def test_fifo_one_at_a_time(model):
    core = EngineCore(model, step_log=True)
    for i in range(3):
        core.add_request(AddRequest(f"r{i}", prompt(10 + i), 5))
    order = []
    while core.has_work():
        for o in core.step().outputs:
            if not order or order[-1] != o.req_id:
                order.append(o.req_id)
    assert order == ["r0", "r1", "r2"]  # each request runs to completion before the next starts
    assert all(rec.num_seqs == 1 for rec in core.step_log)
    assert [rec.prefill_tokens for rec in core.step_log[:2]] == [10, 0]
    assert core.step_log[0].queue_len == 2


def test_stop_token_and_ignore_eos(model):
    p = prompt(3)
    first = greedy_generate(model, torch.tensor([p]), 8)
    core = EngineCore(model)
    core.add_request(AddRequest("stop", p, 8, stop_token_ids=[first[2]]))
    core.add_request(AddRequest("ignore", p, 8, stop_token_ids=[first[2]], ignore_eos=True))
    outs = run_to_completion(core)
    # Stops right after the first occurrence of the stop token (which may repeat earlier in the output).
    assert [t for o in outs["stop"] for t in o.new_token_ids] == first[:first.index(first[2]) + 1]
    assert outs["stop"][-1].finish_reason == "stop"
    assert [t for o in outs["ignore"] for t in o.new_token_ids] == first
    assert outs["ignore"][-1].finish_reason == "length"


def test_abort_running_and_waiting(model):
    core = EngineCore(model)
    core.add_request(AddRequest("run", prompt(4), 50))
    core.add_request(AddRequest("wait", prompt(5), 50))
    core.step()
    core.step()
    core.abort("run")
    core.abort("wait")
    core.abort("unknown")  # already finished or never existed: ignored
    outs = core.step().outputs
    assert {o.req_id: o.finish_reason for o in outs} == {"run": "abort", "wait": "abort"}
    assert core.batch is None and not core.waiting and not core.has_work()
    assert core.stats() == EngineStats(0, 0, 0)


def test_rejects_too_long(model):
    core = EngineCore(model, max_model_len=32)
    with pytest.raises(ValueError, match="max_model_len"):
        core.add_request(AddRequest("x", prompt(6, 30), 5))


# ---------- M3: static batching ----------


def test_static_batch_tokens_match_batch_1(model):
    core = EngineCore(model, max_batch_size=4, step_log=True)
    ps = [prompt(20 + i, n) for i, n in enumerate([4, 15, 9, 22, 7])]
    lens = [12, 5, 9, 3, 6]
    for i, (p, n) in enumerate(zip(ps, lens)):
        core.add_request(AddRequest(f"b{i}", p, n))
    outs = run_to_completion(core)
    for i, (p, n) in enumerate(zip(ps, lens)):
        assert [t for o in outs[f"b{i}"] for t in o.new_token_ids] == greedy_generate(model, torch.tensor([p]), n)
    log = core.step_log
    # First batch: 4 requests prefilled together; it runs until the longest (12 tokens) is done.
    assert log[0].num_seqs == 4 and log[0].prefill_tokens == 4 + 15 + 9 + 22
    assert [r.num_seqs for r in log[:12]] == [4] * 12
    # Rows that finished early are still computed: kept tokens shrink while num_seqs stays 4.
    # Request with n tokens stays active for decode steps 1..n-1 (lengths 12, 5, 9, 3).
    assert [r.decode_tokens for r in log[1:12]] == [4, 4, 3, 3, 2, 2, 2, 2, 1, 1, 1]
    # Then the fifth request runs alone.
    assert log[12].num_seqs == 1 and log[12].prefill_tokens == 7


def test_static_batch_short_request_waits_for_batch(model):
    # Head-of-line: a request arriving while a batch runs waits for the WHOLE batch, even if a slot is idle.
    core = EngineCore(model, max_batch_size=2)
    core.add_request(AddRequest("long", prompt(30), 20))
    core.add_request(AddRequest("short", prompt(31), 2))
    core.step()  # prefill both
    core.add_request(AddRequest("late", prompt(32), 2))
    outs = run_to_completion(core)
    late, long_ = outs["late"][-1].timeline, outs["long"][-1].timeline
    assert outs["short"][-1].timeline["finished"] < late["scheduled"]
    assert late["scheduled"] >= long_["finished"]


def test_abort_inside_batch(model):
    core = EngineCore(model, max_batch_size=3)
    for i in range(3):
        core.add_request(AddRequest(f"x{i}", prompt(40 + i), 30))
    core.step()
    core.abort("x1")
    rest = run_to_completion(core)
    assert rest["x1"][-1].finish_reason == "abort" and len(rest["x1"]) == 1
    assert rest["x0"][-1].finish_reason == rest["x2"][-1].finish_reason == "length"
    assert core.batch is None and core.stats() == EngineStats(0, 0, 0)
