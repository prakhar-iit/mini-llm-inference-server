"""M4 continuous batching. Caches and scheduling change cost, never output: in fp32 every request's tokens equal
greedy_generate's, whatever joins or leaves the batch around it (G1a, rerun for M4).

The slot cache tests pin down `cache/slot.py` (the M4 engine's cache, now S5's contiguous baseline). The engine
tests run on whatever the engine uses now (paged KV since M5a); the paged scheduler has its own tests in
test_paged.py."""

import pytest
import torch

from minillm.engine.core import EngineCore
from conftest import assert_idle
from minillm.engine.protocol import AddRequest
from minillm.generate import greedy_generate
from minillm.cache.slot import SlotKVCache
from minillm.model.loader import from_hf_model
from test_g0_correctness import TINY_VARIANTS, tiny_hf_model


@pytest.fixture(scope="module")
def model():
    return from_hf_model(tiny_hf_model())


@pytest.fixture(params=list(TINY_VARIANTS), ids=list(TINY_VARIANTS))
def variant(request):
    return from_hf_model(tiny_hf_model(**TINY_VARIANTS[request.param]))


def prompt(seed, n=10):
    return torch.randint(3, 256, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


# ---------- the slot cache under the model ----------


@torch.inference_mode()
def prefill(model, cache, start, prompts):
    """Right-padded prefill of `prompts` into slots [start, start + len(prompts)); returns last-token logits [k, V]."""
    lens = [len(p) for p in prompts]
    ids = torch.zeros(len(prompts), max(lens), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, :len(p)] = torch.tensor(p)
    view = cache.batch(start, start + len(prompts), lens)
    return model(ids, kv_cache=view, last_index=torch.tensor(lens) - 1)[:, 0]


@torch.inference_mode()
def decode(model, cache, tokens):
    """One decode step for slots [0, len(tokens)); returns logits [n, V]."""
    view = cache.batch(0, len(tokens), [1] * len(tokens))
    return model(torch.tensor(tokens)[:, None], kv_cache=view, last_only=True)[:, 0]


def slot_cache(model, slots=4, max_len=64):
    return SlotKVCache(model.config, slots, max_len, "cpu", torch.float32)


@torch.inference_mode()
def test_ragged_prefill_and_decode_match_batch_1(variant):
    ps = [prompt(1, 3), prompt(2, 17), prompt(3, 1), prompt(4, 9)]
    cache = slot_cache(variant)
    logits = [prefill(variant, cache, 0, ps)]
    gen = [[int(t)] for t in logits[0].argmax(-1)]
    for _ in range(5):
        logits.append(decode(variant, cache, [g[-1] for g in gen]))
        for g, t in zip(gen, logits[-1].argmax(-1).tolist()):
            g.append(t)
    for i, p in enumerate(ps):
        full = variant(torch.tensor([p + gen[i][:5]]))[0, len(p) - 1:]  # no cache, batch 1
        torch.testing.assert_close(torch.stack([step[i] for step in logits]), full, atol=1e-4, rtol=1e-4)
        assert gen[i] == greedy_generate(variant, torch.tensor([p]), 6)


@torch.inference_mode()
def test_request_joins_mid_flight(model):
    a, b = prompt(5, 12), prompt(6, 4)
    cache = slot_cache(model)
    ga = [int(prefill(model, cache, 0, [a]).argmax(-1))]
    for _ in range(4):
        ga.append(int(decode(model, cache, [ga[-1]]).argmax(-1)))
    gb = [int(prefill(model, cache, 1, [b]).argmax(-1))]  # joins while a is 17 tokens long
    for _ in range(8):
        ta, tb = decode(model, cache, [ga[-1], gb[-1]]).argmax(-1).tolist()
        ga.append(ta)
        gb.append(tb)
    assert ga == greedy_generate(model, torch.tensor([a]), 13)
    assert gb == greedy_generate(model, torch.tensor([b]), 9)


@torch.inference_mode()
def test_move_compacts_and_slots_are_reused(model):
    ps = [prompt(7, 5), prompt(8, 11), prompt(9, 7)]
    cache = slot_cache(model)
    gen = [[int(t)] for t in prefill(model, cache, 0, ps).argmax(-1)]
    for _ in range(3):
        for g, t in zip(gen, decode(model, cache, [g[-1] for g in gen]).argmax(-1).tolist()):
            g.append(t)
    cache.move(2, 0)  # request 0 leaves; request 2 moves into its slot
    gen = [gen[2], gen[1]]
    new = prompt(15, 4)
    gen.append([int(prefill(model, cache, 2, [new]).argmax(-1))])  # reuses slot 2, which still holds stale KV
    for _ in range(4):
        for g, t in zip(gen, decode(model, cache, [g[-1] for g in gen]).argmax(-1).tolist()):
            g.append(t)
    assert gen[0] == greedy_generate(model, torch.tensor([ps[2]]), 8)
    assert gen[1] == greedy_generate(model, torch.tensor([ps[1]]), 8)
    assert gen[2] == greedy_generate(model, torch.tensor([new]), 5)


@torch.inference_mode()
def test_stale_slot_contents_are_never_read(model):
    # Slots are reused without clearing. Fill every buffer with large values: any read past a row's length
    # (a stale key, a padding slot, the wrong "own" position) changes the logits.
    cache = slot_cache(model)
    for buf in cache.k + cache.v:
        buf.fill_(1e4)
    ps = [prompt(10, 2), prompt(11, 14), prompt(12, 6)]
    gen = [[int(t)] for t in prefill(model, cache, 0, ps).argmax(-1)]
    for _ in range(6):
        logits = decode(model, cache, [g[-1] for g in gen])
        assert torch.isfinite(logits).all()
        for g, t in zip(gen, logits.argmax(-1).tolist()):
            g.append(t)
    for p, g in zip(ps, gen):
        assert g == greedy_generate(model, torch.tensor([p]), 7)


def test_positions_continue_per_row(model):
    cache = slot_cache(model)
    prefill(model, cache, 0, [prompt(13, 3), prompt(14, 5)])
    view = cache.batch(0, 2, [1, 1])
    assert view.positions(torch.zeros(2, 1, dtype=torch.long)).tolist() == [[3], [5]]


# ---------- the engine core ----------


def run_to_completion(core, outs=None):
    outs = {} if outs is None else outs
    while core.has_work():
        collect(core.step(), outs)
    return outs


def collect(step_out, outs):
    for o in step_out.outputs:
        outs.setdefault(o.req_id, []).append(o)


def tokens(outs, rid):
    return [t for o in outs[rid] for t in o.new_token_ids]


def test_tokens_match_batch_1_under_churn(model):
    core = EngineCore(model, max_batch_size=3, max_num_batched_tokens=64, step_log=True)
    ps = [prompt(20 + i, n) for i, n in enumerate([4, 15, 9, 22, 7, 1, 12, 5])]
    lens = [12, 5, 9, 3, 6, 10, 2, 7]
    for i, (p, n) in enumerate(zip(ps, lens)):
        core.add_request(AddRequest(f"c{i}", p, n))
    outs = run_to_completion(core)
    for i, (p, n) in enumerate(zip(ps, lens)):
        assert tokens(outs, f"c{i}") == greedy_generate(model, torch.tensor([p]), n)
    decode_steps = [r for r in core.step_log if r.prefill_tokens == 0]
    assert all(r.num_seqs == r.decode_tokens for r in decode_steps)  # a finished request leaves at once
    assert max(r.num_seqs for r in core.step_log) == 3
    assert_idle(core)


def test_late_request_joins_the_running_batch(model):
    core = EngineCore(model, max_batch_size=2, max_num_batched_tokens=64)
    core.add_request(AddRequest("long", prompt(30), 30))
    outs = {}
    collect(core.step(), outs)
    collect(core.step(), outs)
    core.add_request(AddRequest("late", prompt(31), 3))
    run_to_completion(core, outs)
    late, long_ = outs["late"][-1].timeline, outs["long"][-1].timeline
    assert late["scheduled"] < long_["finished"] and late["finished"] < long_["finished"]
    assert tokens(outs, "long") == greedy_generate(model, torch.tensor([prompt(30)]), 30)
    assert tokens(outs, "late") == greedy_generate(model, torch.tensor([prompt(31)]), 3)


def test_freed_capacity_is_refilled_next_step(model):
    core = EngineCore(model, max_batch_size=2, max_num_batched_tokens=64)
    core.add_request(AddRequest("long", prompt(32), 20))
    core.add_request(AddRequest("short", prompt(33), 2))
    core.add_request(AddRequest("next", prompt(34), 2))
    outs = run_to_completion(core)
    short, nxt, long_ = (outs[r][-1].timeline for r in ("short", "next", "long"))
    assert short["finished"] <= nxt["scheduled"] < long_["finished"]


def test_token_budget_splits_a_prefill(model):
    core = EngineCore(model, max_batch_size=4, max_num_batched_tokens=16, step_log=True)
    core.add_request(AddRequest("a", prompt(35, 10), 3))
    core.add_request(AddRequest("b", prompt(36, 10), 3))
    core.step()
    core.step()
    assert [r.prefill_tokens for r in core.step_log] == [10, 10]  # 2 × 10 > 16: one prefill each
    outs = run_to_completion(core)
    assert len(outs) == 2


def test_abort_frees_capacity(model):
    core = EngineCore(model, max_batch_size=2, max_num_batched_tokens=64)
    for i in range(3):
        core.add_request(AddRequest(f"x{i}", prompt(40 + i), 30))
    outs = {}
    collect(core.step(), outs)
    assert core.stats().num_running == 2 and core.stats().num_waiting == 1
    core.abort("x0")
    collect(core.step(), outs)  # reports the abort; x2 takes the freed place in the same step
    assert outs["x0"][-1].finish_reason == "abort"
    assert core.stats().num_running == 2 and core.stats().num_waiting == 0
    run_to_completion(core, outs)
    for i in (1, 2):
        assert outs[f"x{i}"][-1].finish_reason == "length"
        assert tokens(outs, f"x{i}") == greedy_generate(model, torch.tensor([prompt(40 + i)]), 30)


def test_kv_stats_count_tokens_held(model):
    core = EngineCore(model, max_batch_size=2, max_num_batched_tokens=64)
    core.add_request(AddRequest("a", prompt(50, 6), 10))
    core.add_request(AddRequest("b", prompt(51, 3), 10))
    core.step()  # prefill: 6 + 3 prompt tokens cached, one output token each not yet cached
    assert core.stats().kv_tokens_used == 9
    core.step()
    assert core.stats().kv_tokens_used == 11
    run_to_completion(core)
    assert_idle(core)
