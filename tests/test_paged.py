"""M5a paged KV on the torch reference backend. Paging changes where k/v live, never the output: attention over
scattered blocks equals dense attention, and every request's tokens equal greedy_generate's in fp32, including
requests that were preempted and recomputed. R3: blocks are never leaked or held twice."""

import random

import pytest
import torch

from conftest import assert_idle
from minillm.attention.metadata import AttentionMetadata, SeqStep
from minillm.attention.paged_torch_ref import gather, write_kv
from minillm.attention.torch_ref import TorchRefBackend
from minillm.cache.paged import BLOCK_SIZE, PagedKVCache
from minillm.engine.block_manager import BlockManager
from minillm.engine.core import EngineCore
from minillm.engine.protocol import AddRequest
from minillm.engine.scheduler import Request, Scheduler
from minillm.generate import greedy_generate
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


# ---------- metadata and the pool ----------


def test_metadata_for_a_mixed_step():
    # Block size 4. Seq 0: 3 tokens cached in block 7, one new (a decode step). Seq 1: a fresh 5-token prompt
    # in blocks 2 and 9.
    meta = AttentionMetadata.build([SeqStep(3, 1, [7]), SeqStep(0, 5, [2, 9])], 4, "cpu")
    assert meta.query_start_loc == [0, 1, 6] and meta.seq_lens == [4, 5] and meta.num_new == [1, 5]
    assert meta.positions.tolist() == [3, 0, 1, 2, 3, 4]
    assert meta.slot_mapping.tolist() == [7 * 4 + 3, 8, 9, 10, 11, 9 * 4]
    assert meta.block_tables.tolist() == [[7, 0], [2, 9]] and meta.logits_index.tolist() == [0, 5]
    with pytest.raises(ValueError, match="can't hold"):
        AttentionMetadata.build([SeqStep(0, 5, [2])], 4, "cpu")


def test_write_then_gather_round_trip():
    pool = torch.zeros(6, 4, 2, 3)  # 6 blocks of 4 tokens, 2 KV heads, head_dim 3
    other = torch.zeros_like(pool)
    k = torch.randn(7, 2, 3)
    slots = torch.tensor([5 * 4 + i for i in range(4)] + [1 * 4 + i for i in range(3)])  # blocks 5, then 1
    write_kv(pool, other, slots, k, k)
    got = gather(pool, torch.tensor([[5, 1]]), 7)  # [1, Hkv, 7, Dh]
    torch.testing.assert_close(got[0].transpose(0, 1), k)


@torch.inference_mode()
def test_paged_attention_equals_dense():
    # Four sequences: a fresh prompt, a decode step, a prompt continued on top of cached tokens, a 1-token prompt.
    torch.manual_seed(0)
    h, hkv, dh, bs = 4, 2, 8, 4
    steps = [(0, 5), (6, 1), (7, 4), (0, 1)]  # (cached, new)
    pool_k, pool_v = torch.zeros(32, bs, hkv, dh), torch.zeros(32, bs, hkv, dh)
    free = list(range(32))
    random.Random(1).shuffle(free)
    seqs, ctx_k, ctx_v = [], [], []
    for cached, new in steps:
        n = cached + new
        blocks = [free.pop() for _ in range(-(-n // bs))]
        k, v = torch.randn(n, hkv, dh), torch.randn(n, hkv, dh)
        if cached:  # earlier steps already wrote these
            slots = torch.tensor([blocks[p // bs] * bs + p % bs for p in range(cached)])
            write_kv(pool_k, pool_v, slots, k[:cached], v[:cached])
        seqs.append(SeqStep(cached, new, blocks))
        ctx_k.append(k)
        ctx_v.append(v)
    meta = AttentionMetadata.build(seqs, bs, "cpu")
    q = torch.randn(sum(new for _, new in steps), h, dh)
    new_k = torch.cat([k[c:] for k, (c, _) in zip(ctx_k, steps)])
    new_v = torch.cat([v[c:] for v, (c, _) in zip(ctx_v, steps)])
    backend = TorchRefBackend()
    out = backend.forward(q, new_k, new_v, pool_k, pool_v, meta)
    for i, (cached, new) in enumerate(steps):
        s = meta.query_start_loc[i]
        dense = backend.prefill(q[s:s + new].transpose(0, 1)[None], ctx_k[i].transpose(0, 1)[None],
                                ctx_v[i].transpose(0, 1)[None])[0].transpose(0, 1)
        torch.testing.assert_close(out[s:s + new], dense, atol=1e-5, rtol=1e-5)


# ---------- the model on paged KV ----------


@torch.inference_mode()
def run_paged(model, prompts, steps, chunk_first=None, num_blocks=64):
    """Drive the model by hand: one ragged step for all sequences each time, blocks scattered at random and the
    pool filled with large values beforehand, so reading any slot that isn't the sequence's own changes logits.
    `chunk_first` splits the first prefill: that many tokens of every prompt, then the rest on top."""
    cache = PagedKVCache(model.config, num_blocks, "cpu", torch.float32)
    for buf in cache.k + cache.v:
        buf.fill_(1e4)
    free = list(range(num_blocks))
    random.Random(2).shuffle(free)
    tokens = [list(p) for p in prompts]
    blocks = [[] for _ in prompts]
    cached = [0] * len(prompts)
    gen = [[] for _ in prompts]
    while min(len(g) for g in gen) < steps:
        upto = [len(t) for t in tokens]
        if chunk_first and not any(cached):
            upto = [min(chunk_first, len(t)) for t in tokens]
        for i, b in enumerate(blocks):
            while len(b) * BLOCK_SIZE < upto[i]:
                b.append(free.pop())
        meta = AttentionMetadata.build([SeqStep(cached[i], upto[i] - cached[i], blocks[i])
                                        for i in range(len(tokens))], BLOCK_SIZE, "cpu")
        ids = torch.tensor([[t for i, tok in enumerate(tokens) for t in tok[cached[i]:upto[i]]]])
        logits = model(ids, kv_cache=cache.step(meta), last_index=meta.logits_index)
        assert torch.isfinite(logits).all()
        nxt = logits[:, -1].argmax(-1).tolist()
        for i in range(len(tokens)):
            if upto[i] < len(tokens[i]):  # mid-prompt chunk: no token sampled yet
                cached[i] = upto[i]
                continue
            cached[i] = len(tokens[i])
            tokens[i].append(nxt[i])
            gen[i].append(nxt[i])
    return gen


def test_ragged_paged_generation_matches_batch_1(variant):
    ps = [prompt(1, 3), prompt(2, 17), prompt(3, 1), prompt(4, 40)]  # 40 tokens spans three blocks
    for p, g in zip(ps, run_paged(variant, ps, 20)):
        assert g == greedy_generate(variant, torch.tensor([p]), 20)


def test_prompt_prefilled_in_chunks_matches(model):
    # A prompt continued on top of its own cached tokens (what M7's chunked prefill will do every step).
    ps = [prompt(5, 30), prompt(6, 21)]
    for p, g in zip(ps, run_paged(model, ps, 8, chunk_first=13)):
        assert g == greedy_generate(model, torch.tensor([p]), 8)


# ---------- block manager ----------


def test_block_manager_accounting():
    bm = BlockManager(8, 16)
    assert bm.blocks_for(1) == 1 and bm.blocks_for(16) == 1 and bm.blocks_for(17) == 2
    a, b = bm.allocate(3), bm.allocate(5)
    assert bm.num_free == 0 and len(set(a + b)) == 8
    bm.check([a, b])
    with pytest.raises(RuntimeError, match="free"):
        bm.allocate(1)
    with pytest.raises(AssertionError):
        bm.check([a, b, a[:1]])  # a block held twice
    bm.free(a)
    assert a == [] and bm.num_free == 3
    bm.check([b])


# ---------- scheduler ----------


def req(rid, n_prompt, max_tokens=4):
    return Request(rid, list(range(3, 3 + n_prompt)), max_tokens, frozenset(), {})


def prefilled(r):
    """What a step does to a request it ran: everything cached, one new token sampled."""
    r.num_cached = r.num_tokens
    r.output.append(9)


def test_admission_by_tokens_and_blocks():
    s = Scheduler(max_num_seqs=8, max_num_batched_tokens=40, blocks=BlockManager(4, 16))
    rs = [req("r0", 10), req("r1", 10), req("r2", 30), req("r3", 5)]
    for r in rs:
        s.add(r)
    out = s.schedule()
    assert [r.req_id for r in out.prefill] == ["r0", "r1"]  # 10 + 10 + 30 > 40 real tokens (no padding)
    assert [len(r.block_ids) for r in out.prefill] == [1, 1]
    out = s.schedule()
    assert [r.req_id for r in out.prefill] == ["r2"] and len(rs[2].block_ids) == 2
    out = s.schedule()  # r3 needs a block and none is free: every running request decodes instead
    assert out.prefill == [] and [r.req_id for r in out.decode] == ["r0", "r1", "r2"]


def test_admission_keeps_a_block_for_a_request_about_to_cross_a_boundary():
    s = Scheduler(max_num_seqs=8, max_num_batched_tokens=100, blocks=BlockManager(2, 16))
    full = req("full", 16)  # fills its only block exactly: its next token needs a second one
    s.add(full)
    assert s.schedule().prefill == [full] and s.blocks.num_free == 1
    prefilled(full)
    s.add(req("next", 3))
    assert s.schedule().prefill == []  # 1 block + 1 kept for `full` > 1 free


def test_head_request_is_admitted_over_budget_and_nothing_skips_it():
    s = Scheduler(max_num_seqs=4, max_num_batched_tokens=8, blocks=BlockManager(10, 16))
    s.add(req("big", 20))
    s.add(req("small", 3))
    assert [r.req_id for r in s.schedule().prefill] == ["big"]  # over budget alone: no chunked prefill before M7
    assert [r.req_id for r in s.schedule().prefill] == ["small"]


def test_preemption_takes_the_newest_and_requeues_it_first():
    bm = BlockManager(3, 16)
    s = Scheduler(max_num_seqs=4, max_num_batched_tokens=100, blocks=bm)
    old, new = req("old", 16), req("new", 10)
    s.add(old)
    s.schedule()
    prefilled(old)  # 16 cached: its next token starts block 2
    s.add(new)
    s.add(req("late", 1))
    assert [r.req_id for r in s.schedule().prefill] == ["new"]  # late: 1 block + 1 kept for old > 1 free
    prefilled(new)
    out = s.schedule()  # old crosses into a new block and takes the last free one
    assert out.preempted == [] and len(old.block_ids) == 2 and bm.num_free == 0
    old.num_cached, new.num_cached = 32, 16  # both at a block boundary, nothing free
    out = s.schedule()
    assert out.preempted == [new] and [r.req_id for r in out.decode] == ["old"] and len(old.block_ids) == 3
    assert new.block_ids == [] and new.num_cached == 0 and new.num_preemptions == 1
    assert [r.req_id for r in s.waiting] == ["new", "late"]  # back at the front, ahead of later arrivals
    assert new.uncached() == new.prompt + new.output  # recomputed in full when re-admitted
    bm.check([r.block_ids for r in s.running])


def test_remove_waiting():
    s = Scheduler(max_num_seqs=1, max_num_batched_tokens=100, blocks=BlockManager(4, 16))
    s.add(req("a", 2))
    s.add(req("b", 2))
    s.schedule()
    assert s.remove_waiting("b").req_id == "b" and s.remove_waiting("b") is None
    assert s.remove_waiting("a") is None  # running, not waiting


# ---------- engine ----------


def run_checked(core):
    """Run to completion, checking R3 after every step."""
    outs = {}
    while core.has_work():
        for o in core.step().outputs:
            outs.setdefault(o.req_id, []).append(o)
        core.check_blocks()
    return outs


def tokens(outs, rid):
    return [t for o in outs[rid] for t in o.new_token_ids]


def test_preempted_requests_still_match_batch_1(model):
    # 10 blocks = 160 tokens for 6 requests that grow to ~60 tokens each: preemption is certain.
    core = EngineCore(model, max_batch_size=6, num_kv_blocks=10, step_log=True)
    ps = [prompt(10 + i, n) for i, n in enumerate([20, 5, 33, 12, 1, 26])]
    for i, p in enumerate(ps):
        core.add_request(AddRequest(f"p{i}", p, 40))
    outs = run_checked(core)
    for i, p in enumerate(ps):
        assert tokens(outs, f"p{i}") == greedy_generate(model, torch.tensor([p]), 40)
        assert outs[f"p{i}"][-1].finish_reason == "length"
    assert sum(r.num_preempted for r in core.step_log) > 0 and core.stats().num_preemptions > 0
    assert max(r.kv_blocks_used for r in core.step_log) <= 10
    assert_idle(core)


def test_paged_runs_more_requests_than_contiguous_slots_would(model):
    # The same memory M4 gave ONE slot (max_model_len 64 = 4 blocks) runs four short requests at once.
    core = EngineCore(model, max_model_len=64, max_batch_size=4, num_kv_blocks=4, step_log=True)
    for i in range(4):
        core.add_request(AddRequest(f"s{i}", prompt(30 + i, 5), 6))
    run_checked(core)
    assert max(r.num_seqs for r in core.step_log) == 4
    assert sum(r.num_preempted for r in core.step_log) == 0
    assert_idle(core)


def test_abort_returns_blocks(model):
    core = EngineCore(model, max_batch_size=3, num_kv_blocks=12)
    for i in range(3):
        core.add_request(AddRequest(f"a{i}", prompt(40 + i, 20), 30))
    core.step()
    core.step()
    assert core.stats().kv_blocks_used == 6  # 20 prompt tokens + 1 cached output token: 2 blocks each
    core.abort("a1")
    assert core.stats().kv_blocks_used == 4
    core.check_blocks()
    run_checked(core)
    assert_idle(core)


def test_rejects_requests_bigger_than_the_pool(model):
    core = EngineCore(model, max_batch_size=2, num_kv_blocks=2)
    with pytest.raises(ValueError, match="KV blocks"):
        core.add_request(AddRequest("x", prompt(50, 20), 20))  # 40 tokens > 2 blocks × 16
