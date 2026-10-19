"""M6 prefix caching (decision 8). Sharing cached blocks changes cost, never output: in fp32 a request served
partly from cache produces exactly its batch-1 tokens. R3 extended: refcounts match the block tables at every
step, and once the cache is cleared an idle engine has every block free."""

import hashlib
import struct

import pytest
import torch

from conftest import assert_idle
from minillm.cache.prefix import PrefixCache, block_hash, extend_chain, root_hash
from minillm.engine.block_manager import BlockManager
from minillm.engine.core import EngineCore
from minillm.engine.protocol import AddRequest
from minillm.generate import greedy_generate
from minillm.model.loader import from_hf_model
from test_g0_correctness import tiny_hf_model


@pytest.fixture(scope="module")
def model():
    return from_hf_model(tiny_hf_model())


def prompt(seed, n):
    return torch.randint(3, 256, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


# ---------- the hash spec ----------


def test_hash_spec_is_reproducible_byte_for_byte():
    # What a Go or Rust router must compute (decision 8).
    keys = {"model": "m@abc", "kv_dtype": "bfloat16", "layout": "NHD", "block_size": 16, "salt": ""}
    canonical = b'{"block_size":16,"kv_dtype":"bfloat16","layout":"NHD","model":"m@abc","salt":""}'
    root = hashlib.sha256(b"minillm-kv-v1\x00" + canonical).digest()
    assert root_hash(keys) == root
    toks = list(range(100, 116))
    assert block_hash(root, toks) == hashlib.sha256(root + struct.pack("<16I", *toks)).digest()


def test_chain_commits_to_every_earlier_token_and_the_salt():
    root = root_hash({"salt": ""})
    a, b = list(range(32)), [99] + list(range(1, 32))  # differ only in the first token
    ca, cb = [], []
    extend_chain(ca, root, a, 2, 16)
    extend_chain(cb, root, b, 2, 16)
    assert ca[1] != cb[1]  # same second block, different history: different name
    salted = []
    extend_chain(salted, root_hash({"salt": "t"}), a, 2, 16)
    assert salted[0] != ca[0]


def test_prefix_cache_index():
    pc = PrefixCache()
    pc.insert(b"h0", 4)
    pc.insert(b"h1", 9)
    pc.insert(b"h1", 2)  # computed twice at once: the first copy stays the cached one
    assert pc.lookup([b"h0", b"h1", b"h2"]) == [4, 9] and pc.lookup([b"h1"]) == [9]
    assert not pc.is_cached(2)
    pc.forget(4)
    assert pc.lookup([b"h0", b"h1"]) == []


# ---------- refcounts and LRU ----------


def test_refcounts_cached_blocks_and_lru_eviction():
    pc = PrefixCache()
    bm = BlockManager(4, 16, pc)
    a, b = bm.allocate(2), bm.allocate(2)
    pc.insert(b"a0", a[0])
    pc.insert(b"b0", b[0])
    a_ids, b_ids = list(a), list(b)
    bm.free(a)  # a[0] named: cached; a[1] unnamed: free
    bm.free(b)
    assert bm.num_free == 4 and bm.num_cached == 2
    bm.check([])
    shared = pc.lookup([b"b0"])
    bm.acquire(shared)
    bm.acquire(shared)  # two requests share it
    bm.check([shared, shared])
    got = bm.allocate(3)  # 2 free + evict the least recently freed cached block (a0)
    assert a_ids[0] in got and pc.lookup([b"a0"]) == [] and pc.lookup([b"b0"]) == [b_ids[0]]
    with pytest.raises(AssertionError, match="refcounts"):
        bm.check([shared, got])  # shared is held twice, not once
    bm.free(got)
    bm.free(list(shared))
    bm.free(shared)
    bm.check([])
    bm.clear_cache()
    assert bm.num_cached == 0 and len(pc) == 0 and bm.num_free == 4


# ---------- the engine ----------


def run_checked(core):
    outs = {}
    while core.has_work():
        for o in core.step().outputs:
            outs.setdefault(o.req_id, []).append(o)
        core.check_blocks()
    return outs


def tokens(outs, rid):
    return [t for o in outs[rid] for t in o.new_token_ids]


def test_shared_prefix_is_reused_and_tokens_match(model):
    core = EngineCore(model, max_batch_size=4, step_log=True)
    system = prompt(1, 48)  # three full blocks
    core.add_request(AddRequest("first", system + prompt(2, 7), 10))
    first = run_checked(core)
    a, b = system + prompt(3, 5), system + prompt(4, 20)
    core.add_request(AddRequest("a", a, 10))
    core.add_request(AddRequest("b", b, 10))
    outs = run_checked(core)
    assert outs["a"][-1].num_cached_tokens == 48 and outs["b"][-1].num_cached_tokens == 48
    assert core.step_log[-1].step > 0 and [r.prefill_tokens for r in core.step_log if r.prefill_tokens][-1] == 5 + 20
    assert tokens(first, "first") == greedy_generate(model, torch.tensor([system + prompt(2, 7)]), 10)
    assert tokens(outs, "a") == greedy_generate(model, torch.tensor([a]), 10)
    assert tokens(outs, "b") == greedy_generate(model, torch.tensor([b]), 10)
    s = core.stats()
    assert s.prefix_cache_hits == 96 and s.prefix_cache_queries == 55 + 53 + 68
    core.clear_prefix_cache()
    assert_idle(core)


def test_fully_cached_prompt_still_computes_its_last_token(model):
    core = EngineCore(model, max_batch_size=2)
    p = prompt(5, 32)  # exactly two blocks
    core.add_request(AddRequest("x", p, 5))
    run_checked(core)
    core.add_request(AddRequest("y", p, 5))
    outs = run_checked(core)
    assert outs["y"][-1].num_cached_tokens == 16  # only blocks within the first 31 tokens count
    assert tokens(outs, "y") == greedy_generate(model, torch.tensor([p]), 5)


def test_running_requests_share_blocks_and_abort_keeps_them_for_the_other(model):
    core = EngineCore(model, max_batch_size=3)
    shared = prompt(6, 40)
    core.add_request(AddRequest("seed", shared, 1))
    run_checked(core)  # names blocks 0-1 of `shared`
    core.add_request(AddRequest("u", shared + [5], 20))
    core.add_request(AddRequest("v", shared + [9], 20))
    core.step()
    u, v = core.scheduler.running
    assert u.block_ids[:2] == v.block_ids[:2]  # one copy, two references
    core.check_blocks()
    core.abort("u")
    core.check_blocks()
    outs = run_checked(core)
    assert tokens(outs, "v") == greedy_generate(model, torch.tensor([shared + [9]]), 20)[1:]  # step 1 went uncollected
    core.clear_prefix_cache()
    assert_idle(core)


def test_salt_isolates_and_off_switch_disables(model):
    core = EngineCore(model, max_batch_size=2)
    p = prompt(7, 40)
    core.add_request(AddRequest("a", p, 2, cache_salt="t1"))
    run_checked(core)
    core.add_request(AddRequest("b", p, 2, cache_salt="t2"))
    core.add_request(AddRequest("c", p, 2, cache_salt="t1"))
    outs = run_checked(core)
    assert outs["b"][-1].num_cached_tokens == 0 and outs["c"][-1].num_cached_tokens == 32
    off = EngineCore(model, max_batch_size=2, enable_prefix_caching=False)
    for rid in ("a", "b"):
        off.add_request(AddRequest(rid, p, 2))
        o = run_checked(off)
        assert o[rid][-1].num_cached_tokens == 0
    assert off.stats().kv_blocks_cached == 0


def test_eviction_under_pressure_keeps_tokens_exact(model):
    # A pool too small to keep every prefix: cached blocks are evicted LRU to admit new work.
    core = EngineCore(model, max_batch_size=2, num_kv_blocks=8, step_log=True)
    ps = [prompt(10 + i, 40) for i in range(6)]
    for i, p in enumerate(ps):
        core.add_request(AddRequest(f"e{i}", p, 12))
    outs = run_checked(core)
    for i, p in enumerate(ps):
        assert tokens(outs, f"e{i}") == greedy_generate(model, torch.tensor([p]), 12)
    core.clear_prefix_cache()
    assert_idle(core)


def test_preempted_request_resumes_from_its_own_cached_blocks(model):
    # 6 blocks. `short` (15 tokens) takes 1 and `long` (49) takes 4; short's growth takes the last free one. When
    # long fills its 4th block it needs a 5th: none is free, so the newest (long itself) is preempted, and its 4 full
    # blocks stay cached. short then needs one more block and evicts one of them: the tail (block 3), because a
    # table is freed tail first. Evicting block 0 would have made all four unreachable. long resumes from 3 blocks.
    core = EngineCore(model, max_batch_size=2, num_kv_blocks=6, step_log=True)
    ps = {"short": prompt(30, 15), "long": prompt(31, 49)}
    for rid, p in ps.items():
        core.add_request(AddRequest(rid, p, 30))
    outs = run_checked(core)
    assert core.stats().num_preemptions == 1
    assert core.stats().prefix_cache_hits == 48  # the resume found its first three blocks
    resume = [r.prefill_tokens for r in core.step_log if r.prefill_tokens][-1]
    assert resume == 49 + 16 - 48  # instead of all 65
    for rid, p in ps.items():
        assert tokens(outs, rid) == greedy_generate(model, torch.tensor([p]), 30)
    core.clear_prefix_cache()
    assert_idle(core)
