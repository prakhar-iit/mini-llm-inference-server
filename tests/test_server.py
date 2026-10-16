"""M2 end to end: a real server (frontend process + engine process, ZMQ between them) on a tiny model.

G0 through the API: tokens the server streams must equal greedy_generate's. R1: a client that disconnects
mid-stream gets its request aborted and the engine goes idle.
"""

import asyncio
import json
import socket
import subprocess
import sys
import time

import httpx
import pytest
import torch

from conftest import real_model_dir
from minillm.generate import greedy_generate
from minillm.model.loader import load_model
from test_g0_correctness import G0_PROMPTS, tiny_hf_model


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(model_dir, device, extra=()):
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "minillm.server", "--model-dir", str(model_dir), "--port",
                             str(port), "--device", device, *extra])
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 300
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            if httpx.get(base + "/health", timeout=1).status_code == 200:
                return proc, base
        except httpx.TransportError:
            pass
        time.sleep(0.3)
    proc.terminate()
    raise TimeoutError("server did not start")


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    tiny_hf_model(max_position_embeddings=4096).save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def server(tiny_dir, tmp_path_factory):
    logs = tmp_path_factory.mktemp("logs")
    # Batch size 1: these tests pin down M2's one-at-a-time serving; batching is tested separately.
    proc, base = start_server(tiny_dir, "cpu", ["--timeline-log", str(logs / "timeline.jsonl"),
                                                "--max-batch-size", "1"])
    yield base, logs
    proc.terminate()
    proc.wait(timeout=20)


@pytest.fixture(scope="module")
def reference(tiny_dir):
    model = load_model(tiny_dir, device="cpu", dtype=torch.float32)
    # Same stopping rule as the server: the config's EOS ids (token 2 for HF's default tiny config).
    return lambda ids, n: greedy_generate(model, torch.tensor([ids]), n, eos_token_ids=model.config.eos_token_ids)


def sse_events(lines):
    for line in lines:
        if line.startswith("data: "):
            data = line[len("data: "):]
            if data == "[DONE]":
                return
            yield json.loads(data)


PROMPT = [5, 17, 99, 3, 42, 250, 7, 7, 120, 64]


def test_models_and_health(server):
    base, _ = server
    assert httpx.get(base + "/health").json() == {"status": "ok"}
    models = httpx.get(base + "/v1/models").json()["data"]
    assert models[0]["max_model_len"] == 4096


def test_completion_matches_greedy(server, reference):
    base, _ = server
    r = httpx.post(base + "/v1/completions", json={"prompt": PROMPT, "max_tokens": 24, "temperature": 0,
                                                   "return_token_ids": True}, timeout=60).json()
    choice = r["choices"][0]
    assert choice["token_ids"] == reference(PROMPT, 24)
    n = len(choice["token_ids"])
    assert choice["finish_reason"] == ("length" if n == 24 else "stop")
    assert r["usage"] == {"prompt_tokens": 10, "completion_tokens": n, "total_tokens": 10 + n}
    tl = r["minillm_timeline"]
    stages = ["received", "tokenized", "queued", "scheduled", "first_token", "finished"]
    assert [tl[s] for s in stages] == sorted(tl[s] for s in stages)


def test_streaming_matches_greedy(server, reference):
    base, logs = server
    body = {"prompt": PROMPT, "max_tokens": 12, "stream": True, "return_token_ids": True,
            "stream_options": {"include_usage": True}}
    with httpx.stream("POST", base + "/v1/completions", json=body, timeout=60) as r:
        events = list(sse_events(r.iter_lines()))
    token_chunks = [e for e in events if e["choices"]]
    assert all(len(e["choices"][0]["token_ids"]) == 1 for e in token_chunks)  # one token per chunk
    assert [e["choices"][0]["token_ids"][0] for e in token_chunks] == reference(PROMPT, 12)
    assert token_chunks[-1]["choices"][0]["finish_reason"] in ("length", "stop")
    assert "minillm_timeline" in token_chunks[-1]
    assert events[-1]["usage"]["completion_tokens"] == len(token_chunks)
    rows = [json.loads(line) for line in (logs / "timeline.jsonl").read_text().splitlines()]
    assert any(row["req_id"] == events[0]["id"] for row in rows)


def test_concurrent_requests_are_served_fifo(server, reference):
    base, _ = server
    prompts = [[i, 2 * i + 1, 3, 4, 5] for i in range(1, 5)]

    async def go():
        async with httpx.AsyncClient(timeout=60) as c:
            return await asyncio.gather(*[
                c.post(base + "/v1/completions", json={"prompt": p, "max_tokens": 30, "return_token_ids": True})
                for p in prompts])

    results = [r.json() for r in asyncio.run(go())]
    for p, r in zip(prompts, results):
        assert r["choices"][0]["token_ids"] == reference(p, 30)
    # One at a time: each request is scheduled only after the previous one finished.
    tls = sorted((r["minillm_timeline"] for r in results), key=lambda t: t["scheduled"])
    assert all(a["finished"] <= b["scheduled"] for a, b in zip(tls, tls[1:]))


@pytest.mark.parametrize("body,message", [
    ({"prompt": PROMPT, "temperature": 0.7}, "greedy"),
    ({"prompt": PROMPT, "max_tokens": 5000}, "maximum context length"),
    ({"prompt": "hello"}, "no tokenizer"),
    ({"prompt": [PROMPT, PROMPT]}, "one prompt"),
    ({"prompt": PROMPT, "n": 2}, "n must be 1"),
])
def test_rejects_bad_requests(server, body, message):
    base, _ = server
    r = httpx.post(base + "/v1/completions", json=body, timeout=30)
    assert r.status_code == 400
    assert message in r.json()["error"]["message"]


def test_chat_needs_a_template(server):
    base, _ = server
    r = httpx.post(base + "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400 and "chat template" in r.json()["error"]["message"]


def metric(base, name):
    for line in httpx.get(base + "/metrics").text.splitlines():
        if line.startswith(name + " ") or line.startswith(name + "{"):
            return float(line.rsplit(" ", 1)[1])
    return None


def test_r1_disconnect_aborts_request(server, reference):
    base, _ = server
    aborts_before = metric(base, 'minillm:request_success_total{finished_reason="abort"}') or 0
    body = {"prompt": PROMPT, "max_tokens": 4000, "ignore_eos": True, "stream": True}
    with httpx.stream("POST", base + "/v1/completions", json=body, timeout=60) as r:
        for i, _ in enumerate(r.iter_lines()):
            if i >= 6:
                break  # leaving the block closes the connection mid-stream
    deadline = time.time() + 20
    while time.time() < deadline:
        aborts = metric(base, 'minillm:request_success_total{finished_reason="abort"}') or 0
        if aborts == aborts_before + 1 and metric(base, "minillm:num_requests_running") == 0:
            break
        time.sleep(0.2)
    assert aborts == aborts_before + 1, "request was not aborted after the client disconnected"
    assert metric(base, "minillm:num_requests_running") == 0
    assert metric(base, "minillm:kv_cache_tokens_used") == 0
    # The server keeps working afterwards.
    r = httpx.post(base + "/v1/completions", json={"prompt": PROMPT, "max_tokens": 5, "return_token_ids": True})
    assert r.json()["choices"][0]["token_ids"] == reference(PROMPT, 5)


def test_metrics_count_tokens(server):
    base, _ = server
    assert metric(base, "minillm:generation_tokens_total") > 0
    assert metric(base, "minillm:time_to_first_token_seconds_count") > 0


@pytest.mark.slow
def test_llama_32_1b_text_through_api():
    """G0 through the API on the real model: text in, the server's tokens equal greedy_generate's."""
    model_dir = real_model_dir()
    if model_dir is None:
        pytest.skip("Llama-3.2-1B not downloaded (scripts/download_model.py)")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    proc, base = start_server(model_dir, "auto")
    try:
        got = []
        for p in G0_PROMPTS:
            r = httpx.post(base + "/v1/completions", json={"prompt": p, "max_tokens": 32, "return_token_ids": True},
                           timeout=300).json()
            got.append((p, r["choices"][0]["token_ids"], r["choices"][0]["text"]))
    finally:
        proc.terminate()
        proc.wait(timeout=30)

    from minillm.device import select_device

    info = select_device("auto")
    model = load_model(model_dir, info.device, info.dtype)
    for p, ids, text in got:
        expected = greedy_generate(model, tok(p, return_tensors="pt").input_ids.to(info.device), 32,
                                   eos_token_ids=model.config.eos_token_ids)
        assert ids == expected
        assert text == tok.decode(expected, skip_special_tokens=True)


def test_static_batching_through_the_server(tiny_dir, reference):
    """M3 end to end: concurrent requests are prefilled together, and each still gets its batch-1 tokens."""
    proc, base = start_server(tiny_dir, "cpu", ["--max-batch-size", "4"])
    try:
        prompts = [[i, 2 * i + 1, 3, 4, 5, 6][: 2 + i] for i in range(1, 5)]  # ragged lengths 3..6

        async def go():
            async with httpx.AsyncClient(timeout=60) as c:
                # A first request occupies the engine so the next four queue up and form one batch.
                first = asyncio.create_task(c.post(base + "/v1/completions",
                                                   json={"prompt": [9, 9], "max_tokens": 1500, "ignore_eos": True}))
                await asyncio.sleep(0.3)
                rest = await asyncio.gather(*[
                    c.post(base + "/v1/completions",
                           json={"prompt": p, "max_tokens": 20, "ignore_eos": True, "return_token_ids": True})
                    for p in prompts])
                await first
                return rest

        results = [r.json() for r in asyncio.run(go())]
        model = load_model(tiny_dir, device="cpu", dtype=torch.float32)
        for p, r in zip(prompts, results):
            assert r["choices"][0]["token_ids"] == greedy_generate(model, torch.tensor([p]), 20)
        assert len({r["minillm_timeline"]["scheduled"] for r in results}) == 1  # one batch, one prefill
        assert httpx.get(base + "/v1/models").json()["data"][0]["max_batch_size"] == 4
    finally:
        proc.terminate()
        proc.wait(timeout=20)
