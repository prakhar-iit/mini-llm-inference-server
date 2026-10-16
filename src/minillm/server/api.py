"""OpenAI-compatible API frontend (M2).

Endpoints: POST /v1/completions, POST /v1/chat/completions, GET /v1/models, GET /health, GET /metrics.
The frontend tokenizes, sends token ids to the engine process, detokenizes what comes back and streams
it as server-sent events in OpenAI's chunk format.

Extensions to the OpenAI request (all optional): `ignore_eos` (fixed output length for benchmarks),
`return_token_ids` (each choice carries the token ids it added; used by G0-through-the-API and by the
workload C client to append exact tokens). Every final response or final stream chunk carries
`minillm_timeline` (decision 6): monotonic-ns timestamps for each stage of the request.

Only greedy decoding exists so far: `temperature` must be 0 or omitted, `n` must be 1.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from ..engine.core import now_ns
from ..engine.protocol import AddRequest, RequestOutput
from .engine_client import EngineClient, EngineDeadError

# ---------- request bodies ----------


class StreamOptions(BaseModel):
    include_usage: bool = False


class _Common(BaseModel):
    model: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = None
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    ignore_eos: bool = False
    return_token_ids: bool = False


class CompletionRequest(_Common):
    prompt: str | list[int] | list[str] | list[list[int]]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(_Common):
    messages: list[ChatMessage]
    max_completion_tokens: int | None = Field(default=None, ge=1)


class BadRequest(Exception):
    pass


# ---------- metrics (decision 6; names follow vLLM's) ----------


class Metrics:
    def __init__(self) -> None:
        self.registry = r = CollectorRegistry()
        self.requests = Counter("minillm:request_success", "Finished requests", ["finished_reason"], registry=r)
        self.errors = Counter("minillm:request_errors", "Rejected or failed requests", ["reason"], registry=r)
        self.prompt_tokens = Counter("minillm:prompt_tokens", "Prompt tokens processed", registry=r)
        self.generation_tokens = Counter("minillm:generation_tokens", "Tokens generated", registry=r)
        buckets = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
        self.ttft = Histogram("minillm:time_to_first_token_seconds", "Server-side TTFT", registry=r, buckets=buckets)
        self.tpot = Histogram("minillm:time_per_output_token_seconds", "Server-side TPOT", registry=r,
                              buckets=(0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1))
        self.e2e = Histogram("minillm:e2e_request_latency_seconds", "Server-side end-to-end latency", registry=r,
                             buckets=buckets)
        self.queue = Histogram("minillm:request_queue_time_seconds", "Time waiting for the engine", registry=r,
                               buckets=buckets)
        self.running = Gauge("minillm:num_requests_running", "Requests being generated", registry=r)
        self.waiting = Gauge("minillm:num_requests_waiting", "Requests queued in the engine", registry=r)
        self.kv_tokens = Gauge("minillm:kv_cache_tokens_used", "Tokens held in KV caches", registry=r)

    def observe(self, timeline: dict[str, int], out: RequestOutput) -> None:
        reason = out.finish_reason or "unknown"
        self.requests.labels(finished_reason=reason if not reason.startswith("error") else "error").inc()
        self.prompt_tokens.inc(out.num_prompt_tokens)
        self.generation_tokens.inc(out.num_output_tokens)
        s = 1e-9
        if "queued" in timeline and "scheduled" in timeline:
            self.queue.observe((timeline["scheduled"] - timeline["queued"]) * s)
        if "first_token" in timeline:
            self.ttft.observe((timeline["first_token"] - timeline["received"]) * s)
            if out.num_output_tokens > 1:
                self.tpot.observe((timeline["finished"] - timeline["first_token"]) * s / (out.num_output_tokens - 1))
        self.e2e.observe((timeline["finished"] - timeline["received"]) * s)


# ---------- detokenization ----------


class IncrementalDetokenizer:
    """Turns a growing list of token ids into text deltas.

    Decodes the whole output each time and emits the new suffix (simple; O(n²) over a long output, fine
    for now). Holds back while the text ends in U+FFFD: a multi-byte character split across tokens.
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.text = ""

    def push(self, new_ids: list[int], final: bool = False) -> str:
        if self.tokenizer is None:
            return ""
        self.ids.extend(new_ids)
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        if text.endswith("�") and not final:
            return ""
        delta, self.text = text[len(self.text):], text
        return delta


# ---------- app ----------


def create_app(model_dir: str, device: str = "auto", max_model_len: int | None = None,
               served_model_name: str | None = None, step_log_path: str | None = None,
               timeline_log_path: str | None = None, max_batch_size: int = 8) -> FastAPI:
    from pathlib import Path

    from ..model.config import LlamaConfig

    config = LlamaConfig.from_pretrained(model_dir)
    max_len = max_model_len or min(config.max_position_embeddings, 8192)
    model_name = served_model_name or Path(model_dir).name
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except (OSError, ValueError):
        tokenizer = None  # token-id prompts only (tests use tiny tokenizer-less models)
    eos_ids = list(config.eos_token_ids)
    metrics = Metrics()
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = EngineClient(model_dir, device, max_len, step_log_path, max_batch_size)
        state["engine"] = engine
        await engine.start()
        yield
        await engine.shutdown()

    app = FastAPI(title="minillm", lifespan=lifespan)

    @app.exception_handler(BadRequest)
    async def _bad_request(_: Request, exc: BadRequest):
        metrics.errors.labels(reason="bad_request").inc()
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request_error"}})

    @app.exception_handler(EngineDeadError)
    async def _engine_dead(_: Request, exc: EngineDeadError):
        metrics.errors.labels(reason="engine_dead").inc()
        return JSONResponse(status_code=503, content={"error": {"message": str(exc), "type": "server_error"}})

    @app.get("/health")
    async def health():
        engine: EngineClient = state["engine"]
        if not engine.is_alive():
            return JSONResponse(status_code=503, content={"status": "engine not running"})
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "minillm",
                                            "max_model_len": max_len, "max_batch_size": max_batch_size}]}

    @app.get("/metrics")
    async def prometheus():
        s = state["engine"].stats
        metrics.running.set(s.num_running)
        metrics.waiting.set(s.num_waiting)
        metrics.kv_tokens.set(s.kv_tokens_used)
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    def validate(body: _Common, prompt_ids: list[int], max_tokens: int) -> None:
        if body.temperature not in (None, 0, 0.0):
            raise BadRequest("only greedy decoding is supported so far: set temperature to 0 or omit it")
        if body.n != 1:
            raise BadRequest("n must be 1")
        if body.stop:
            raise BadRequest("stop strings are not supported yet")
        if not prompt_ids:
            raise BadRequest("prompt is empty")
        if len(prompt_ids) + max_tokens > max_len:
            raise BadRequest(f"prompt ({len(prompt_ids)} tokens) + max_tokens ({max_tokens}) exceeds "
                             f"the maximum context length {max_len}")

    def tokenize_prompt(prompt) -> list[int]:
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            return list(prompt)
        if isinstance(prompt, str):
            if tokenizer is None:
                raise BadRequest("this model has no tokenizer; send the prompt as a list of token ids")
            return tokenizer(prompt).input_ids
        raise BadRequest("send one prompt per request (a string or a list of token ids)")

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        received = now_ns()
        prompt_ids = tokenize_prompt(body.prompt)
        max_tokens = body.max_tokens or 16  # OpenAI's default for completions
        validate(body, prompt_ids, max_tokens)
        return await run(body, prompt_ids, max_tokens, received, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        received = now_ns()
        if tokenizer is None or not getattr(tokenizer, "chat_template", None):
            raise BadRequest("this model has no chat template (base models don't); use /v1/completions")
        prompt_ids = tokenizer.apply_chat_template([m.model_dump() for m in body.messages],
                                                   add_generation_prompt=True, tokenize=True)
        if isinstance(prompt_ids, dict):  # some tokenizers return a BatchEncoding
            prompt_ids = prompt_ids["input_ids"]
        max_tokens = body.max_completion_tokens or body.max_tokens or max(1, max_len - len(prompt_ids))
        validate(body, prompt_ids, max_tokens)
        return await run(body, prompt_ids, max_tokens, received, chat=True)

    async def run(body: _Common, prompt_ids: list[int], max_tokens: int, received: int, chat: bool):
        engine: EngineClient = state["engine"]
        req_id = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        timeline = {"received": received, "tokenized": now_ns()}
        req = AddRequest(req_id, prompt_ids, max_tokens, stop_token_ids=eos_ids, ignore_eos=body.ignore_eos,
                         timeline=timeline)
        created = int(time.time())
        detok = IncrementalDetokenizer(tokenizer)

        def finish(out: RequestOutput) -> dict[str, int]:
            tl = {**timeline, **(out.timeline or {})}
            metrics.observe(tl, out)
            if timeline_log_path:
                with open(timeline_log_path, "a") as f:
                    f.write(json.dumps({"req_id": req_id, "finish_reason": out.finish_reason,
                                        "prompt_tokens": out.num_prompt_tokens,
                                        "output_tokens": out.num_output_tokens, "timeline": tl}) + "\n")
            return tl

        outputs = engine.generate(req, on_abandoned=finish)  # a disconnected client's abort is still recorded

        def usage(out: RequestOutput) -> dict[str, int]:
            return {"prompt_tokens": out.num_prompt_tokens, "completion_tokens": out.num_output_tokens,
                    "total_tokens": out.num_prompt_tokens + out.num_output_tokens}

        if not body.stream:
            text, token_ids = "", []
            async for out in outputs:
                text += detok.push(out.new_token_ids, final=out.finished)
                token_ids += out.new_token_ids
            if out.finish_reason and out.finish_reason.startswith("error"):
                raise BadRequest(out.finish_reason)
            tl = finish(out)
            choice: dict[str, Any] = {"index": 0, "finish_reason": out.finish_reason}
            if chat:
                choice["message"] = {"role": "assistant", "content": text}
            else:
                choice["text"] = text
            if body.return_token_ids:
                choice["token_ids"] = token_ids
            return {"id": req_id, "object": "chat.completion" if chat else "text_completion", "created": created,
                    "model": model_name, "choices": [choice], "usage": usage(out), "minillm_timeline": tl}

        async def sse() -> AsyncIterator[str]:
            obj = "chat.completion.chunk" if chat else "text_completion"

            def chunk(text: str, ids: list[int], reason: str | None, first: bool = False) -> dict[str, Any]:
                c: dict[str, Any] = {"index": 0, "finish_reason": reason}
                if chat:
                    c["delta"] = ({"role": "assistant"} if first else {}) | ({"content": text} if text or not first else {})
                else:
                    c["text"] = text
                if body.return_token_ids:
                    c["token_ids"] = ids
                return {"id": req_id, "object": obj, "created": created, "model": model_name, "choices": [c]}

            if chat:
                yield f"data: {json.dumps(chunk('', [], None, first=True))}\n\n"
            async for out in outputs:
                if out.finish_reason and out.finish_reason.startswith("error"):
                    yield f"data: {json.dumps({'error': {'message': out.finish_reason}})}\n\n"
                    return
                payload = chunk(detok.push(out.new_token_ids, final=out.finished), out.new_token_ids,
                                out.finish_reason if out.finished else None)
                if out.finished:
                    payload["minillm_timeline"] = finish(out)
                yield f"data: {json.dumps(payload)}\n\n"
                if out.finished and body.stream_options and body.stream_options.include_usage:
                    yield f"data: {json.dumps({'id': req_id, 'object': obj, 'created': created, 'model': model_name, 'choices': [], 'usage': usage(out)})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app
