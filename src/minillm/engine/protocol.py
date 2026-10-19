"""Messages between the API frontend and the engine core (decision 1).

Plain dataclasses, encoded with msgpack as tagged maps: no pickle, so a Go/Rust router (decision 10)
can read them too. The engine deals in token ids only; text lives in the frontend.

    frontend -> engine:  AddRequest, Abort
    engine -> frontend:  Ready (once, after the model loads), StepOutputs (after every engine step)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import msgpack


@dataclass
class AddRequest:
    req_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    stop_token_ids: list[int] = field(default_factory=list)  # EOS and friends; ignored if ignore_eos
    ignore_eos: bool = False  # benchmarks fix output length this way (decision 2)
    timeline: dict[str, int] = field(default_factory=dict)  # frontend stages so far (monotonic ns)


@dataclass
class Abort:
    req_id: str


@dataclass
class Ready:
    model: str
    device: str
    dtype: str


@dataclass
class RequestOutput:
    req_id: str
    new_token_ids: list[int]
    finished: bool = False
    finish_reason: str | None = None  # "stop" | "length" | "abort"
    timeline: dict[str, int] | None = None  # engine stages, sent once the request finishes
    num_prompt_tokens: int = 0
    num_output_tokens: int = 0


@dataclass
class EngineStats:
    num_waiting: int
    num_running: int
    kv_tokens_used: int  # tokens whose k/v are in the pool right now
    kv_blocks_used: int = 0  # M5: paged pool blocks held by running requests
    kv_blocks_total: int = 0
    num_preemptions: int = 0  # since the engine started


@dataclass
class StepOutputs:
    outputs: list[RequestOutput]
    stats: EngineStats


_TYPES = {cls.__name__: cls for cls in (AddRequest, Abort, Ready, StepOutputs)}


def encode(msg: Any) -> bytes:
    return msgpack.packb({"type": type(msg).__name__, "body": asdict(msg)})


def decode(data: bytes) -> Any:
    raw = msgpack.unpackb(data)
    cls, body = _TYPES[raw["type"]], raw["body"]
    if cls is StepOutputs:
        return StepOutputs([RequestOutput(**o) for o in body["outputs"]], EngineStats(**body["stats"]))
    return cls(**body)
