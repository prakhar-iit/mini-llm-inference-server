# 01 — Engine process architecture

**Decided:** 2026-10-11 · **Needed by:** M2 · **Status:** accepted

## Choice
Two processes, as in vLLM V1:

```
 clients ──HTTP──▶ [ API frontend ]  ──ZMQ: token ids──▶  [ engine core ]
                   FastAPI, tokenize,  ◀──ZMQ: new tokens──  scheduler + executor
                   detokenize, stream                       (one loop, owns the GPU)
```

- **API frontend:** OpenAI-compatible HTTP, tokenization, detokenization, streaming, per-request timing.
- **Engine core:** a busy loop of schedule → run model → emit new tokens. Sees token ids only.
- **Executor interface** inside the engine core: with one GPU the model runs in-process; at D1 it
  starts one worker process per GPU, and neither the frontend nor the scheduler changes.
- **Transport:** ZMQ sockets (work across machines, needed for the D2 router and Phase 3).
- **Encoding:** msgpack, not pickle (plain data, faster, readable from the Go/Rust router of decision 10).

## Why
Python runs one thread at a time per process. In a single process, HTTP parsing, tokenizing and
detokenizing for many clients stall the engine loop between GPU steps, and a decode step can be under a
millisecond on an H100. Splitting the frontend out keeps the GPU loop free. It also matches what Phase 2
needs: one worker per GPU (D1), replicas behind a router (D2), separate prefill and decode engines (D3).

## Rejected
- **Single process (asyncio):** simplest, but request handling stalls the GPU loop under load, and
  Phase 2 would need a rewrite.
- **Frontend + scheduler + per-GPU workers from day one:** the final D1 shape, but an extra process hop
  every step with no benefit on one GPU. The executor interface leaves room for it.

## Costs to watch
- Messaging overhead per engine step (expected 50–200 µs); S2 measures it.
- A message protocol to design and version (add request, abort, step outputs with timings).
- Debugging across two processes; the engine core must also run in-process for tests.
