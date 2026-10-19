# 09 — Scheduler policy

**Decided:** 2026-10-18 · **Needed by:** M4 · **Status:** accepted

## Choice
Iteration-level scheduling (Orca): every engine step decides again which requests run, so a request
joins as soon as a slot is free and leaves the moment it finishes.

- **Admission: FCFS, no skipping.** Requests are admitted in arrival order. If the request at the head
  of the queue doesn't fit, nothing behind it is admitted either, so a long prompt can't be starved
  by a stream of short ones.
- **Capacity: slots.** `max_batch_size` slots, each reserving `max_model_len` tokens of contiguous KV,
  allocated once at startup. A request admitted into a slot always has room for its full output, so
  M4 never preempts and can't run out of KV memory mid-request.
- **Token budget per step:** `max_num_batched_tokens` (default 8192) caps the tokens computed in one
  prefill step, counting padding (k requests × the longest prompt). The head request is always
  admitted even when it alone exceeds the budget: there is no chunked prefill until M7, and the API
  already caps prompts at `max_model_len`.
- **Step kind: prefill first.** If requests are waiting and a slot is free, the step prefills the
  newly admitted requests; otherwise every running request decodes one token. Running decodes pause
  during a prefill step. That pause is the ITL spike S4 predicts, and M7 removes it by mixing prefill
  chunks into decode steps.
- **Overload (R2):** the frontend bounds the requests it has in flight to `max_batch_size +
  max_waiting` (default `max_waiting` = 64) and rejects the rest with HTTP 503 and `Retry-After`.
  It counts requests itself, so the bound is exact and needs no round trip to the engine.
- **Preemption:** none at M4, because slots reserve their full length. From M5 (paged KV, blocks
  allocated as tokens are generated), the newest running request is preempted first and resumes by
  recompute.
- **Since M5a (paged KV):** capacity is blocks, not slots. A request is admitted when the pool has blocks for
  all its tokens plus one for every request whose next token starts a new block (a one-step lookahead, so an
  admission doesn't force a preemption on the next decode). The token budget counts real tokens, since a ragged
  step has no padding. A decode step that finds no free block preempts the newest running request; it goes back
  to the front of the queue and is recomputed (prompt + output so far) when admitted again.
- **Priority hook:** admission order comes from one method, `Scheduler._admission_order()`. Today it
  returns the queue as it is. D4/P3 will change only this method so resumed agent sessions can jump
  the queue.

## Why
- FCFS with no skipping is the fairest policy that's easy to reason about, and it matches vLLM's
  default, so S11 compares like with like.
- Reserving `max_model_len` per slot is the honest contiguous baseline that S5 measures paged KV
  against. It wastes memory on short requests; that waste is the point of M5.
- Prefill-first gives the best TTFT, and its cost (decode stalls) is exactly what S4 and S7 are
  meant to expose.

## Rejected
- **Decode-first** (admit only once the running batch drains): this is static batching again, with
  the head-of-line blocking from S3.
- **Shortest-job-first:** better mean latency, but output length isn't known in advance, and it
  starves long requests.
- **Mixed prefill + decode steps now:** this needs chunked prefill to bound the step time. It's M7's job.
- **Admission control in the engine:** the engine can't answer HTTP, and its queue length reaches
  the frontend one step late.

## Costs to watch
- Attention in a decode step runs over the longest running request's length for every row
  (contiguous slots, masked). Paged attention (M5) removes this.
- Compaction: when a request leaves, the request in the last slot moves into its slot so the running
  rows stay contiguous (a view, not a gather, every decode step). That copies one request's KV per
  departure.
- The 503 threshold is a fixed count. An SLO-aware bound (queue length × expected service time vs the
  TTFT SLO) is a later improvement if R2 shows admitted requests missing their SLO.
