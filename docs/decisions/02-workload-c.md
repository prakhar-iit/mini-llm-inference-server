# 02 — Workload C (agent traffic)

**Decided:** 2026-10-11 · **Needed by:** M2 · **Status:** accepted

## Choice
A synthetic generator with explicit, sweepable distributions. Public traces are used to check it, not
to drive it:
- **Azure LLM inference traces / BurstGPT:** fit the arrival burstiness (the CV knob below).
- **Mooncake (Kimi) trace:** added as a replay mode at P1, as a second data point for cluster caching.

## What agent traffic looks like
- Every session of one agent type starts with the same system prompt + tool schemas (shared prefix).
- Each turn resends the whole history plus the new tool result or message (growing context).
- Between turns the session pauses while a tool runs; its KV cache sits idle (D4, P3).
- Most outputs are short tool calls; only the final turn is long.

## Generator spec (defaults; every value is a flag)

| Parameter | Default |
|---|---|
| Agent types, each with its own shared prompt | 4 |
| Shared prompt + tool schemas | 3,500 tokens (range 2,000–5,000) |
| Turns per session | lognormal, mean 6, clipped to 2–20 |
| New tokens per turn | lognormal, median 300; 10% are large tool outputs of 1,000–2,000 |
| Output per turn | ~80 tokens (tool call); final turn ~300; fixed via `max_tokens` + `ignore_eos` |
| Tool pause between turns | lognormal, median 2 s, clipped to 0.2–30 s |
| Session arrivals | gamma process; rate is the swept variable; CV 1 = Poisson, CV > 1 bursty (fitted to traces) |

## Mechanics
- Prompts are sent as **token-id arrays** (the OpenAI completions API accepts them): exact lengths and
  exact prefix sharing, unaffected by tokenization.
- The client appends each turn's **actual generated tokens** to the session history, as a real agent does.
- A seeded run writes all sessions to JSONL, so every engine and milestone sees identical traffic.
- Content is abstract (token ids drawn from real text for realistic id statistics); no proprietary data.

## Metrics specific to C
Per-turn TTFT, session completion time, prefix-cache hit rate (from M6), and goodput as sessions/s
meeting the SLO (decision 3).

## Rejected
- **Trace replay only:** realistic arrivals, but Azure/BurstGPT have no content (no prefix structure) and
  Mooncake is chat traffic with 512-token blocks; neither can sweep the shared-prefix fraction S6 needs.
- **Generator and Mooncake replay both at M2:** doubles the harness work before anything needs it.
