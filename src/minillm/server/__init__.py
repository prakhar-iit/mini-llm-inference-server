"""OpenAI-compatible API frontend (M2): /v1/completions and /v1/chat/completions with streaming.

Talks to the engine process over ZMQ. Records per-request timelines (queue, prefill, decode,
cache-hit tokens) and exposes Prometheus metrics.
"""
