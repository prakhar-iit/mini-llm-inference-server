# Design decisions

One entry per decision from the plan's "Design review — open decisions" list. Add a row when a
decision is made; write a short note `NN-<topic>.md` if the reasoning needs more than a line.

| # | Decision | Needed by | Choice | Date |
|---|---|---|---|---|
| 1 | Engine process architecture | M2 | | |
| 2 | Workload C (agent) spec | M2 | | |
| 3 | SLO and metric definitions | M2 | | |
| 4 | Correctness method | M2 | | |
| 5 | Benchmark client | M2 | | |
| 6 | Observability | M2 | | |
| 7 | Attention backend and KV layout | M5 | | |
| 8 | Block-hash spec | M6 | | |
| 9 | Scheduler policy | M4 | | |
| 10 | Systems language (Go / Rust) | D2 | | |
| 11 | Router state and tokenization | D2 | | |
| 12 | Session / agent API contract | D4 | | |
| 13 | Pipeline parallelism gap | N2 | | |
| 14 | KV transfer mechanism | D3 | | |
| 15 | Failure model | D5 | | |
| 16 | Multi-GPU dev hardware | D1 | | |
| 17 | Cluster cache data vs control plane | P1 | | |
| 18 | KV compatibility across engines | P1 | | |
| 19 | Eviction and TTL across tiers | D4 | | |
| 20 | AWS access and quotas | Phase 3 | | |
| 21 | Budget per phase | Phase 1 | | |
| 24 | Phase 5 choices (checkpoint, hardware, EFA all-to-all, weights, budget) | K3 | | |
