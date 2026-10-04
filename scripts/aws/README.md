# AWS scripts

Every paid session is scripted end to end before the instance starts: setup, model download, runs,
copy results out, stop.

| Instance | GPUs | Use |
|---|---|---|
| g6.xlarge | 1× L4 24 GB | CUDA path, study dry runs |
| g6.12xlarge | 4× L4 (PCIe, no NVLink) | Multi-GPU correctness (D1–D3) |
| p4d / p4de | 8× A100 40/80 GB | Official study numbers, Phase 2 performance |
| p5 | 8× H100 80 GB | Phase 2–3 performance, multi-node over EFA |
| p5en | 8× H200 141 GB | Phase 5 (Kimi K2) |

Checklist before any p-instance session: vCPU quota, Capacity Block if needed, placement group and
EFA-enabled AMI for multi-node, weights pre-staged in S3 in the same region, AWS Budgets alarm on.
