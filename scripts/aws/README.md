# AWS scripts

Every paid session is scripted end to end before the instance starts: setup, model download, runs,
copy results out, stop.

| Instance | GPUs | Use |
|---|---|---|
| g6.xlarge | 1× L4 24 GB | CUDA path, study dry runs |
| g6.12xlarge | 4× L4 (PCIe, no NVLink) | Multi-GPU correctness (D1–D3) |
| p5.4xlarge | 1× H100 80 GB | Phase 1 official numbers, S0–S11 |
| p5.48xlarge | 8× H100 80 GB (NVLink, EFA) | Phases 2–4 performance, multi-node over EFA |
| p5en.48xlarge | 8× H200 141 GB | Phase 5 (Kimi K2) |

One measurement GPU per chain of studies (decision 25). p5.4xlarge needs 16 vCPUs of
"Running On-Demand P instances" quota.

Checklist before any p-instance session: vCPU quota, Capacity Block if needed, placement group and
EFA-enabled AMI for multi-node, weights pre-staged in S3 in the same region, AWS Budgets alarm on.

IAM policy for the `minillm-agent` user: [`iam/`](iam/README.md).
