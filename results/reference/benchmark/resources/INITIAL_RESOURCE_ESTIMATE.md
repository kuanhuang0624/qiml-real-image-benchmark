# Initial resource estimate

Frozen before dataset processing. Estimates are intentionally conservative and will be replaced by measured pilot throughput before each phase.

| Phase | CPU-hours | GPU-hours | Peak memory | Persistent storage | Dominant count |
|---|---:|---:|---:|---:|---|
| data/splits | 1 | 0 | 4 GiB | 0.5 GiB | 4 datasets |
| ResNet feature extraction | 4 | 3 | 8 GiB CPU / 4 GiB GPU | 0.5 GiB | about 250k images |
| fixed quantum features and validation | 28 | 0 | 32 GiB | 8 GiB | <1M eight-qubit states |
| QK exact/finite-shot 1k+5k | 36 | 0 | 24 GiB | 24 GiB | hundreds of millions of kernel entries/draws |
| VQC-DR and Yomo-Matched smoke/validation/scaling | 20 | 24 | 16 GiB / 8 GiB GPU | 8 GiB | checkpointed batched statevectors |
| PdrQC-Matched search and evaluation | 32 | 0 | 32 GiB | 12 GiB | 276 Pauli candidates and depth≤3 search |
| CNN and neural controls | 4 | 14 | 12 GiB / 8 GiB GPU | 10 GiB | nested scales × seeds |
| statistics/tables/figures/audit | 22 | 0 | 32 GiB | 20 GiB | 10k bootstrap + 10k permutations |
| **estimated total** | **147** | **41** | **32 GiB** | **83 GiB** | within initial global caps |

The estimate is below the 200 CPU-hour and 48 GPU-hour launch caps but leaves little GPU margin. Full-data VQC-DR, Yomo-Matched, PdrQC-Matched, C-Poly, C-RBF, and 10k QK-IQP remain separately resource-gated as required. If pilot throughput projects the total above either cap, `RESOURCE_ESCALATION_REQUIRED.md` will be created and execution will pause before the excessive phase. No method, seed, shot budget, or dataset will be silently removed.
