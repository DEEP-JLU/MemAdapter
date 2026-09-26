# Retained MemAdapter ablations

The public repository retains exactly two ablations. Both reuse the same frozen retrieval input, sample selection, model settings, and judge contract as the full method.

| Variant | Retained stages | Final generator |
| --- | --- | --- |
| `stage1-baseline` | Stage 1 | Direct baseline with Stage-1 boundary cards |
| `stage1-stage2-baseline` | Stages 1 and 2 | Direct baseline with Stage-2 use instructions |

Use `run_ablation.ps1` for one `(model, memory-system)` cell, or the two batch scripts for all registered systems. Supply local credentials and endpoints through environment variables; no secrets or endpoint defaults are stored in this repository.
