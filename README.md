# Memory-System Evaluation Code

Code for the paper's evaluation pipeline. This repository deliberately contains only source code, launch scripts, prompt templates, and non-sensitive example configuration. It contains no datasets, retrieval artifacts, generations, judgments, tables, logs, model weights, endpoints, or credentials.

| Stage | Location | Purpose |
| --- | --- | --- |
| Retrieval | `external_benchmarks/build_retrieval.py` and `external_benchmarks/launcher/run_external_retrieval.sh` | Freeze retrieved memories for each dataset/system pair. |
| Generation | `methods/memadapter/`, `methods/baseline/`, and `methods/shared/` | Run direct and memory-guided generation. |
| Judge | `external_benchmarks/judges/` and `external_benchmarks/run_external_judge.py` | Apply dataset-specific judge prompts. |
| Experiment matrix | `generation/systems/` | Records each memory system's datasets, model families, and retained ablations. |

The supported model-family labels are `DeepSeek`, `Qwen`, and `GPT`. Endpoints, model identifiers, and credentials must be supplied locally through environment variables; see `config/models.example.env`.

## Quick start

1. Create a Python environment and install `requirements.txt`.
2. Obtain benchmark datasets separately and place them outside this repository (or configure their locations locally).
3. Export the model family's key, endpoint, and model name in your shell. Do not write these values into tracked files.
4. Run retrieval, generation, then judging. Each stage writes to an ignored output directory, so experiment artifacts cannot be committed accidentally.

## Retained ablations

Only two MemAdapter ablations are retained for the public release:

- `stage1_only`: Stage 1 followed by the direct baseline generator.
- `stage1_stage2`: Stages 1 and 2 followed by the direct baseline generator.

All other exploratory variants and every experimental result are excluded.

## Release policy

Before publishing, run `git status --ignored` and a secret scanner in the repository root. Never add `.env` files, dataset exports, model caches, outputs, or logs. The code intentionally requires endpoints at runtime rather than embedding them as defaults.
