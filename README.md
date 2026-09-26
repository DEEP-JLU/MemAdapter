# MemAdapter

Official code for evaluating memory systems with retrieval, memory-guided generation, and rubric-based judging. The repository is intentionally code-only: it contains source code, launchers, prompt templates, and non-sensitive configuration examples, but no benchmark data, retrieved memories, model outputs, judge outputs, tables, logs, checkpoints, endpoint URLs, or credentials.

## Overview

MemAdapter separates memory use into three stages:

1. **Counterfactual boundary induction** creates a query-independent risk/boundary card for each retrieved memory.
2. **Context-aware reflection** converts those cards and the current task context into per-memory use instructions.
3. **Memory-use-guided generation** produces the final response under the validated instructions.

Retrieval is frozen before generation, so all methods compared within one experimental cell receive byte-identical retrieved memories. The judge consumes the benchmark's task-specific rubric and emits structured outputs; metrics are computed exclusively from these structured outputs.

## Repository layout

```text
external_benchmarks/       Retrieval, shared protocol, official-rubric judge, validation
  launcher/                Retrieval and generation/judging launchers
  judges/                  Dataset-specific structured judge parsers
  rubrics/                 Task-specific prompt templates required by the protocol
generation/systems/        Per-memory-system experiment matrix
methods/
  memadapter/              Three-stage method, local/API model clients, retained ablations
  baseline/                Direct generation prompt construction
  shared/                  Shared experiment helpers
config/models.example.env  Non-secret local configuration template
```

The registered memory systems are A-MEM, Mem0, naiveRAG, MemoryBank, and LightMem. Each system's supported datasets, backbones, and retained ablations are recorded in `generation/systems/<system>/protocol.json`.

## Backbones and decoding

All methods compared within an experimental setting use the same generation backbone and decoding configuration.

| Backbone | Execution | Decoding configuration |
| --- | --- | --- |
| DeepSeek-V4-Flash | API | temperature 0.2, maximum output length 4,096, thinking disabled |
| GPT-5.6-sol | API | temperature 0.2, maximum output length 4,096 |
| Qwen3-8B | Local Transformers | 4-bit NF4 quantization, bfloat16 computation, greedy decoding, thinking disabled |

Judges use deterministic decoding (temperature 0.0) and return structured outputs. Retrieval embeddings are computed locally with BGE-M3.

## Setup

Create an environment and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Copy `config/models.example.env` to a local, ignored environment file or export its values through your shell/secret manager. API endpoints and credentials are deliberately not stored in this repository. For Qwen3-8B, set `QWEN_LOCAL_MODEL_PATH` to an authorized local model directory.

Obtain each benchmark from its official distribution and keep its data outside this repository. Point the runners to locally obtained benchmark locations through their documented environment variables and command-line arguments.

## Running the pipeline

The intended order is:

1. Build and freeze retrieval for each `(dataset, task, memory system)` cell with `external_benchmarks/build_retrieval.py` or `external_benchmarks/launcher/run_external_retrieval.sh`.
2. Run baseline and MemAdapter generation on the same frozen retrieval file with `external_benchmarks/run_external_baseline.py` and `methods/memadapter/run_memadapter.py`.
3. Run the shared rubric judge using `external_benchmarks/run_external_judge.py`.
4. Validate protocol consistency, coverage, and judge parsing with `external_benchmarks/validate_external.py`.

For a MemAdapter run, inspect the available arguments with:

```bash
python methods/memadapter/run_memadapter.py --help
```

Generated artifacts are resumable and record retrieved memories, boundary cards, memory-use instructions, final responses, structured judge outputs, and runtime metadata per evaluated instance. Those artifacts are ignored by Git and are not released here.

## Retained ablations

The public release retains only the two paper ablations below; other exploratory variants are excluded.

| Variant | Retained computation | Final generator |
| --- | --- | --- |
| `stage1-baseline` | Stage 1 | Direct baseline conditioned on Stage-1 boundary cards |
| `stage1-stage2-baseline` | Stages 1 and 2 | Direct baseline conditioned on Stage-2 use instructions |

Use `methods/memadapter/ablations/run_ablation.ps1` for one experiment cell, or the batch scripts in the same directory for all registered systems.

## Evaluation protocol

The implementation follows the official protocol and task-specific rubrics for each supported benchmark. Prompt templates in `external_benchmarks/rubrics/` are protocol assets, not experimental results. Some official templates contain synthetic illustrative user scenarios; they are retained only because the judge requires the exact task rubric, and they do not identify project members, contributors, or evaluated users.

## Privacy and release policy

Before release, scan tracked files for credentials, endpoints, personal identifiers, organization names, machine paths, raw datasets, retrievals, generations, judgments, logs, and checkpoints. Do not commit `.env` files or any generated artifact. The repository's `.gitignore` blocks these common sources of accidental disclosure by default.

## Citation

Citation information will be added with the paper's public release.
