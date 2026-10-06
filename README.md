# MemAdapter

MemAdapter is a three-stage framework for using retrieved memories in language-model generation. This repository contains the evaluation code, prompt templates, retrieval adapters, and launchers used to compare memory systems under a shared protocol.

## About

Memory can be useful, outdated, overly broad, or irrelevant to the current request. MemAdapter makes this decision explicit before producing an answer:

1. **Counterfactual boundary induction** identifies when each retrieved memory should and should not influence a response.
2. **Context-aware reflection** turns those boundaries into task-specific instructions for the current request.
3. **Memory-use-guided generation** produces the final response using the validated instructions.

The repository supports A-MEM, Mem0, naiveRAG, MemoryBank, and LightMem. It also includes direct generation and four post-retrieval comparison methods: Anti-Sycophancy, Self-ReCheck, Dynamic Partition, and MemGate.

## Repository Structure

```text
external_benchmarks/       Retrieval, evaluation protocol, judges, and validation
  launcher/                Retrieval and generation/judging launchers
  judges/                  Dataset-specific structured judge parsers
  rubrics/                 Task-specific prompt templates
generation/systems/        Experiment matrix for each memory system
methods/
  memadapter/              Three-stage method, model clients, and ablations
  baseline/                Direct generation baseline
  shared/                  Post-retrieval comparison methods
config/models.example.env  Local configuration template
```

`external_benchmarks/memory_systems/` contains the bundled retrieval adapters for MemoryBank and LightMem. Per-system datasets, backbones, and ablations are recorded in `generation/systems/<system>/protocol.json`.

## Installation

```bash
python -m pip install -r requirements.txt
```

Copy `config/models.example.env` to a local environment file, or export its values through your shell or secret manager. For local Qwen3-8B inference, set `QWEN_LOCAL_MODEL_PATH` to the model directory. Obtain each benchmark from its official distribution and provide its local location through the documented environment variables or command-line arguments.

## Quick Start

Inspect the available options for the main components:

```bash
python methods/memadapter/run_memadapter.py --help
python -m external_benchmarks.build_retrieval --help
python -m external_benchmarks.run_external_baseline --help
python -m external_benchmarks.run_external_judge --help
```

The pipeline runs in four steps:

1. Build and freeze retrieval for each `(dataset, task, memory system)` cell.
2. Run Baseline, MemAdapter, or a post-retrieval comparison method on that retrieval file.
3. Run the task-specific judge.
4. Validate coverage and judge parsing.

## Reproducing Experiments

Each experimental cell uses one frozen retrieval record, shared by every compared generation method. Methods receive the current request and the frozen memories; they do not receive dialogue context, query-session history, or task evidence.

Within a setting, all methods use the same generation backbone and decoding configuration:

| Backbone | Execution | Configuration |
| --- | --- | --- |
| DeepSeek-V4-Flash | API | temperature 0.2, maximum output length 4,096, thinking disabled |
| GPT-5.6-sol | API | temperature 0.2, maximum output length 4,096 |
| Qwen3-8B | Local Transformers | 4-bit NF4 quantization, bfloat16 computation, greedy decoding, thinking disabled |

Judges use deterministic decoding and task-specific rubrics. Metrics are computed from their structured outputs. Retrieval embeddings use BGE-M3.

Two ablations are included:

| Variant | Retained stages |
| --- | --- |
| `stage1-baseline` | Stage 1 only |
| `stage1-stage2-baseline` | Stages 1 and 2 |

Use `methods/memadapter/ablations/run_ablation.ps1` for one ablation cell, or the batch launchers in the same directory for all registered systems.

## Citation

Citation information will be added with the paper's public release.
