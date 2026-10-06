# MemAdapter

Official code for **"MemAdapter: Counterfactual Adaptation Against Memory-induced Sycophancy"**, a novel framework that adaptively integrates retrieved memories to support objective and reliable reasoning.

## Overview

MemAdapter mitigates memory-induced sycophancy via three stages:

1. **Counterfactual boundary induction** creates a query-independent risk/boundary card for each retrieved memory.
2. **Context-aware reflection** converts those cards and the current task context into per-memory use instructions.
3. **Memory-use-guided generation** produces the final response under the validated instructions.

Retrieval is frozen before generation, so all methods compared within one experimental cell receive byte-identical retrieved memories. Every post-retrieval generation method receives only the current request and those frozen memories; dialogue context, query-session history, and task evidence are not passed to any method. The judge consumes the benchmark's task-specific rubric and emits structured outputs; metrics are computed exclusively from these structured outputs.

**📃 Please [cite our paper](#-citation)** if you find this paper or repository helpful.

```bibtex
@article{ningmemadapter,
  title={Memadapter: Counterfactual Adaptation Against Memory-induced Sycophancy},
  author={Ning, Ruqing and Meng, Haibo and Xiang, Zhishang and Chen, Zerui and Su, Jinsong and Wang, Xin and Zhang, Qinggang},
  journal={arXiv preprint arXiv:2610.05162},
  year={2026}
}
```

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
  shared/                  Shared helpers and post-retrieval comparison runner
config/models.example.env  Non-secret local configuration template
```

The registered memory systems are A-MEM, Mem0, naiveRAG, MemoryBank, and LightMem. Each system's supported datasets, backbones, and retained ablations are recorded in `generation/systems/<system>/protocol.json`.

`external_benchmarks/memory_systems/` provides the bundled retrieval adapters for MemoryBank and LightMem. Each adapter constructs a per-instance memory store from the benchmark dialogue, retrieves the top-k memories, and passes the frozen retrieval record to the common generation and judge pipeline. The bundled vendor components retain their MIT license notices in the corresponding system directories.

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

Copy `config/models.example.env` to a local environment file or export its values through your shell/secret manager. For Qwen3-8B, set `QWEN_LOCAL_MODEL_PATH` to an authorized local model directory.

Obtain each benchmark from its official distribution and point the runners to the benchmark locations through their documented environment variables and command-line arguments.

## Running the pipeline

The intended order is:

1. Build and freeze retrieval for each `(dataset, task, memory system)` cell with `python -m external_benchmarks.build_retrieval` or `external_benchmarks/launcher/run_external_retrieval.sh`.
2. Run baseline and MemAdapter generation on the same frozen retrieval file with `python -m external_benchmarks.run_external_baseline` and `methods/memadapter/run_memadapter.py`.
3. Run the shared rubric judge using `python -m external_benchmarks.run_external_judge`.
4. Validate protocol consistency, coverage, and judge parsing with `python -m external_benchmarks.validate_external`.

For a MemAdapter run, inspect the available arguments with:

```bash
python methods/memadapter/run_memadapter.py --help
```

Generated artifacts are resumable and record retrieved memories, boundary cards, memory-use instructions, final responses, structured judge outputs, and runtime metadata per evaluated instance.

## Post-retrieval comparison methods

Anti-Sycophancy, Self-ReCheck, Dynamic Partition, and MemGate are implemented
in `methods/shared/run_extra_interventions.py`. Together with Baseline and
MemAdapter, they operate on a frozen retrieval record and use only the current
request and the retrieved memories. They do not consume dialogue context,
query-session history, or task evidence.
The runner uses DeepSeek-V4-Flash with temperature 0.2, a 4,096-token limit,
and thinking disabled. Credentials and endpoint URLs are supplied through the
local environment.

For example:

```bash
python methods/shared/run_extra_interventions.py --help
```

## Paper ablations

| Variant | Retained computation | Final generator |
| --- | --- | --- |
| `stage1-baseline` | Stage 1 | Direct baseline conditioned on Stage-1 boundary cards |
| `stage1-stage2-baseline` | Stages 1 and 2 | Direct baseline conditioned on Stage-2 use instructions |

Use `methods/memadapter/ablations/run_ablation.ps1` for one experiment cell, or the batch scripts in the same directory for all registered systems.

## Evaluation protocol

The implementation follows the official protocol and task-specific rubrics for each supported benchmark. Prompt templates in `external_benchmarks/rubrics/` provide the task-specific judge instructions, including synthetic illustrative scenarios used by the official evaluation protocol.

## Citation

```bibtex
@article{ningmemadapter,
  title={Memadapter: Counterfactual Adaptation Against Memory-induced Sycophancy},
  author={Ning, Ruqing and Meng, Haibo and Xiang, Zhishang and Chen, Zerui and Su, Jinsong and Wang, Xin and Zhang, Qinggang},
  journal={arXiv preprint arXiv:2610.05162},
  year={2026}
}
```
