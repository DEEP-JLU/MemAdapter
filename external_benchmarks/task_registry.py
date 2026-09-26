"""Registry of the external task names, their rubrics and their metrics.

Task names are prefixed (``memtrap_*`` / ``persist_*``) so they can never collide
with the five MemSyco-Bench task names that are hard-coded throughout
``MemAdapter/run_memadapter.py`` and ``benchmark/evaluation/``.

The MemTrapBench sub-dataset -> rubric mapping mirrors the upstream
``runners/eval/eval_common.py::DATASET_CONFIGS`` exactly, including the
per-rubric placeholder family (``judge_type``). The three rubrics do NOT share a
dimension set: ``shared`` scores four dimensions, ``poison`` and ``number_game``
score two each, with different key names. The judge therefore parses dimensions
generically rather than assuming a fixed count.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    task: str
    dataset: str
    subset: str
    #: Path relative to ``paths.MEMTRAP_ROOT`` (MemTrapBench) or
    #: ``paths.PERSIST_ROOT`` (PersistBench).
    data_file: str
    #: Rubric basename under ``paths.RUBRICS_ROOT/<rubric_dir>``.
    rubric_file: str
    rubric_dir: str
    #: MemTrapBench placeholder family; empty for PersistBench.
    judge_type: str
    #: Independent generations per sample. PersistBench draws 3 for the two
    #: safety classes (so FR@1/FR@2/FR@3 are meaningful) and 1 for beneficial.
    k_generations: int
    #: ``dim5`` = 0-5 per-dimension rubric; ``score5`` = 1-5 failure score;
    #: ``rating3`` = 1-3 beneficial rating (inverted when reported).
    metric: str
    #: JSON key the judge's own rubric asks for.
    score_key: str

    @property
    def is_memtrap(self) -> bool:
        return self.dataset == "memtrapbench"


_MEMTRAP = [
    # task suffix,      data file,                                  judge_type,   rubric
    ("poison", "memtrapbench/safety/poison_200.json", "poison", "judge_poison.txt"),
    ("number_game", "memtrapbench/cognitive_bias/number_game_200.json", "number_game", "judge_number_game.txt"),
    ("hurt", "memtrapbench/trauma/hurt_150.json", "shared", "judge_shared.txt"),
    ("hallucination", "memtrapbench/task_boundary/hallucination_150.json", "shared", "judge_shared.txt"),
    ("inertia", "memtrapbench/cognitive_bias/Inertia_150.json", "shared", "judge_shared.txt"),
    ("unclear", "memtrapbench/task_boundary/unclear_200.json", "shared", "judge_shared.txt"),
]

_PERSIST = [
    # failure_type,               k, metric,    score_key
    ("cross_domain", 3, "score5", "score"),
    ("sycophancy", 3, "score5", "score"),
    ("beneficial_memory_usage", 1, "rating3", "rating"),
]

EXTERNAL_TASKS: dict[str, TaskSpec] = {}

for _suffix, _data, _jtype, _rubric in _MEMTRAP:
    _task = f"memtrap_{_suffix}"
    EXTERNAL_TASKS[_task] = TaskSpec(
        task=_task,
        dataset="memtrapbench",
        subset=_suffix,
        data_file=_data,
        rubric_file=_rubric,
        rubric_dir="memtrap",
        judge_type=_jtype,
        k_generations=1,
        metric="dim5",
        score_key="dimensions",
    )

for _ftype, _k, _metric, _key in _PERSIST:
    _task = f"persist_{_ftype}"
    EXTERNAL_TASKS[_task] = TaskSpec(
        task=_task,
        dataset="persistbench",
        subset=_ftype,
        data_file="baseline/"
        + {
            "cross_domain": "cross_domain.jsonl",
            "sycophancy": "sycophancy.jsonl",
            "beneficial_memory_usage": "beneficial_samples.jsonl",
        }[_ftype],
        rubric_file={
            "cross_domain": "judge_cross_domain.txt",
            "sycophancy": "judge_sycophancy.txt",
            "beneficial_memory_usage": "judge_beneficial.txt",
        }[_ftype],
        rubric_dir="persist",
        judge_type="",
        k_generations=_k,
        metric=_metric,
        score_key=_key,
    )


def is_external_task(task: str) -> bool:
    return task in EXTERNAL_TASKS


def spec(task: str) -> TaskSpec:
    try:
        return EXTERNAL_TASKS[task]
    except KeyError:
        raise KeyError(
            f"Unknown external task {task!r}; known: {sorted(EXTERNAL_TASKS)}"
        ) from None


def tasks_for_dataset(dataset: str) -> list[str]:
    return [t for t, s in EXTERNAL_TASKS.items() if s.dataset == dataset]
