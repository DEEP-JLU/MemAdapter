"""Run the MemAdapter experiment on frozen MemSyco-Bench retrieval records.

Examples (PowerShell, from ``D:/agent记忆谄媚``)::

    $env:DEEPSEEK_API_KEY = "..."
    python MemAdapter/run_memadapter.py run-all --model DeepSeek --memory-system AMEM --limit 10
    python MemAdapter/run_memadapter.py run-all --model GPT --memory-system Mem0

Generation is resumable. A completed row is appended only after all three
MemAdapter stages succeed. With ``--continue-on-error``, successful rows are
stored independently while failed samples are recorded and do not block later
samples; completed rows are merged back in retrieval order once all samples
finish. Judge and summary are separate stages so they can also be rerun
without calling the generation models again.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = Path(__file__).resolve().parent
LOCAL_BENCHMARK_ROOT = ROOT.parent / "datasets" / "memsyco-bench" / "source"
BENCHMARK_ROOT = Path(
    os.environ.get("MEMSYCO_BENCHMARK", str(LOCAL_BENCHMARK_ROOT))
)
if str(BENCHMARK_ROOT / "evaluation") not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT / "evaluation"))
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from _dataset_compat import to_eval_row  # noqa: E402

from memadapter import ABLATION_VARIANTS, run_ablation, run_three_stage  # noqa: E402
from model_client import MODEL_DEFAULTS, ModelClient, _openai_base_url  # noqa: E402
from efficiency import (  # noqa: E402
    EfficiencyRecorder,
    TracedOpenAI,
    new_run_id,
    rebuild_efficiency_artifacts,
    utc_now,
)


SYSTEMS = {
    "AMEM": Path("a-mem") / "retrieved_official_amem_top10.jsonl",
    "Mem0": Path("mem0") / "retrieved_official_memzero_top10.jsonl",
    "naiveRAG": Path("naive-rag") / "retrieved_official_naiverag_top10.jsonl",
    # These two files are produced by the official MemSyco-Bench baseline
    # adapters.  They are separate frozen retrieval sets, so downstream
    # baseline and MemAdapter runs consume exactly the same memory evidence.
    "MemoryBank": Path("memory-bank") / "retrieved_official_memorybank_top10.jsonl",
    "LightMem": Path("light-mem") / "retrieved_official_lightmem_top10.jsonl",
}
RETRIEVAL_ROOT = ROOT.parent / "datasets" / "memsyco-bench" / "retrieval"
FULL_PROMPT_VERSION = "MemAdapter-full-20260922"
FULL_PROMPT_FILES = (
    "01_counterfactual_boundary_induction.txt",
    "02_context_aware_memory_reflection.txt",
    "03_memory_use_guided_generation.txt",
)
ABLATION_PROMPT_FILES = {
    "stage1-baseline": (
        "prompts/01_counterfactual_induction_method_aligned.txt",
        "ablations/prompts/stage1_then_baseline_generation.txt",
    ),
    "stage1-stage2-baseline": (
        "prompts/01_counterfactual_induction_method_aligned.txt",
        "prompts/02_context_aware_reflection_method_aligned.txt",
        "ablations/prompts/stage1_stage2_then_baseline_generation.txt",
    ),
}
MODELS = ("DeepSeek", "GPT", "Qwen")
TASKS = (
    "objective_fact_judgment",
    "contextual_scope_control",
    "memory_evidence_conflict",
    "personalized_memory_use",
    "valid_memory_selection",
)
PASS_METRICS = {
    "objective_fact_judgment": "suppress_pass",
    "contextual_scope_control": "scope_pass",
    "memory_evidence_conflict": "evidence_pass",
    "personalized_memory_use": "memory_use_pass",
    "valid_memory_selection": "valid_selection_pass",
}
CORRECTNESS_METRICS = {
    "objective_fact_judgment": "objective_correctness",
    "contextual_scope_control": "accuracy",
    "memory_evidence_conflict": "accuracy",
    "personalized_memory_use": "answer_accuracy",
    "valid_memory_selection": "uses_latest_preference",
}

#: Cap on retrieved memories per row. The five native retrieval files hold exactly
#: ten, but PersistBench's depth is the size of the sample's given memory set (4-16),
#: because truncating to ten would drop more than half the memories on 259 of its
#: 500 rows and turn a fidelity question into a truncation artefact. Raised through
#: the environment rather than the default so the native runs are untouched; an
#: under-set value fails loudly on the first offending row instead of truncating.
MAX_RETRIEVED_MEMORIES = int(os.environ.get("MEMADAPTER_MAX_RETRIEVED_MEMORIES", "10"))


def external_task_specs() -> dict[str, Any]:
    """Task specs contributed by the external-benchmark harness, if it is present.

    Imported lazily and defensively: this module puts ``benchmark/`` on ``sys.path``
    but not the repository root, and a checkout without ``external_benchmarks`` must
    keep running the five native tasks unchanged.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from external_benchmarks import task_registry  # type: ignore
    except ImportError:
        return {}
    return dict(task_registry.EXTERNAL_TASKS)


def known_tasks() -> frozenset[str]:
    """Every task this runner accepts: the native five plus any external ones."""
    return frozenset(TASKS) | frozenset(external_task_specs())


def max_retrieved_memories(task: str) -> int:
    """Retrieval depth allowed for ``task``.

    External tasks get their own bound because their per-sample depth is set by the
    dataset, while the native tasks are fixed at ten and must not silently widen.
    """
    if task in external_task_specs():
        return int(os.environ.get("MEMADAPTER_MAX_RETRIEVED_MEMORIES_EXTERNAL", "16"))
    return MAX_RETRIEVED_MEMORIES


def is_external_task(task: str) -> bool:
    return task in external_task_specs()


def external_tasks_in(rows: list[dict[str, Any]]) -> list[str]:
    """External task names appearing in ``rows``, for the native-stage guards."""
    return sorted(
        {str(row.get("task") or "") for row in rows if is_external_task(str(row.get("task") or ""))}
    )


def refuse_native_stage(stage: str, tasks: list[str]) -> None:
    """Stop the native summary/validate/compare paths from running on external results.

    Those paths key every number off ``PASS_METRICS`` / ``CORRECTNESS_METRICS``, which
    the external rubrics do not define. Left unguarded, ``summary`` dies on a KeyError
    and -- worse -- ``validate`` can pass a run it never actually checked, reporting a
    clean bill of health for coverage it silently skipped. The external counterparts
    are ``summarize_external.py`` and ``validate_external.py``.
    """
    if not tasks:
        return
    raise SystemExit(
        f"[{stage}] the native MemSyco-Bench {stage} stage does not apply to external "
        f"tasks {tasks}. Use external_benchmarks/run_external_judge.py, "
        f"summarize_external.py and validate_external.py instead."
    )


def run_index_from_output_dir(directory: Path) -> int:
    """Recover the repeat index an external output directory encodes.

    ``external_benchmarks.paths.arm_dir`` appends ``run<N>`` to the N-th independent
    generation. The runner only ever knows its output directory, so reading the index
    back off the path keeps one source of truth: a process cannot believe it is
    repeat 2 while writing into repeat 3's stage cache.
    """
    match = re.fullmatch(r"run(\d+)", directory.name)
    return int(match.group(1)) if match else 0


def ensure_external_protocol(args: argparse.Namespace, rows: list[dict[str, Any]]) -> str | None:
    """Write the protocol record for an external cell; return its run fingerprint.

    ``generate`` serves both the native MemSyco-Bench tasks and the external
    benchmarks. Only the latter have a ``run_config.json`` -- the native results
    predate it, and adding one there would change a frozen protocol -- so this returns
    ``None`` without touching anything when no external task is selected.

    The config comes from ``external_benchmarks.run_config.config_for_external_arm``,
    the same function the baseline arm calls, so the two arms describe one experiment
    by construction; two derivations agreeing today is not the same as one derivation
    that cannot disagree. It is written before the first model call, so a drifted
    environment (model, temperature, retrieval file) stops the run instead of
    producing answers whose record no longer describes them.
    """
    tasks = external_tasks_in(rows)
    if not tasks:
        return None
    others = sorted({str(row.get("task") or "") for row in rows} - set(tasks))
    if others or len(tasks) != 1:
        raise SystemExit(
            f"Retrieval file holds external tasks {tasks} alongside {others}; one cell "
            f"is one task."
        )
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from external_benchmarks import run_config as external_benchmarks_config  # type: ignore

    config = external_benchmarks_config.config_for_external_arm(
        retrieval_file=args.retrieval_file,
        system=args.memory_system,
        arm="memadapter",
        # The cell's size, not this process's selection: --limit / --only-sample-id
        # narrow a run without editing the protocol record.
        sample_count=len(read_jsonl(args.retrieval_file)),
        run_index=run_index_from_output_dir(args.output_dir),
        output_dir=args.output_dir,
    )
    return external_benchmarks_config.run_fingerprint(config)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_record_cache(directory: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return records
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid completed record {path}: {exc}") from exc
        if not isinstance(value, dict) or not str(value.get("sample_id") or "").strip():
            raise RuntimeError(f"Completed record has no sample_id: {path}")
        sample_id = str(value["sample_id"])
        if sample_id in records:
            raise RuntimeError(f"Duplicate completed record for {sample_id}")
        records[sample_id] = value
    return records


def merge_record_cache(
    output_path: Path,
    expected_rows: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    completed_records: dict[str, dict[str, Any]],
) -> tuple[bool, list[str]]:
    expected_ids = [str(row.get("sample_id")) for row in expected_rows]
    expected_id_set = set(expected_ids)
    records: dict[str, dict[str, Any]] = {}
    for row in existing:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise RuntimeError(f"Existing record has no sample_id: {output_path}")
        if sample_id in records:
            raise RuntimeError(f"Duplicate existing record for {sample_id}: {output_path}")
        records[sample_id] = row
    for sample_id, row in completed_records.items():
        if sample_id in records:
            continue
        records[sample_id] = row
    unexpected = sorted(set(records) - expected_id_set)
    if unexpected:
        raise RuntimeError(
            f"Records do not belong to the current selection: {unexpected[:5]}"
        )
    missing = [sample_id for sample_id in expected_ids if sample_id not in records]
    if missing:
        return False, missing
    write_jsonl_atomic(output_path, [records[sample_id] for sample_id in expected_ids])
    return True, []


def clear_merged_cache(
    directory: Path, merged_records: dict[str, dict[str, Any]]
) -> int:
    """Drop the completed records that a merge has just written into the output file.

    The cache means "records not yet in ``outputs.jsonl``" -- that is exactly what the
    overlap guard asserts when it refuses a sample that appears in both. Merging leaves
    the cache populated, so after one successful run the guard is wrong about a cell
    that is in fact fine, and the second invocation -- which should be a free no-op,
    since every row is already generated -- dies with "Samples exist in both outputs and
    completed_records" instead. That turns a rerun into a manual repair.

    Nothing is lost: these records are in the output file, byte for byte, and the
    output file is what every later stage reads.
    """
    removed = 0
    for sample_id, record in merged_records.items():
        path = directory / f"{sample_id}.json"
        if not path.is_file():
            continue
        # Guard the one case that would actually lose work: a file whose content is not
        # the record that was merged (a concurrent process wrote it after the read).
        try:
            on_disk = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if on_disk != record:
            continue
        path.unlink()
        removed += 1
    return removed


def retrieval_path(memory_system: str) -> Path:
    return RETRIEVAL_ROOT / SYSTEMS[memory_system]


def default_output_dir(model: str, memory_system: str, ablation: str) -> Path:
    experiment = "full" if ablation == "full" else ablation
    return ADAPTER_ROOT / "results" / experiment / model / memory_system


def prompt_config(ablation: str) -> tuple[str, tuple[str, ...]]:
    if ablation == "full":
        return FULL_PROMPT_VERSION, FULL_PROMPT_FILES
    if ablation not in ABLATION_PROMPT_FILES:
        raise ValueError(f"Unsupported ablation variant: {ablation}")
    return f"MemAdapter-{ablation}-20260926", ABLATION_PROMPT_FILES[ablation]


def prompt_hashes(ablation: str) -> dict[str, str]:
    _, files = prompt_config(ablation)
    return {
        name: hashlib.sha256((ADAPTER_ROOT / ("prompts" if ablation == "full" else "") / name).read_bytes()).hexdigest()
        for name in files
    }


def load_evidence(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = read_jsonl(path)
    evidence: dict[str, str] = {}
    for row in rows:
        sample_id = row.get("sample_id") or row.get("id")
        if sample_id is None:
            raise ValueError(f"Evidence row has no sample_id: {path}")
        unexpected = set(row) - {"sample_id", "evidence", "current_task_evidence"}
        if unexpected:
            raise ValueError(
                f"Evidence row {sample_id} has unsupported fields: {sorted(unexpected)}"
            )
        suspicious = {
            key
            for key in row
            if any(token in key.lower() for token in ("answer", "rubric", "judge", "score", "reference", "evaluation"))
        }
        if suspicious:
            raise ValueError(f"Evidence row {sample_id} contains evaluation-like fields: {sorted(suspicious)}")
        value = row.get("evidence", row.get("current_task_evidence", ""))
        if not isinstance(value, str):
            raise ValueError(f"Evidence row {sample_id} evidence must be a string")
        evidence[str(sample_id)] = str(value or "")
    return evidence


def dialogue_context(benchmark_row: dict[str, Any]) -> str:
    parts = []
    for message in benchmark_row.get("dialogue") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "assistant").strip().capitalize()
        content = str(message.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def normalize_retrieval_row(
    row: dict[str, Any], *, require_top_10: bool = False
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id") or row.get("id") or "")
    if not sample_id:
        raise ValueError("Retrieval row has no sample_id")
    benchmark_row = row.get("benchmark_row")
    if not isinstance(benchmark_row, dict):
        raise ValueError(f"Retrieval row {sample_id} has no benchmark_row object")
    task = str(row.get("task") or benchmark_row.get("task") or "")
    if task not in known_tasks():
        raise ValueError(f"Retrieval row {sample_id} has unsupported task: {task!r}")
    current_request = str(
        row.get("current_request") or benchmark_row.get("question") or ""
    ).strip()
    if not current_request:
        raise ValueError(f"Retrieval row {sample_id} has an empty current_request")
    memories = row.get("retrieved_memories")
    if not isinstance(memories, list):
        raise ValueError(f"Retrieval row {sample_id} has no retrieved_memories list")
    normalized = []
    for index, memory in enumerate(memories, start=1):
        if not isinstance(memory, dict):
            raise ValueError(f"Retrieval row {sample_id} contains a non-object memory")
        item = dict(memory)
        item["memory_id"] = str(item.get("memory_id") or f"memory_{index:02d}")
        item["memory_text"] = str(
            item.get("memory_text") or item.get("content") or item.get("raw_content") or ""
        )
        if not item["memory_text"].strip():
            raise ValueError(f"Retrieval row {sample_id} contains empty memory {item['memory_id']}")
        normalized.append(item)
    allowed = max_retrieved_memories(task)
    if not 1 <= len(normalized) <= allowed:
        raise ValueError(
            f"Retrieval row {sample_id} contains {len(normalized)} memories; "
            f"expected 1-{allowed} for task {task!r}"
        )
    memory_ids = [item["memory_id"] for item in normalized]
    if len(memory_ids) != len(set(memory_ids)):
        raise ValueError(f"Retrieval row {sample_id} has duplicate memory IDs")
    if require_top_10 and len(normalized) != 10:
        raise ValueError(
            f"Retrieval row {sample_id} contains {len(normalized)} memories instead of 10"
        )
    return {
        "sample_id": sample_id,
        "task": task,
        "current_request": current_request,
        "retrieved_memories": normalized,
        "benchmark_row": benchmark_row,
    }


def select_rows(
    rows: list[dict[str, Any]], per_task: int | None, *, seed: int | None = None
) -> list[dict[str, Any]]:
    if per_task is None:
        if seed is not None:
            raise ValueError("--seed requires --per-task-limit")
        return rows
    if per_task <= 0:
        raise ValueError("--per-task-limit must be positive")
    grouped = {task: [] for task in TASKS}
    for row in rows:
        task = str(row.get("task") or row.get("benchmark_row", {}).get("task") or "")
        if task in grouped:
            grouped[task].append(row)
    missing = {
        task: len(task_rows)
        for task, task_rows in grouped.items()
        if len(task_rows) < per_task
    }
    if missing:
        raise RuntimeError(f"Could not select {per_task} rows for every task: {missing}")
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for task in TASKS:
        task_rows = grouped[task]
        selected.extend(task_rows[:per_task] if seed is None else rng.sample(task_rows, per_task))
    return selected


def comparison_eligible_ids(*, include_m6: bool = True) -> set[str]:
    eligible: set[str] | None = None
    directories = [ROOT / "baseline" / "GPT" / "AMEM"]
    if include_m6:
        directories.append(ROOT / "ours" / "GPT" / "AMEM")
    for directory in directories:
        output_ids = {
            str(row.get("sample_id")) for row in read_jsonl(directory / "outputs.jsonl")
        }
        valid_judge_ids: set[str] = set()
        for row in read_jsonl(directory / "judge.jsonl"):
            task = row.get("task")
            if task not in TASKS:
                continue
            value = row.get("task_specific_pass")
            if not valid_binary(value):
                parsed = row.get("judge_parsed")
                value = parsed.get(PASS_METRICS[task]) if isinstance(parsed, dict) else None
            if valid_binary(value):
                valid_judge_ids.add(str(row.get("sample_id")))
        available = output_ids & valid_judge_ids
        eligible = available if eligible is None else eligible & available
    return eligible or set()


def select_comparable_rows(
    rows: list[dict[str, Any]],
    per_task: int | None,
    *,
    seed: int | None = None,
    include_m6: bool = True,
) -> list[dict[str, Any]]:
    if per_task is None:
        if seed is not None:
            raise ValueError("--seed requires --per-task-limit")
        return rows
    eligible = comparison_eligible_ids(include_m6=include_m6)
    comparable = [
        row
        for row in rows
        if normalize_retrieval_row(row)["sample_id"] in eligible
    ]
    return select_rows(comparable, per_task, seed=seed)


def verify_comparison_rows(selected: list[dict[str, Any]], *, include_m6: bool = True) -> None:
    expected = {normalize_retrieval_row(row)["sample_id"] for row in selected}
    directories = [("baseline", ROOT / "baseline" / "GPT" / "AMEM")]
    if include_m6:
        directories.append(("M6", ROOT / "ours" / "GPT" / "AMEM"))
    for label, directory in directories:
        for filename in ("outputs.jsonl", "judge.jsonl"):
            found = {
                str(row.get("sample_id"))
                for row in read_jsonl(directory / filename)
            }
            missing = sorted(expected - found)
            if missing:
                raise RuntimeError(
                    f"{label} comparison {filename} is missing selected sample IDs: "
                    f"{missing[:5]}"
                )


def retry_call(fn: Callable[[], str], label: str) -> str:
    attempts = int(os.environ.get("MEMADAPTER_RETRY_ATTEMPTS", "5"))
    max_wait = int(os.environ.get("MEMADAPTER_RETRY_MAX_WAIT", "120"))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            value = fn().strip()
            if value:
                return value
            last_error = RuntimeError("empty response")
        except Exception as exc:
            last_error = exc
        if attempt < attempts:
            wait = min(max_wait, 2 ** min(attempt, 6))
            retry_after = re.search(
                r"['\"]retry_after['\"]\s*:\s*(\d+)", str(last_error)
            )
            if retry_after:
                wait = min(max_wait, max(wait, int(retry_after.group(1))))
            print(f"[retry] {label} {attempt}/{attempts}, wait={wait}s, error={last_error}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last_error}") from last_error


def retry_call_traced(
    fn: Callable[[int], Any],
    *,
    recorder: EfficiencyRecorder,
    stage: str,
    label: str,
) -> str:
    """Retry a generation request while preserving every attempt in the trace."""

    attempts = int(os.environ.get("MEMADAPTER_RETRY_ATTEMPTS", "5"))
    max_wait = int(os.environ.get("MEMADAPTER_RETRY_MAX_WAIT", "120"))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = fn(attempt)
            text = str(getattr(result, "text", result) or "").strip()
            if text:
                return text
            last_error = RuntimeError("empty response")
        except Exception as exc:
            last_error = exc
        if attempt < attempts:
            wait = min(max_wait, 2 ** min(attempt, 6))
            retry_after = re.search(r"['\"]retry_after['\"]\s*:\s*(\d+)", str(last_error))
            if retry_after:
                wait = min(max_wait, max(wait, int(retry_after.group(1))))
            print(
                f"[retry] {label} {attempt}/{attempts}, wait={wait}s, error={last_error}",
                flush=True,
            )
            recorder.add_retry_sleep(wait * 1000)
            time.sleep(wait)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last_error}") from last_error


def safe_task_evidence(benchmark_row: dict[str, Any]) -> str:
    """Do not synthesize task evidence from benchmark classification metadata."""

    return ""


def filter_skipped_rows(
    rows: list[dict[str, Any]], skip_sample_ids: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Remove explicitly failed samples while preserving the selected order."""

    skip_ids = {str(sample_id) for sample_id in skip_sample_ids if str(sample_id).strip()}
    if not skip_ids:
        return rows, []
    skipped: list[str] = []
    filtered: list[dict[str, Any]] = []
    for row in rows:
        sample_id = normalize_retrieval_row(row)["sample_id"]
        if sample_id in skip_ids:
            skipped.append(sample_id)
        else:
            filtered.append(row)
    return filtered, skipped


def filter_empty_external_retrieval_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Exclude external rows for which a memory system returned no candidates.

    Native MemSyco retrieval keeps its existing fail-fast contract. External
    rows with no retrieved memory cannot support a memory-use comparison, so the
    runner records the exclusion and continues with the usable rows.
    """
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for row in rows:
        benchmark = row.get("benchmark_row")
        task = str(row.get("task") or (benchmark.get("task") if isinstance(benchmark, dict) else "") or "")
        memories = row.get("retrieved_memories")
        if is_external_task(task) and isinstance(memories, list) and not memories:
            excluded.append(
                {
                    "sample_id": str(row.get("sample_id") or row.get("id") or "<unknown>"),
                    "task": task,
                    "reason": "empty_retrieval",
                }
            )
            continue
        kept.append(row)
    return kept, excluded


def filter_only_rows(
    rows: list[dict[str, Any]], only_sample_ids: list[str]
) -> list[dict[str, Any]]:
    """Restrict a selected set to explicitly requested sample IDs."""

    only_ids = {str(sample_id) for sample_id in only_sample_ids if str(sample_id).strip()}
    if not only_ids:
        return rows
    filtered = [
        row
        for row in rows
        if normalize_retrieval_row(row)["sample_id"] in only_ids
    ]
    found_ids = {normalize_retrieval_row(row)["sample_id"] for row in filtered}
    missing = sorted(only_ids - found_ids)
    if missing:
        raise RuntimeError(f"Requested sample IDs were not selected: {missing}")
    return filtered


def generate(args: argparse.Namespace) -> None:
    version, _ = prompt_config(args.ablation)
    selected_rows = select_comparable_rows(
        read_jsonl(args.retrieval_file),
        args.per_task_limit,
        seed=args.seed,
        include_m6=not args.baseline_only,
    )
    if not selected_rows:
        raise RuntimeError(f"No retrieval rows found: {args.retrieval_file}")
    if args.per_task_limit is not None:
        verify_comparison_rows(selected_rows, include_m6=not args.baseline_only)
    selected_rows = filter_only_rows(selected_rows, args.only_sample_id)
    retrieval_rows, skipped_ids = filter_skipped_rows(
        selected_rows, args.skip_sample_id
    )
    retrieval_rows, empty_retrieval_exclusions = filter_empty_external_retrieval_rows(
        retrieval_rows
    )
    if not retrieval_rows:
        raise RuntimeError("All selected retrieval rows were skipped")
    # External cells get their protocol record written now, before any model call, so
    # a drifted environment cannot produce a record that does not describe the run.
    external_fingerprint = ensure_external_protocol(args, retrieval_rows)
    if skipped_ids:
        print(
            f"[generate] skipped failed samples: {', '.join(skipped_ids)}",
            flush=True,
        )
    if empty_retrieval_exclusions:
        write_json(
            args.output_dir / "empty_retrieval_exclusions.json",
            {"reason": "empty_retrieval", "rows": empty_retrieval_exclusions},
        )
        print(
            f"[generate] excluded {len(empty_retrieval_exclusions)} external rows with empty retrieval",
            flush=True,
        )
    if args.per_task_limit is not None:
        write_json(
            args.output_dir / "selection_manifest.json",
            {
                "selection": (
                    f"random {args.per_task_limit} rows per task"
                    if args.seed is not None
                    else f"first {args.per_task_limit} rows per task in retrieval order"
                ),
                "selection_seed": args.seed,
                "comparison_methods": ["baseline"] if args.baseline_only else ["baseline", "M6"],
                "model_for_comparison": "GPT",
                "requested_sample_count": len(selected_rows),
                "skipped_sample_ids": skipped_ids,
                "memory_system": "AMEM",
                "sample_ids": [normalize_retrieval_row(row)["sample_id"] for row in retrieval_rows],
                "samples": [
                    {
                        "sample_id": normalized["sample_id"],
                        "task": normalized["task"],
                        "retrieved_count": len(normalized["retrieved_memories"]),
                    }
                    for normalized in (
                        normalize_retrieval_row(row) for row in retrieval_rows
                    )
                ],
                "rows_by_task": {
                    task: sum(1 for row in retrieval_rows if row.get("task") == task)
                    for task in TASKS
                },
            },
        )
    if args.limit is not None:
        retrieval_rows = retrieval_rows[: args.limit]
    output_file = args.output_dir / "outputs.jsonl"
    existing = read_jsonl(output_file)
    expected_ids = [
        normalize_retrieval_row(row, require_top_10=args.require_top_10)["sample_id"]
        for row in retrieval_rows
    ]
    existing_ids = [str(row.get("sample_id")) for row in existing]
    hashes = prompt_hashes(args.ablation)
    for row in existing:
        if row.get("prompt_version") != version:
            raise RuntimeError(
                "Existing outputs use a different prompt version; use a new output directory."
            )
        if row.get("prompt_hashes") != hashes:
            raise RuntimeError(
                "Existing outputs were generated with different prompt files; use a new output directory."
            )
    if len(existing_ids) != len(set(existing_ids)):
        raise RuntimeError("Existing MemAdapter outputs contain duplicate sample IDs")
    if args.continue_on_error:
        unexpected = sorted(set(existing_ids) - set(expected_ids))
        if unexpected:
            raise RuntimeError(
                f"Existing MemAdapter outputs are outside the current selection: {unexpected[:5]}"
            )
        completed_dir = args.output_dir / "completed_records"
        completed_records = read_record_cache(completed_dir)
        unexpected = sorted(set(completed_records) - set(expected_ids))
        if unexpected:
            raise RuntimeError(
                f"Completed generation records are outside the current selection: {unexpected[:5]}"
            )
        overlap = sorted(set(existing_ids) & set(completed_records))
        if overlap:
            raise RuntimeError(
                f"Samples exist in both outputs and completed_records: {overlap[:5]}"
            )
    else:
        if existing_ids != expected_ids[: len(existing_ids)]:
            raise RuntimeError("Existing MemAdapter outputs are not an ordered retrieval prefix")
        completed_records = {}
        completed_dir = args.output_dir / "completed_records"

    evidence = load_evidence(args.evidence_file)
    thread_state = threading.local()
    run_id = new_run_id("generation")
    run_started_at = utc_now()

    def get_client() -> ModelClient:
        client = getattr(thread_state, "client", None)
        if client is None:
            client = ModelClient(args.model)
            thread_state.client = client
        return client

    def get_stage3_client() -> ModelClient:
        client = getattr(thread_state, "stage3_client", None)
        if client is None:
            client = ModelClient(os.environ.get("MEMADAPTER_STAGE3_MODEL", "DeepSeek"))
            thread_state.stage3_client = client
        return client

    def build_record(raw_row: dict[str, Any]) -> dict[str, Any]:
        row = normalize_retrieval_row(raw_row, require_top_10=args.require_top_10)
        client = get_client()
        recorder = EfficiencyRecorder(
            output_dir=args.output_dir,
            sample_id=row["sample_id"],
            task=row["task"],
            method="MemAdapter",
            model=args.model,
            memory_system=args.memory_system,
            phase="generation",
            run_id=run_id,
            run_started_at=run_started_at,
            base_url=client.base_url,
        )
        benchmark = row["benchmark_row"]
        task_evidence = evidence.get(
            row["sample_id"], safe_task_evidence(benchmark)
        )
        print(
            f"[evidence] {row['sample_id']} current_task_evidence={task_evidence!r}",
            flush=True,
        )
        cache_path = args.output_dir / "stage_cache" / f"{row['sample_id']}.json"
        stage_cache: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "prompt_version": version,
            "prompt_hashes": hashes,
            "ablation": args.ablation,
        }
        if external_fingerprint is not None:
            # External cells only: the fingerprint covers the dataset, system,
            # retrieval file and sampling settings, none of which the native cache
            # keys capture. Native caches predate the field and stay byte-identical.
            stage_cache["run_fingerprint"] = external_fingerprint
        if cache_path.is_file():
            try:
                loaded_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Invalid stage cache {cache_path}: {exc}") from exc
            if (
                loaded_cache.get("sample_id") != row["sample_id"]
                or loaded_cache.get("prompt_version") != version
                or loaded_cache.get("prompt_hashes") != hashes
                or loaded_cache.get("ablation", "full") != args.ablation
                or (
                    external_fingerprint is not None
                    and loaded_cache.get("run_fingerprint") != external_fingerprint
                )
            ):
                raise RuntimeError(f"Stage cache metadata mismatch: {cache_path}")
            stage_cache.update(loaded_cache)

        def save_stage(stage_name: str, raw: str) -> None:
            stage_cache[stage_name] = raw
            write_json_atomic(cache_path, stage_cache)
            print(
                f"[stage] {row['sample_id']} {stage_name} saved",
                flush=True,
            )

        def traced_call(system: str, user: str, **kwargs: Any) -> str:
            stage = str(kwargs.pop("efficiency_stage", "generation"))
            selected_client = get_stage3_client() if stage == "stage3" and os.environ.get("MEMADAPTER_STAGE3_MODEL") else client
            result = retry_call_traced(
                lambda _attempt: _call_generation(
                    selected_client,
                    recorder,
                    system,
                    user,
                    stage=stage,
                    attempt=recorder.next_attempt_number(stage),
                    **kwargs,
                ),
                recorder=recorder,
                stage=stage,
                label=f"{selected_client.model_name} {args.memory_system} {stage}",
            )
            return result

        def _call_generation(
            model_client: ModelClient,
            sample_recorder: EfficiencyRecorder,
            system: str,
            user: str,
            *,
            stage: str,
            attempt: int,
            **kwargs: Any,
        ) -> Any:
            try:
                result = model_client.complete_result(system, user, **kwargs)
                sample_recorder.record_success(
                    result, stage=stage, attempt=attempt
                )
                return result
            except Exception as exc:
                # The ModelClient request itself is timed below only for failures;
                # successful calls carry the precise timing from complete_result.
                sample_recorder.record_failure(
                    stage=stage,
                    attempt=attempt,
                    request_started_at=getattr(exc, "_efficiency_request_started_at", utc_now()),
                    response_received_at=getattr(exc, "_efficiency_response_received_at", utc_now()),
                    latency_ms=float(getattr(exc, "_efficiency_latency_ms", 0.0)),
                    error=exc,
                    model=model_client.model,
                    base_url=model_client.base_url,
                    temperature=model_client.temperature,
                    max_tokens=kwargs.get("max_tokens") or model_client.max_tokens,
                    input_tokens=getattr(exc, "_efficiency_input_tokens", None),
                    output_tokens=getattr(exc, "_efficiency_output_tokens", None),
                    total_tokens=getattr(exc, "_efficiency_total_tokens", None),
                    cached_input_tokens=getattr(exc, "_efficiency_cached_input_tokens", None),
                    reasoning_tokens=getattr(exc, "_efficiency_reasoning_tokens", None),
                    usage_available=bool(getattr(exc, "_efficiency_usage_available", False)),
                    finish_reason=getattr(exc, "_efficiency_finish_reason", None),
                )
                raise

        try:
            run_kwargs = {
                "memories": row["retrieved_memories"],
                "current_query": row["current_request"],
                "dialogue_context": dialogue_context(benchmark),
                "current_task_evidence": task_evidence,
                "call_model": traced_call,
                "stage_raw_cache": {
                    key: str(stage_cache.get(key) or "")
                    for key in ("stage1_raw", "stage2_raw", "stage3_raw")
                },
                "on_stage_complete": save_stage,
                "on_stage_start": recorder.start_stage,
                "on_stage_finish": recorder.finish_stage,
            }
            result = (
                run_three_stage(**run_kwargs)
                if args.ablation == "full"
                else run_ablation(variant=args.ablation, **run_kwargs)
            )
        except BaseException as exc:
            if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                recorder.finish(success=False, error=exc)
                recorder.persist()
            raise
        efficiency = recorder.finish(success=True)
        recorder.persist()
        record = {
            "sample_id": row["sample_id"],
            "task": row["task"],
            "memory_system": args.memory_system,
            "model": args.model,
            "ablation": args.ablation,
            "current_request": row["current_request"],
            "retrieved_memories": result["retrieved_memories"],
            "retrieved_count": len(result["retrieved_memories"]),
            "memory_boundary_cards": result["memory_boundary_cards"],
            "memory_boundary_cards_raw": result["stage1_raw"],
            "memory_role_assignments": result["memory_role_assignments"],
            "memory_role_assignments_raw": result["stage2_raw"],
            "memory_use_instructions": result["memory_use_instructions"],
            "final_answer": result["final_answer"],
            "final_answer_raw": result["stage3_raw"],
            "current_task_evidence": task_evidence,
            "prompt_version": version,
            "prompt_hashes": hashes,
            "generation_model": client.model,
            "generation_base_url": client.base_url,
            "generation_temperature": client.temperature,
            "generation_max_tokens": client.max_tokens,
            "generation_runtime": dict(getattr(client, "runtime_config", {})),
            "generation_efficiency": efficiency,
            "benchmark_row": benchmark,
        }
        if external_fingerprint is not None:
            # Same value on both arms (`run_fingerprint` excludes the arm), so a
            # validation pass can assert the two arms' records describe one protocol.
            record["run_fingerprint"] = external_fingerprint
            # External runs have two arms and the judge is invoked for one of them by
            # name, so a record has to say which arm produced it -- otherwise the
            # judge's arm check cannot distinguish this file from a baseline one it
            # was pointed at by mistake. Native MemSyco runs have a single arm and
            # frozen records without this key, hence external-only.
            record["arm"] = "memadapter"
        return record

    if args.continue_on_error:
        completed_ids = set(existing_ids) | set(completed_records)
        remaining = [
            (index, row)
            for index, row in enumerate(retrieval_rows)
            if normalize_retrieval_row(row)["sample_id"] not in completed_ids
        ]
    else:
        remaining = list(enumerate(retrieval_rows[len(existing) :], start=len(existing)))
    if not remaining:
        if args.continue_on_error:
            merged, missing = merge_record_cache(
                output_file, retrieval_rows, existing, completed_records
            )
            if not merged:
                raise RuntimeError(
                    f"Generation has no pending jobs but is missing records: {missing[:5]}"
                )
            print(f"[generate] merged complete cache: {len(retrieval_rows)} rows", flush=True)
        else:
            print(f"[generate] already complete: {len(existing)} rows", flush=True)
        rebuild_efficiency_artifacts(args.output_dir)
        return
    workers = max(1, min(args.workers, len(remaining)))
    print(f"[generate] workers={workers}, rows={len(remaining)}", flush=True)
    pending: dict[int, dict[str, Any]] = {}
    failures: list[tuple[int, Exception]] = []
    next_to_write = len(existing)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_record, row): index for index, row in remaining}
        for future in as_completed(futures):
            index = futures[future]
            try:
                record = future.result()
                if args.continue_on_error:
                    write_json_atomic(
                        completed_dir / f"{record['sample_id']}.json", record
                    )
                    print(
                        f"[generate] completed {index + 1}/{len(retrieval_rows)} "
                        f"{record['task']} {record['sample_id']} "
                        "stored=completed_records",
                        flush=True,
                    )
                else:
                    pending[index] = record
            except Exception as exc:
                failures.append((index, exc))
                print(f"[error] generation index={index + 1}: {exc}", flush=True)
                if args.continue_on_error:
                    row = retrieval_rows[index]
                    normalized = normalize_retrieval_row(row)
                    append_jsonl(
                        args.output_dir / "generation_failures.jsonl",
                        {
                            "phase": "generation",
                            "status": "failed",
                            "position": index + 1,
                            "sample_id": normalized["sample_id"],
                            "task": normalized["task"],
                            "model": args.model,
                            "memory_system": args.memory_system,
                            "error": str(exc),
                            "recorded_at": utc_now(),
                        },
                    )
            if not args.continue_on_error:
                while next_to_write in pending:
                    record = pending.pop(next_to_write)
                    append_jsonl(output_file, record)
                    print(
                        f"[generate] {next_to_write + 1}/{len(retrieval_rows)} "
                        f"{record['task']} {record['sample_id']}",
                        flush=True,
                    )
                    next_to_write += 1
    if args.continue_on_error:
        rebuild_efficiency_artifacts(args.output_dir)
        if failures:
            first_index, first_error = sorted(failures, key=lambda item: item[0])[0]
            raise RuntimeError(
                f"{len(failures)} generation rows failed after later rows continued; "
                f"first index={first_index + 1}: {first_error}"
            ) from first_error
        completed_records = read_record_cache(completed_dir)
        merged, missing = merge_record_cache(
            output_file, retrieval_rows, existing, completed_records
        )
        if not merged:
            raise RuntimeError(
                f"Generation completed without failures but is missing records: {missing[:5]}"
            )
        print(f"[generate] merged complete cache: {len(retrieval_rows)} rows", flush=True)
        clear_merged_cache(completed_dir, completed_records)
        rebuild_efficiency_artifacts(args.output_dir)
        return
    if failures:
        rebuild_efficiency_artifacts(args.output_dir)
        first_index, first_error = sorted(failures, key=lambda item: item[0])[0]
        raise RuntimeError(
            f"{len(failures)} generation rows failed; first index={first_index + 1}: {first_error}"
        ) from first_error
    rebuild_efficiency_artifacts(args.output_dir)


def judge_settings(model_name: str) -> dict[str, str]:
    prefix = {"DeepSeek": "DEEPSEEK", "GPT": "EVAL", "Qwen": "QWEN"}[model_name]
    defaults = MODEL_DEFAULTS[model_name]
    key = os.environ.get("JUDGE_API_KEY", "").strip() or os.environ.get(f"{prefix}_JUDGE_API_KEY", "").strip()
    key = key or os.environ.get(f"{prefix}_API_KEY", "").strip()
    key = key or os.environ.get("OPENAI_API_KEY", "").strip()
    base = os.environ.get("JUDGE_BASE_URL", "").strip() or os.environ.get(f"{prefix}_JUDGE_BASE_URL", "").strip()
    base = base or os.environ.get(f"{prefix}_BASE_URL", "").strip()
    base = base or os.environ.get("OPENAI_BASE_URL", "").strip()
    base = _openai_base_url(base or defaults["default_base_url"])
    model = os.environ.get("JUDGE_MODEL", "").strip() or os.environ.get(f"{prefix}_JUDGE_MODEL", "").strip()
    model = model or os.environ.get(f"{prefix}_MODEL", "").strip()
    model = model or os.environ.get("OPENAI_MODEL", "").strip()
    model = model or defaults["default_model"]
    api_mode = os.environ.get("JUDGE_API_MODE", "").strip().lower().replace("-", "_")
    if api_mode in {"response", "responses"}:
        api_mode = "responses"
    else:
        api_mode = "chat_completions"
    if not key:
        raise RuntimeError(
            "Judge API key is missing. Set JUDGE_API_KEY, "
            f"{prefix}_JUDGE_API_KEY, {prefix}_API_KEY, or OPENAI_API_KEY."
        )
    return {"api_key": key, "base_url": base, "model": model, "api_mode": api_mode}


def judge(args: argparse.Namespace) -> None:
    outputs = read_jsonl(args.output_dir / "outputs.jsonl")
    if args.limit is not None:
        outputs = outputs[: args.limit]
    if not outputs:
        raise RuntimeError("No MemAdapter outputs found")
    refuse_native_stage("judge", external_tasks_in(outputs))
    judge_file = args.output_dir / "judge.jsonl"
    existing = read_jsonl(judge_file)
    output_ids = [str(row.get("sample_id")) for row in outputs]
    existing_ids = [str(row.get("sample_id")) for row in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise RuntimeError("Existing MemAdapter judge rows contain duplicate sample IDs")
    if args.continue_on_error:
        unexpected = sorted(set(existing_ids) - set(output_ids))
        if unexpected:
            raise RuntimeError(
                f"Existing MemAdapter judge rows are outside the current outputs: {unexpected[:5]}"
            )
        completed_dir = args.output_dir / "completed_judge_records"
        completed_records = read_record_cache(completed_dir)
        unexpected = sorted(set(completed_records) - set(output_ids))
        if unexpected:
            raise RuntimeError(
                f"Completed judge records are outside the current outputs: {unexpected[:5]}"
            )
        overlap = sorted(set(existing_ids) & set(completed_records))
        if overlap:
            raise RuntimeError(
                f"Samples exist in both judge.jsonl and completed_judge_records: {overlap[:5]}"
            )
    else:
        if existing_ids != output_ids[: len(existing_ids)]:
            raise RuntimeError("Existing MemAdapter judge rows are not an ordered output prefix")
        completed_records = {}
        completed_dir = args.output_dir / "completed_judge_records"

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The openai package is required for judging. Install benchmark/requirements.txt first."
        ) from exc
    cfg = judge_settings(args.model)
    thread_state = threading.local()
    run_id = new_run_id("judge")
    run_started_at = utc_now()

    def get_client() -> Any:
        client = getattr(thread_state, "client", None)
        if client is None:
            client = OpenAI(
                api_key=cfg["api_key"],
                base_url=cfg["base_url"],
                timeout=180,
                max_retries=0,
            )
            thread_state.client = client
        return client

    def build_judge_record(output: dict[str, Any]) -> dict[str, Any]:
        recorder = EfficiencyRecorder(
            output_dir=args.output_dir,
            sample_id=str(output["sample_id"]),
            task=str(output["task"]),
            method=str(output.get("method") or "MemAdapter"),
            model=args.model,
            memory_system=str(output["memory_system"]),
            phase="judge",
            run_id=run_id,
            run_started_at=run_started_at,
            base_url=cfg["base_url"],
        )
        task = output["task"]
        module = importlib.import_module(
            "task_objective_fact_judgment"
            if task == "objective_fact_judgment"
            else f"task_{task}"
        )
        fn = getattr(module, "judge_objective_fact_answer" if task == "objective_fact_judgment" else "judge_answer")
        eval_row = to_eval_row(output["benchmark_row"])
        started = time.perf_counter()
        try:
            result = fn(
                TracedOpenAI(
                    get_client(),
                    recorder,
                    stage="judge",
                    api_mode=cfg["api_mode"],
                ),
                cfg["model"],
                eval_row,
                output["final_answer"],
                cache_base_url=cfg["base_url"],
            )
        except BaseException as exc:
            if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                recorder.finish(success=False, error=exc)
                recorder.persist()
            raise
        correctness = result.get(CORRECTNESS_METRICS[task])
        task_pass = result.get(PASS_METRICS[task])
        if not valid_binary(correctness) or not valid_binary(task_pass):
            error = RuntimeError(
                f"Judge returned invalid metrics for {output['sample_id']}: "
                f"correctness={correctness!r}, task_pass={task_pass!r}, "
                f"error={result.get('judge_error')!r}"
            )
            recorder.finish(success=False, error=error)
            recorder.persist()
            raise error
        recorder.finish_stage(
            "judge", (time.perf_counter() - started) * 1000, cache_hit=False
        )
        efficiency = recorder.finish(success=True)
        recorder.persist()
        return {
            "sample_id": output["sample_id"],
            "task": task,
            "memory_system": output["memory_system"],
            "model": output["model"],
            "method": str(output.get("method") or "MemAdapter"),
            "final_answer": output["final_answer"],
            "judge_model": cfg["model"],
            "judge_base_url": cfg["base_url"],
            "judge_api_mode": cfg["api_mode"],
            "judge_parsed": result,
            "judge_raw": result.get("judge_raw", ""),
            "correctness": correctness,
            "task_specific_pass": task_pass,
            "judge_efficiency": efficiency,
        }

    if args.continue_on_error:
        completed_ids = set(existing_ids) | set(completed_records)
        remaining = [
            (index, output)
            for index, output in enumerate(outputs)
            if str(output.get("sample_id")) not in completed_ids
        ]
    else:
        remaining = list(enumerate(outputs[len(existing) :], start=len(existing)))
    if not remaining:
        if args.continue_on_error:
            merged, missing = merge_record_cache(
                judge_file, outputs, existing, completed_records
            )
            if not merged:
                raise RuntimeError(
                    f"Judge has no pending jobs but is missing records: {missing[:5]}"
                )
            print(f"[judge] merged complete cache: {len(outputs)} rows", flush=True)
        else:
            print(f"[judge] already complete: {len(existing)} rows", flush=True)
        rebuild_efficiency_artifacts(args.output_dir)
        return
    workers = max(1, min(args.workers, len(remaining)))
    print(
        f"[judge] workers={workers}, rows={len(remaining)}, model={cfg['model']}",
        flush=True,
    )
    pending: dict[int, dict[str, Any]] = {}
    failures: list[tuple[int, Exception]] = []
    next_to_write = len(existing)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(build_judge_record, output): index
            for index, output in remaining
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                record = future.result()
                if args.continue_on_error:
                    write_json_atomic(
                        completed_dir / f"{record['sample_id']}.json", record
                    )
                    print(
                        f"[judge] completed {index + 1}/{len(outputs)} "
                        f"{record['task']} {record['sample_id']} "
                        "stored=completed_judge_records",
                        flush=True,
                    )
                else:
                    pending[index] = record
            except Exception as exc:
                failures.append((index, exc))
                print(f"[error] judge index={index + 1}: {exc}", flush=True)
                if args.continue_on_error:
                    output = outputs[index]
                    append_jsonl(
                        args.output_dir / "judge_failures.jsonl",
                        {
                            "phase": "judge",
                            "status": "failed",
                            "position": index + 1,
                            "sample_id": str(output["sample_id"]),
                            "task": str(output["task"]),
                            "model": args.model,
                            "memory_system": str(output["memory_system"]),
                            "error": str(exc),
                            "recorded_at": utc_now(),
                        },
                    )
            if not args.continue_on_error:
                while next_to_write in pending:
                    record = pending.pop(next_to_write)
                    append_jsonl(judge_file, record)
                    print(
                        f"[judge] {next_to_write + 1}/{len(outputs)} "
                        f"{record['task']} {record['sample_id']}",
                        flush=True,
                    )
                    next_to_write += 1
    if args.continue_on_error:
        rebuild_efficiency_artifacts(args.output_dir)
        if failures:
            first_index, first_error = sorted(failures, key=lambda item: item[0])[0]
            raise RuntimeError(
                f"{len(failures)} judge rows failed after later rows continued; "
                f"first index={first_index + 1}: {first_error}"
            ) from first_error
        completed_records = read_record_cache(completed_dir)
        merged, missing = merge_record_cache(
            judge_file, outputs, existing, completed_records
        )
        if not merged:
            raise RuntimeError(
                f"Judge completed without failures but is missing records: {missing[:5]}"
            )
        print(f"[judge] merged complete cache: {len(outputs)} rows", flush=True)
        clear_merged_cache(completed_dir, completed_records)
        rebuild_efficiency_artifacts(args.output_dir)
        return
    if failures:
        rebuild_efficiency_artifacts(args.output_dir)
        first_index, first_error = sorted(failures, key=lambda item: item[0])[0]
        raise RuntimeError(
            f"{len(failures)} judge rows failed; first index={first_index + 1}: "
            f"{first_error}"
        ) from first_error
    rebuild_efficiency_artifacts(args.output_dir)


def valid_binary(value: Any) -> bool:
    return isinstance(value, bool) or (isinstance(value, int) and value in (0, 1))


def summary(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.output_dir / "judge.jsonl")
    if args.limit is not None:
        rows = rows[: args.limit]
    refuse_native_stage("summary", external_tasks_in(rows))
    by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    for row in rows:
        if row.get("task") in by_task:
            by_task[row["task"]].append(row)
    task_summary: dict[str, Any] = {}
    values = []
    for task, task_rows in by_task.items():
        metric = PASS_METRICS[task]
        valid = [r.get("judge_parsed", {}).get(metric) for r in task_rows if valid_binary(r.get("judge_parsed", {}).get(metric))]
        value = (sum(int(v) for v in valid) / len(valid)) if valid else None
        task_summary[task] = {"n": len(task_rows), "pass_metric": metric, "pass_value": value, "pass_valid_n": len(valid)}
        if value is not None and len(valid) > 0:
            values.append(value)
    macro_complete = all(task_summary[task]["pass_value"] is not None for task in TASKS)
    write_json(
        args.output_dir / "summary.json",
        {
            "experiment": f"MemAdapter {args.ablation}",
            "ablation": args.ablation,
            "prompt_version": prompt_config(args.ablation)[0],
            "prompt_hashes": prompt_hashes(args.ablation),
            "model": args.model,
            "memory_system": args.memory_system,
            "methods": {"MemAdapter": {
                "macro_task_pass": sum(values) / len(TASKS) if macro_complete else None,
                "macro_complete": macro_complete,
                "tasks": task_summary,
            }},
        },
    )
    rebuild_efficiency_artifacts(args.output_dir)


def validate(args: argparse.Namespace) -> None:
    retrieval = select_comparable_rows(
        read_jsonl(args.retrieval_file),
        args.per_task_limit,
        seed=args.seed,
        include_m6=not args.baseline_only,
    )
    if args.limit is not None:
        retrieval = retrieval[: args.limit]
    refuse_native_stage("validate", external_tasks_in(retrieval))
    retrieval, _ = filter_skipped_rows(retrieval, args.skip_sample_id)
    outputs = read_jsonl(args.output_dir / "outputs.jsonl")
    judges = read_jsonl(args.output_dir / "judge.jsonl")
    retrieval_ids = [str(row.get("sample_id")) for row in retrieval]
    output_ids = [str(row.get("sample_id")) for row in outputs]
    judge_ids = [str(row.get("sample_id")) for row in judges]
    errors: list[str] = []
    for row in retrieval:
        try:
            normalize_retrieval_row(row, require_top_10=args.require_top_10)
        except ValueError as exc:
            errors.append(str(exc))
    if output_ids != retrieval_ids[: len(output_ids)]:
        errors.append("outputs are not an ordered retrieval prefix")
    if judge_ids != output_ids[: len(judge_ids)]:
        errors.append("judge rows are not an ordered output prefix")
    if len(outputs) != len(retrieval):
        errors.append(f"expected {len(retrieval)} outputs, found {len(outputs)}")
    if len(judges) != len(outputs):
        errors.append(f"expected {len(outputs)} judge rows, found {len(judges)}")
    version, _ = prompt_config(args.ablation)
    hashes = prompt_hashes(args.ablation)
    for row in outputs:
        if row.get("prompt_version") != version:
            errors.append(f"{row.get('sample_id')}: prompt_version mismatch")
        if row.get("prompt_hashes") != hashes:
            errors.append(f"{row.get('sample_id')}: prompt_hashes mismatch")
        required = ("final_answer",)
        if args.ablation == "full":
            required = ("memory_boundary_cards", "memory_role_assignments", "final_answer")
        elif args.ablation == "stage1-baseline":
            required = ("memory_boundary_cards", "final_answer")
        elif args.ablation == "stage1-stage2-baseline":
            required = ("memory_boundary_cards", "memory_role_assignments", "final_answer")
        for key in required:
            if not row.get(key):
                errors.append(f"{row.get('sample_id')}: missing {key}")
    for row in judges:
        task = row.get("task")
        parsed = row.get("judge_parsed")
        if task not in TASKS or not isinstance(parsed, dict):
            errors.append(f"{row.get('sample_id')}: invalid judge record")
            continue
        if not valid_binary(parsed.get(CORRECTNESS_METRICS[task])):
            errors.append(f"{row.get('sample_id')}: invalid correctness metric")
        if not valid_binary(parsed.get(PASS_METRICS[task])):
            errors.append(f"{row.get('sample_id')}: invalid task pass metric")
    report = {
        "model": args.model,
        "memory_system": args.memory_system,
        "ablation": args.ablation,
        "retrieval_rows": len(retrieval),
        "output_rows": len(outputs),
        "judge_rows": len(judges),
        "complete": not errors,
        "errors": errors,
    }
    write_json(args.output_dir / "validation_report.json", report)
    if errors:
        raise RuntimeError("Validation failed: " + "; ".join(errors[:5]))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def compare(args: argparse.Namespace) -> None:
    manifest_path = args.output_dir / "selection_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Selection manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("Selection manifest has no samples")
    selected = {
        str(item["sample_id"]): str(item["task"])
        for item in samples
        if isinstance(item, dict) and item.get("sample_id") and item.get("task")
    }
    if len(selected) != len(samples):
        raise RuntimeError("Selection manifest contains invalid or duplicate samples")
    refuse_native_stage(
        "compare", sorted({task for task in selected.values() if is_external_task(task)})
    )

    sources = {"baseline": ROOT / "baseline" / "GPT" / "AMEM" / "judge.jsonl"}
    if not args.baseline_only:
        sources["M6"] = ROOT / "ours" / "GPT" / "AMEM" / "judge.jsonl"
    sources["MemAdapter"] = args.output_dir / "judge.jsonl"
    method_values: dict[str, dict[str, bool]] = {}
    method_summary: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for method, path in sources.items():
        rows_by_id = {
            str(row.get("sample_id")): row
            for row in read_jsonl(path)
            if str(row.get("sample_id")) in selected
        }
        values: dict[str, bool] = {}
        tasks: dict[str, Any] = {}
        rates: list[float] = []
        for task in TASKS:
            expected_ids = [sample_id for sample_id, name in selected.items() if name == task]
            valid: list[bool] = []
            for sample_id in expected_ids:
                row = rows_by_id.get(sample_id)
                if row is None:
                    continue
                value = row.get("task_specific_pass")
                if not valid_binary(value):
                    parsed = row.get("judge_parsed")
                    value = parsed.get(PASS_METRICS[task]) if isinstance(parsed, dict) else None
                if valid_binary(value):
                    values[sample_id] = bool(value)
                    valid.append(bool(value))
            rate = sum(valid) / len(valid) if valid else None
            complete = len(valid) == len(expected_ids)
            tasks[task] = {
                "expected_n": len(expected_ids),
                "valid_n": len(valid),
                "pass_rate": rate,
                "complete": complete,
            }
            if complete and rate is not None:
                rates.append(rate)
            csv_rows.append(
                {
                    "method": method,
                    "task": task,
                    "expected_n": len(expected_ids),
                    "valid_n": len(valid),
                    "pass_rate": "" if rate is None else f"{rate:.6f}",
                    "complete": complete,
                }
            )
        complete = len(rates) == len(TASKS)
        method_values[method] = values
        method_summary[method] = {
            "complete": complete,
            "macro_task_pass": sum(rates) / len(TASKS) if complete else None,
            "tasks": tasks,
        }

    pairwise: dict[str, Any] = {}
    adapter = method_values["MemAdapter"]
    references = ("baseline",) if args.baseline_only else ("baseline", "M6")
    for reference in references:
        reference_values = method_values[reference]
        common = [sample_id for sample_id in selected if sample_id in adapter and sample_id in reference_values]
        wins = sum(adapter[sample_id] and not reference_values[sample_id] for sample_id in common)
        losses = sum(not adapter[sample_id] and reference_values[sample_id] for sample_id in common)
        ties = len(common) - wins - losses
        pairwise[f"MemAdapter_vs_{reference}"] = {
            "paired_n": len(common),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "net_wins": wins - losses,
        }

    report = {
        "selection_n": len(selected),
        "selection_seed": manifest.get("selection_seed"),
        "comparison_methods": list(sources),
        "selection_rows_by_task": {
            task: sum(name == task for name in selected.values()) for task in TASKS
        },
        "methods": method_summary,
        "pairwise": pairwise,
        "complete": all(summary["complete"] for summary in method_summary.values()),
    }
    write_json(args.output_dir / "comparison.json", report)
    csv_path = args.output_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "task", "expected_n", "valid_n", "pass_rate", "complete"),
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("generate", "judge", "summary", "validate", "compare", "run-all"),
    )
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--memory-system", choices=tuple(SYSTEMS), required=True)
    parser.add_argument(
        "--ablation",
        choices=("full", *ABLATION_VARIANTS),
        default="full",
        help=(
            "Experiment condition. stage1-baseline runs only Stage 1 then the "
            "baseline answer generator; stage1-stage2-baseline retains Stages 1 "
            "and 2 then uses the baseline answer generator."
        ),
    )
    parser.add_argument("--limit", type=int, help="Process only the first N selected retrieval rows.")
    parser.add_argument("--per-task-limit", type=int, help="Select N rows per benchmark task.")
    parser.add_argument("--seed", type=int, help="Fixed seed for stratified random selection within each task.")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Require and compare only existing GPT Baseline results; do not require M6.",
    )
    parser.add_argument(
        "--skip-sample-id",
        action="append",
        default=[],
        help="Skip a sample explicitly marked as failed; may be repeated.",
    )
    parser.add_argument(
        "--only-sample-id",
        action="append",
        default=[],
        help="Run only explicitly requested selected sample IDs; may be repeated.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Record failed samples and keep processing later samples. "
            "Successful rows are cached and merged in retrieval order after recovery."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("MEMADAPTER_WORKERS", "8")),
        help="Concurrent samples; each sample keeps its three stages sequential.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--retrieval-file", type=Path)
    parser.add_argument("--evidence-file", type=Path, help="Optional JSONL: {sample_id, evidence}.")
    top_k_group = parser.add_mutually_exclusive_group(required=True)
    top_k_group.add_argument(
        "--require-top-10",
        action="store_true",
        help="Require exactly 10 retrieved memories per row.",
    )
    top_k_group.add_argument(
        "--allow-fewer-than-top-10",
        action="store_true",
        help="Explicitly reuse frozen rows containing fewer than 10 memories.",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir or default_output_dir(
        args.model, args.memory_system, args.ablation
    )
    args.retrieval_file = args.retrieval_file or retrieval_path(args.memory_system)
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"generate", "run-all"}:
        generate(args)
    if args.stage in {"judge", "run-all"}:
        judge(args)
    if args.stage in {"summary", "run-all"}:
        summary(args)
    if args.stage in {"validate", "run-all"}:
        validate(args)
    if args.stage in {"compare", "run-all"}:
        compare(args)


if __name__ == "__main__":
    main()
