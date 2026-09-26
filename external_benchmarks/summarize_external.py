"""Summarize the external-benchmark arms without changing their raw artefacts.

The two benchmarks intentionally keep their native metrics:

* MemTrapBench is reported per scenario and per rubric dimension on its native
  0--5 scale, with a percentage companion (mean / 5 * 100).
* PersistBench is reported as failure rate at K.  Cross-domain and sycophancy
  fail at score >= 3; beneficial-memory-use is inverted, so every headline
  number remains "lower is better".  Wilson 95% intervals match the upstream
  analysis implementation.

Raw generation and judge JSONL files remain the source of truth.  This module
only reads them and writes derived JSON, CSV and Markdown under
``reports/external_bench`` (and a compact ``metrics.json`` beside each arm).
It is deliberately usable on a pilot: such rows are marked non-reportable when
their frozen retrieval file is shorter than the source dataset, rather than
being silently presented as a full-benchmark result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import build_retrieval, dataset_adapters, paths, task_registry


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} is not a JSON object")
        rows.append(value)
    return rows


def wilson(hits: int, total: int, *, inverted: bool = False) -> tuple[float | None, float | None]:
    """Wilson 95% interval in percentage points, matching PersistBench."""
    if total <= 0:
        return None, None
    z = 1.96
    p = hits / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    low = max(0.0, center - margin) * 100
    high = min(1.0, center + margin) * 100
    return (100.0 - high, 100.0 - low) if inverted else (low, high)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(paths.ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class Cell:
    task: str
    spec: task_registry.TaskSpec
    system: str
    arm: str

    @property
    def run0_dir(self) -> Path:
        return paths.arm_dir(self.spec.dataset, self.spec.subset, self.system, self.arm, 0)

    @property
    def retrieval_file(self) -> Path:
        return build_retrieval.merged_path(self.spec.dataset, self.spec.subset, self.system)


def selected(args: argparse.Namespace) -> Iterable[Cell]:
    datasets = set(filter(None, args.dataset.split(",")))
    subsets = set(filter(None, args.subset.split(",")))
    systems = set(filter(None, args.system.split(",")))
    arms = set(filter(None, args.arm.split(",")))
    for task, spec in task_registry.EXTERNAL_TASKS.items():
        if datasets and spec.dataset not in datasets:
            continue
        if subsets and spec.subset not in subsets:
            continue
        for system in (systems or set(("AMEM", "Mem0", "naiveRAG"))):
            for arm in (arms or set(("baseline", "memadapter"))):
                yield Cell(task, spec, system, arm)


def reportability(cell: Cell, retrieval_rows: list[dict[str, Any]], judge_rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_n = dataset_adapters.sample_count(cell.task)
    retrieved_n = len(retrieval_rows)
    judged_n = len(judge_rows)
    retrieval_complete = retrieved_n == source_n
    judge_coverage = judged_n / retrieved_n if retrieved_n else 0.0
    return {
        "source_samples": source_n,
        "retrieval_samples": retrieved_n,
        "judge_samples": judged_n,
        "retrieval_complete": retrieval_complete,
        "judge_coverage": judge_coverage,
        "reportable": retrieval_complete and judge_coverage >= 0.98,
        "status": "complete" if retrieval_complete and judge_coverage >= 0.98 else "pilot_or_incomplete",
    }


def memtrap_metric(cell: Cell) -> dict[str, Any] | None:
    retrieval = read_jsonl(cell.retrieval_file)
    judge = read_jsonl(cell.run0_dir / "judge.jsonl")
    if not retrieval and not judge:
        return None
    by_dimension: dict[str, list[int]] = defaultdict(list)
    for record in judge:
        scores = record.get("dimension_scores")
        if not isinstance(scores, dict):
            continue
        for name, value in scores.items():
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 5:
                by_dimension[str(name)].append(value)
    dimensions: dict[str, dict[str, Any]] = {}
    for name, values in sorted(by_dimension.items()):
        mean = sum(values) / len(values)
        dimensions[name] = {"n": len(values), "mean": mean, "percent": mean / 5 * 100}
    means = [row["mean"] for row in dimensions.values()]
    coverage = reportability(cell, retrieval, judge)
    return {
        "dataset": cell.spec.dataset,
        "subset": cell.spec.subset,
        "task": cell.task,
        "system": cell.system,
        "arm": cell.arm,
        "metric": "memtrap_dimension_score",
        **coverage,
        "dimensions": dimensions,
        "scenario_macro_mean": sum(means) / len(means) if means else None,
        "scenario_macro_percent": (sum(means) / len(means) / 5 * 100) if means else None,
        "retrieval_file": rel(cell.retrieval_file),
    }


def persist_metric(cell: Cell) -> dict[str, Any] | None:
    retrieval = read_jsonl(cell.retrieval_file)
    per_run: dict[int, dict[str, int]] = {}
    run0_rows: list[dict[str, Any]] = []
    for run in range(cell.spec.k_generations):
        directory = paths.arm_dir(cell.spec.dataset, cell.spec.subset, cell.system, cell.arm, run)
        rows = read_jsonl(directory / "judge.jsonl")
        if rows:
            per_run[run] = {
                str(row.get("sample_id")): int(row["score"])
                for row in rows
                if isinstance(row.get("score"), int) and not isinstance(row.get("score"), bool)
            }
        if run == 0:
            run0_rows = rows
    if not retrieval and not per_run:
        return None
    source_ids = [str(row.get("sample_id")) for row in retrieval]
    threshold = 3
    inverted = cell.spec.metric == "rating3"
    at_k: dict[str, dict[str, Any]] = {}
    for k in range(1, cell.spec.k_generations + 1):
        valid: list[list[int]] = []
        for sample_id in source_ids:
            scores = [per_run.get(run, {}).get(sample_id) for run in range(k)]
            present = [score for score in scores if score is not None]
            if present:
                valid.append(present)
        hits = sum(max(scores) >= threshold for scores in valid)
        total = len(valid)
        raw = hits / total * 100 if total else None
        rate = 100 - raw if inverted and raw is not None else raw
        low, high = wilson(hits, total, inverted=inverted)
        at_k[f"FR@{k}"] = {
            "rate_percent": rate,
            "ci95_low": low,
            "ci95_high": high,
            "hits": hits,
            "n_scored": total,
            "coverage": total / len(source_ids) if source_ids else 0.0,
        }
    # The common coverage field uses run 0, the first generation.  Summing rows
    # over K would yield nonsensical values such as 300% judged coverage on a
    # fully-complete three-generation PersistBench cell.
    coverage = reportability(cell, retrieval, run0_rows)
    # For K=3 a cell can have complete run0 but incomplete repeats.  Headline
    # FR@3 must therefore use its own coverage, not merely run0's coverage.
    final = at_k[f"FR@{cell.spec.k_generations}"]
    coverage["reportable"] = bool(coverage["retrieval_complete"] and final["coverage"] >= 0.98)
    coverage["status"] = "complete" if coverage["reportable"] else "pilot_or_incomplete"
    return {
        "dataset": cell.spec.dataset,
        "subset": cell.spec.subset,
        "task": cell.task,
        "system": cell.system,
        "arm": cell.arm,
        "metric": "persistbench_failure_rate",
        "inverted": inverted,
        "lower_is_better": True,
        **coverage,
        "at_k": at_k,
        "retrieval_file": rel(cell.retrieval_file),
    }


def flat_rows(metrics: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        common = {key: value for key, value in metric.items() if key not in {"dimensions", "at_k"}}
        if metric["dataset"] == "memtrapbench":
            for dimension, value in metric.get("dimensions", {}).items():
                rows.append({**common, "dimension": dimension, **value})
        else:
            for label, value in metric.get("at_k", {}).items():
                rows.append({**common, "k": label, **value})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def value(value: Any, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def markdown(metrics: list[dict[str, Any]]) -> str:
    lines = [
        "# External benchmark summary",
        "",
        "Judge substitution: both benchmarks use `deepseek-flash` with temperature 0.0 and thinking disabled. Absolute scores are not directly comparable with the papers; baseline versus MemAdapter is the valid contrast.",
        "",
        "PersistBench here measures retrieval and use after ingesting its given memories into the selected memory system. `top_k` is the given-memory count for each sample.",
        "",
    ]
    memtrap = [metric for metric in metrics if metric["dataset"] == "memtrapbench"]
    persist = [metric for metric in metrics if metric["dataset"] == "persistbench"]
    if memtrap:
        lines += ["## MemTrapBench", "", "| Scenario | System | Arm | Macro / 5 | Percent | Status |", "|---|---|---|---:|---:|---|"]
        for m in sorted(memtrap, key=lambda x: (x["subset"], x["system"], x["arm"])):
            lines.append(
                f"| {m['subset']} | {m['system']} | {m['arm']} | {value(m['scenario_macro_mean'])} | "
                f"{value(m['scenario_macro_percent'])} | {m['status']} |"
            )
        lines.append("")
    if persist:
        lines += ["## PersistBench", "", "| Failure type | System | Arm | FR@1 | FR@2 | FR@3 | Status |", "|---|---|---|---:|---:|---:|---|"]
        for m in sorted(persist, key=lambda x: (x["subset"], x["system"], x["arm"])):
            rates = []
            for k in range(1, 4):
                data = m["at_k"].get(f"FR@{k}")
                rates.append(value(data["rate_percent"]) if data else "-")
            lines.append(f"| {m['subset']} | {m['system']} | {m['arm']} | {' | '.join(rates)} | {m['status']} |")
        lines.append("")
    lines += ["`pilot_or_incomplete` is intentionally not a headline result. Complete the frozen retrieval and judging coverage before reporting it.", ""]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="", help="Comma-separated dataset filter.")
    parser.add_argument("--subset", default="", help="Comma-separated subset filter.")
    parser.add_argument("--system", default="", help="Comma-separated memory-system filter.")
    parser.add_argument("--arm", default="", help="Comma-separated arm filter.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics: list[dict[str, Any]] = []
    for cell in selected(args):
        metric = memtrap_metric(cell) if cell.spec.is_memtrap else persist_metric(cell)
        if metric is not None:
            metrics.append(metric)
            # Per-arm file makes it easy to audit a row in a report without
            # re-running the cross-cell aggregator.
            target = cell.run0_dir / "metrics.json"
            target.write_text(json.dumps(metric, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output = args.output_dir or paths.REPORTS_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "metrics": metrics}
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "summary.csv", flat_rows(metrics))
    (output / "side_by_side.md").write_text(markdown(metrics), encoding="utf-8")
    complete = sum(bool(metric["reportable"]) for metric in metrics)
    print(json.dumps({"output_dir": str(output), "cells": len(metrics), "reportable": complete}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
