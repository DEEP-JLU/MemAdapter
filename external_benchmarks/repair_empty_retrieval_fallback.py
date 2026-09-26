"""Repair only frozen retrieval rows whose primary retrieval is empty.

The primary store and every non-empty retrieval line are retained unchanged.  For
an empty row, the source dialogue is inserted as raw vector entries into a new
store namespace and retrieved again.  This is a targeted fallback for the
with-memory protocol, not a global replacement of the memory-system pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from . import dataset_adapters as adapters
from . import paths, run_config, task_registry
from .build_retrieval import _load_cached_builder, build_eval_config, retrieve_sample
from .memory_ingest import IngestPlan

FALLBACK_IDENTITY = "ingest=raw_dialogue_fallback"
MANIFEST_NAME = "EMPTY_RETRIEVAL_FALLBACK_MANIFEST.json"


def _configure_cpu_threads() -> int | None:
    """Bound per-process Torch parallelism for multi-process fallback runs.

    A raw dialogue fallback encodes tens of short turns per sample.  Launching
    several processes with Torch's unrestricted default thread pool turns a
    CPU-only server into hundreds of competing worker threads and is slower
    than a smaller, bounded pool.  The setting is optional so ordinary one-off
    runs preserve their existing behaviour.
    """
    value = os.environ.get("FALLBACK_TORCH_THREADS", "").strip()
    if not value:
        return None
    threads = int(value)
    if threads < 1:
        raise ValueError("FALLBACK_TORCH_THREADS must be positive")
    import torch
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # It may already have been set by an embedding import in an unusual
        # launcher.  The intra-op limit remains useful in that case.
        pass
    return threads


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _retrieval_path(task: str, system: str) -> Path:
    spec = task_registry.spec(task)
    folder = paths.retrieval_dir(spec.dataset, spec.subset, system)
    candidates = sorted(folder.glob("retrieved_top*.jsonl"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"{task}/{system}: expected one merged retrieval file in {folder}")
    return candidates[0]


def _fallback_plan(sample: adapters.AdaptedSample) -> IngestPlan:
    # Raw writing stores each original dialogue turn as an independently
    # retrievable vector, bypassing Mem0's LLM fact extraction only for a primary
    # retrieval that produced no vectors at all.
    return IngestPlan(
        mode="raw",
        messages=sample.ingest_messages,
        top_k=sample.top_k,
        sample_id=sample.sample_id,
        label=FALLBACK_IDENTITY,
    )


def _read_raw(path: Path) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not row.get("sample_id"):
                raise ValueError(f"{path}:{line_no}: missing sample_id")
            rows.append((raw, row))
    return rows


def repair_one(task: str, system: str, *, dry_run: bool = False) -> dict[str, Any]:
    source = _retrieval_path(task, system)
    before_sha = _sha(source)
    original = _read_raw(source)
    empty_indices = [
        int(row["row_index"]) for _, row in original
        if not list(row.get("retrieved_memories") or [])
    ]
    if not empty_indices:
        return {
            "task": task, "system": system, "source": str(source),
            "before_sha256": before_sha, "empty_primary": 0,
            "repaired": 0, "still_empty": 0, "changed": False,
        }

    if dry_run:
        return {
            "task": task, "system": system, "source": str(source),
            "before_sha256": before_sha, "empty_primary": len(empty_indices),
            "row_indices": empty_indices, "changed": False,
        }

    samples = {
        sample.row_index: sample
        for sample in adapters.load_samples(task, indices=empty_indices)
    }
    if set(samples) != set(empty_indices):
        raise RuntimeError(f"{task}/{system}: source rows cannot be reconstructed exactly")

    spec = task_registry.spec(task)
    config = build_eval_config(
        system, dataset=spec.dataset, subset=spec.subset, top_k=10
    )
    build_context = _load_cached_builder()
    replacements: dict[str, dict[str, Any]] = {}
    still_empty: list[str] = []
    started = time.time()

    for ordinal, (_, old) in enumerate(original, start=1):
        memories = list(old.get("retrieved_memories") or [])
        if memories:
            continue
        row_index = int(old["row_index"])
        sample = samples[row_index]
        if sample.sample_id != str(old["sample_id"]):
            raise RuntimeError(
                f"{task}/{system}: sample id drift at row {row_index}: "
                f"{sample.sample_id!r} != {old['sample_id']!r}"
            )
        replacement = retrieve_sample(
            sample, system, config, build_context, ingest_plan=_fallback_plan(sample)
        )
        cfg = dict(replacement["memory_config"])
        cfg.update({
            "ingest_mode": "raw_dialogue_fallback",
            "ingest_identity": FALLBACK_IDENTITY,
            "fallback_trigger": "empty_primary_retrieval",
            "primary_retrieved_count": 0,
            "fallback_applied_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        replacement["memory_config"] = cfg
        replacement["fallback_from_empty_primary"] = True
        if not replacement["retrieved_memories"]:
            still_empty.append(sample.sample_id)
        replacements[sample.sample_id] = replacement
        if ordinal % 20 == 0 or ordinal == len(original):
            print(
                f"[fallback] {task}/{system} {len(replacements)}/{len(empty_indices)} "
                f"still_empty={len(still_empty)}",
                flush=True,
            )

    backup = source.with_name(source.stem + ".pre_empty_fallback.jsonl")
    if not backup.exists():
        shutil.copy2(source, backup)
    elif _sha(backup) != before_sha:
        raise RuntimeError(f"{task}/{system}: backup exists but differs from source")

    temp = source.with_suffix(source.suffix + ".fallback.tmp")
    preserved = repaired = 0
    with temp.open("w", encoding="utf-8", newline="") as out:
        for raw, old in original:
            sid = str(old["sample_id"])
            if sid not in replacements:
                out.write(raw)
                preserved += 1
                continue
            out.write(json.dumps(replacements[sid], ensure_ascii=False) + "\n")
            repaired += 1
    # The only permitted modifications are rows which were empty in the source.
    rebuilt = _read_raw(temp)
    if len(rebuilt) != len(original):
        raise RuntimeError(f"{task}/{system}: row count changed")
    for (old_raw, old), (new_raw, new) in zip(original, rebuilt):
        if list(old.get("retrieved_memories") or []):
            if old_raw != new_raw:
                raise RuntimeError(f"{task}/{system}: nonempty row changed: {old['sample_id']}")
        elif old["sample_id"] != new["sample_id"]:
            raise RuntimeError(f"{task}/{system}: row order or id changed")
    temp.replace(source)

    return {
        "task": task, "system": system, "source": str(source),
        "before_sha256": before_sha, "after_sha256": _sha(source),
        "backup": str(backup), "empty_primary": len(empty_indices),
        "repaired": repaired, "still_empty": len(still_empty),
        "still_empty_ids": still_empty,
        "nonempty_rows_preserved_byte_exact": preserved,
        "elapsed_seconds": round(time.time() - started, 3),
        "changed": True,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", choices=tuple(task_registry.EXTERNAL_TASKS))
    parser.add_argument("--system", action="append", choices=tuple(run_config.SYSTEMS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=paths.RETRIEVAL_ROOT / MANIFEST_NAME)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run_config.load_env_file()
    configured_threads = _configure_cpu_threads()
    args = _parse_args(argv)
    tasks = args.task or list(task_registry.EXTERNAL_TASKS)
    systems = args.system or list(run_config.SYSTEMS)
    results: list[dict[str, Any]] = []
    for task in tasks:
        for system in systems:
            result = repair_one(task, system, dry_run=args.dry_run)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    summary = {
        "protocol": "raw dialogue fallback only for empty primary retrieval rows",
        "fallback_identity": FALLBACK_IDENTITY,
        "fallback_torch_threads": configured_threads,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "totals": {
            "empty_primary": sum(int(r.get("empty_primary", 0)) for r in results),
            "repaired": sum(int(r.get("repaired", 0)) for r in results),
            "still_empty": sum(int(r.get("still_empty", 0)) for r in results),
            "nonempty_rows_preserved_byte_exact": sum(
                int(r.get("nonempty_rows_preserved_byte_exact", 0)) for r in results
            ),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["totals"]["still_empty"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
