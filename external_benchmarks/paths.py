"""Filesystem layout for the external-benchmark runs.

Everything written by this harness lives outside the frozen MemSyco-Bench
artifacts (``retrieval/``, ``baseline/``, ``ours/``) so a re-run here can never
overwrite or invalidate the existing results.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent

RUBRICS_ROOT = PACKAGE_ROOT / "rubrics"

MEMTRAP_ROOT = ROOT / "datasets" / "memtrapbench" / "source"
PERSIST_ROOT = ROOT / "datasets" / "persistbench" / "source" / "benchmark_samples" / "persistbench"

#: Frozen retrieval files, mirroring ``retrieval/<system>/retrieved_official_*.jsonl``.
RETRIEVAL_ROOT = ROOT / "retrieval_external"
#: Per-sample memory stores, kept apart from ``retrieval/<system>/process/memory_store``.
STORE_ROOT = ROOT / "external_results" / "_stores"
#: Generation and judge records.
RESULTS_ROOT = Path(os.environ.get("EXTERNAL_RESULTS_ROOT") or ROOT / "external_results").resolve()
#: Cross-cutting tables and provenance.
REPORTS_ROOT = ROOT / "reports" / "external_bench"
LOG_ROOT = RESULTS_ROOT / "logs"

#: Intermediate adapter output (wrapper rows fed to build_retrieval).
ROWS_ROOT = RESULTS_ROOT / "_rows"


def benchmark_root() -> Path:
    return ROOT / "datasets" / "memsyco-bench" / "evaluation"


def ensure_benchmark_on_path() -> Path:
    """Make ``benchmark/`` and ``benchmark/baselines/`` importable.

    The repo's evaluation code imports ``from baselines import ...`` (a package
    living at ``benchmark/baselines``) while its own scripts import
    ``from common import ...`` (a top-level module inside that package), so both
    directories have to be on ``sys.path``. Idempotent and safe to call often.
    """
    for path in (benchmark_root(), benchmark_root() / "baselines"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return benchmark_root()


def load_memadapter():
    """Import ``MemAdapter/run_memadapter.py``, which is a script directory, not a package.

    Its helpers are reused rather than reimplemented on purpose: row normalization
    decides the memory text both arms see, and ``retry_call`` decides how both arms
    retry. Two copies of either are two chances to drift apart, and the drift would
    show up as an arm difference that is not the intervention.
    """
    directory = ROOT / "methods" / "memadapter"
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    import run_memadapter  # type: ignore

    return run_memadapter


def retrieval_dir(dataset: str, subset: str, system: str) -> Path:
    if os.environ.get("NORMALIZED_RESULTS_LAYOUT") == "1" and dataset == "persistbench":
        return ROOT / "datasets" / "persistbench" / "retrieval" / subset / system
    return RETRIEVAL_ROOT / dataset / subset / system


def store_dir(dataset: str, subset: str, system: str) -> Path:
    return STORE_ROOT / dataset / subset / system


def arm_dir(dataset: str, subset: str, system: str, arm: str, run_index: int = 0) -> Path:
    """Output directory for one (dataset, subset, system, arm, repeat).

    The run index is part of the path because PersistBench draws K independent
    generations for the safety classes; each repeat needs its own ``stage_cache``.
    """
    if os.environ.get("NORMALIZED_RESULTS_LAYOUT") == "1" and dataset == "persistbench":
        system_dir = {
            "AMEM": "a-mem", "Mem0": "mem0", "naiveRAG": "naive-rag",
            "MemoryBank": "memory-bank", "LightMem": "light-mem",
        }[system]
        arm_dir_name = {"anti_sycophancy": "anti-sycophancy", "self_recheck": "self-recheck", "dynamic_partition": "dynamic-partition"}.get(arm, arm)
        base = RESULTS_ROOT / system_dir / arm_dir_name / subset
    else:
        base = RESULTS_ROOT / dataset / subset / system / arm
    return base if run_index == 0 else base / f"run{run_index}"
