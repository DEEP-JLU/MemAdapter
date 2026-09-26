"""Per-dataset judge implementations.

Both arms of a cell are judged by the same function here, so the comparison cannot be
confounded by two judging paths. That is also why this is a package of its own rather
than a branch inside ``MemAdapter/run_memadapter.py``: that runner's judge stage
refuses external tasks outright (``refuse_native_stage``), because its summary keys
every number off the native five tasks' pass metrics.

The two datasets share almost nothing -- MemTrapBench scores 0-5 dimensions, one
generation per sample; PersistBench scores a single 1-5 (or 1-3) judgement, three
generations for the two safety classes -- so only the parts that keep the arms
comparable live in :mod:`external_benchmarks.judges.common`.
"""

from __future__ import annotations

__all__ = ["common", "judge_memtrap", "judge_persist"]
