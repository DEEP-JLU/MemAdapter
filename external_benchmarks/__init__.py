"""External-benchmark harness: MemTrapBench and PersistBench.

Runs the repo's two method arms (``baseline`` and ``MemAdapter``) against two
external cognitive-trap datasets with a frozen, shared retrieval pass. The two
arms are guaranteed to share model, temperature, retrieval file, judge and split;
they differ only in the final-answer prompt path, which is the intervention.
"""

from __future__ import annotations

__all__ = ["paths", "task_registry"]
