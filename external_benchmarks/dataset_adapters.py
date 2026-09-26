"""Adapters turning the two external datasets into the repo's wrapper-row schema.

Both adapters emit ``{sample_id, task, source_file, row_index, current_request,
benchmark_row}`` -- the same contract the five native MemSyco-Bench tasks use --
so every downstream reader works untouched. They additionally return the exact
message list to ingest. That second part is not optional:

    ``benchmark/baselines/common.py::parse_dialogue_to_messages`` splits the
    dialogue on newline runs and flushes on every blank line, then re-derives
    roles from ``^(User|Assistant|System):`` prefixes. Measured over all 1,050
    MemTrapBench samples, that round-trip corrupts 502 of them (47.8%): 6,732
    turns contain a blank line inside their content, which the parser reads as a
    turn boundary. Five further turns open a line with a bare role marker.

Feeding the prior text through the parser would therefore silently truncate or
re-split roughly half the corpus. Instead both adapters hand the ingest path an
explicit ``[{"role", "content"}]`` list, and the two datasets differ only in how
that list is written into the memory store:

*   MemTrapBench -- ``messages`` mode. The source really is a dialogue, so it
    goes through each layer's own ``add_message``, exactly as the native tasks do.
*   PersistBench -- ``raw`` mode. The memories are given strings, not dialogue;
    they are written straight into the store without a speaker prefix and without
    MemZero's LLM re-extraction, so the text the store holds is the text the
    benchmark specified.

The MemTrapBench record keeps the entire original sample in ``benchmark_row``
(including ``gold_standard``, ``poisoned_fact`` and ``objective_truth``) because
the judge's per-subset rubric reads those fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from . import paths, task_registry

MESSAGE_INGEST = "messages"
RAW_INGEST = "raw"

#: Fields each MemTrapBench sub-dataset is allowed to be missing from the judge's
#: point of view; recorded so the judge can assert rather than KeyError.
_MEMTRAP_JUDGE_FIELDS = {
    "poison": ("poisoned_fact", "objective_truth"),
    "number_game": ("gold_standard",),
}


@dataclass(frozen=True)
class AdaptedSample:
    """One external sample, ready for retrieval and generation."""

    sample_id: str
    task: str
    dataset: str
    subset: str
    source_file: str
    row_index: int
    current_request: str
    benchmark_row: dict[str, Any]
    ingest_mode: str
    ingest_messages: tuple[dict[str, str], ...]
    #: Per-sample retrieval depth. ``None`` means "use the run config's top_k".
    top_k: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> task_registry.TaskSpec:
        return task_registry.spec(self.task)

    @property
    def prior_text(self) -> str:
        """The prior as a string, for logging and legacy display paths.

        This is deliberately *not* what gets ingested -- see the module docstring.
        It exists so an audit can see the dialogue in one blob.
        """
        return "\n\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in self.ingest_messages
        )

    @property
    def messages_count(self) -> int:
        return len(self.ingest_messages)

    def wrapper_row(self) -> dict[str, Any]:
        """The dataset-agnostic row written to ``_rows/<task>.jsonl``."""
        return {
            "sample_id": self.sample_id,
            "task": self.task,
            "source_file": self.source_file,
            "row_index": self.row_index,
            "current_request": self.current_request,
            "benchmark_row": self.benchmark_row,
        }

    def prior_sha256(self) -> str:
        from .run_config import sha256_text

        return sha256_text(
            json.dumps(
                [list(m.items()) for m in self.ingest_messages],
                ensure_ascii=False,
                sort_keys=True,
            )
        )


# --- MemTrapBench ----------------------------------------------------------


def _memtrap_messages(raw: dict[str, Any]) -> list[dict[str, str]]:
    """One message per non-empty context turn, in order.

    ``context_history`` turns carry ``{content, role, turn}``. The turn index is
    preserved in ``meta`` rather than injected into the text, because the memory
    layers already record their own ordering metadata.
    """
    messages: list[dict[str, str]] = []
    for turn in raw.get("context_history") or []:
        if not isinstance(turn, dict):
            continue
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        role = str(turn.get("role") or "user").strip().lower()
        if role not in {"user", "assistant", "system"}:
            role = "user"
        messages.append({"role": role, "content": content})
    return messages


def adapt_memtrap(
    task: str,
    data_path: Path,
    row_index: int,
    raw: dict[str, Any],
) -> AdaptedSample:
    spec = task_registry.spec(task)
    messages = _memtrap_messages(raw)
    if not messages:
        raise ValueError(f"{task} row {row_index} has no usable context_history turns")

    final_trigger = str(raw.get("final_trigger") or "").strip()
    if not final_trigger:
        raise ValueError(f"{task} row {row_index} has an empty final_trigger")

    sample_id = str(raw.get("id") or f"{spec.subset}_{row_index:06d}")

    # The judge rubric for these subsets interpolates fields that only exist here.
    # Exactly one row in the whole corpus is affected (poison row 109 has no
    # ``poisoned_fact``). Upstream tolerates that -- ``build_judge_prompt`` fills
    # the placeholder from ``item.get(field, "")`` -- so the sample is kept and
    # the gap is recorded rather than failing the subset, which would drop 200
    # rows to punish a single missing key.
    missing_judge_fields = [
        name
        for name in _MEMTRAP_JUDGE_FIELDS.get(spec.subset, ())
        if not str(raw.get(name) or "").strip()
    ]

    benchmark_row = dict(raw)
    benchmark_row["sample_id"] = sample_id
    benchmark_row["task"] = task
    # Kept for schema compatibility and audit; ingest does not read it.
    benchmark_row["dialogue_context_turns"] = [
        {
            "speaker": "User" if m["role"] == "user" else "Assistant",
            "content": f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}",
            "is_query": False,
        }
        for m in messages
    ]

    return AdaptedSample(
        sample_id=sample_id,
        task=task,
        dataset=spec.dataset,
        subset=spec.subset,
        source_file=str(data_path),
        row_index=row_index,
        current_request=final_trigger,
        benchmark_row=benchmark_row,
        ingest_mode=MESSAGE_INGEST,
        ingest_messages=tuple(messages),
        top_k=None,
        meta={
            "paper_category": _paper_category(spec.subset),
            # Upstream metadata, audit only -- never the ingest count. It counts
            # user/assistant *pairs* (poison: 25 for 50 messages) and disagrees with
            # ``context_history`` on 61 rows: 58 ``unclear`` rows declare 40 more
            # messages than are present, and 3 ``hurt`` rows hold an odd turn count,
            # i.e. one turn short of a pair. ``len(ingest_messages)`` is the truth
            # here; the corpus has no empty-content and no unknown-role turns, so
            # that length is exactly ``len(context_history)``.
            "upstream_history_turns": raw.get("history_turns"),
            "user_turns": sum(1 for m in messages if m["role"] == "user"),
            "assistant_turns": sum(1 for m in messages if m["role"] == "assistant"),
            "missing_judge_fields": missing_judge_fields,
        },
    )


_PAPER_CATEGORY = {
    "poison": "Safety",
    "number_game": "Cognitive Bias",
    "hurt": "Trauma",
    "hallucination": "Task Boundary",
    "inertia": "Cognitive Bias",
    "unclear": "Task Boundary",
}


def formatted_memories(memories: list[str]) -> str:
    """The ``<memories>`` XML list PersistBench feeds both its generator and its judge.

    Byte-identical to upstream ``prompts.formatted_memories`` -- the same function
    ``build_generation_prompt`` and ``build_judge_prompt`` both call -- so the memories
    reach the model in the dataset's own framing rather than one we invented. The judge
    is given the *complete* set (upstream prefers ``full_memories`` over the retrieved
    subset), which is why this takes strings rather than a retrieval row.
    """
    return "<memories>\n" + "\n".join([f"- {memory}" for memory in memories]) + "\n</memories>"


def _paper_category(subset: str) -> str:
    return _PAPER_CATEGORY[subset]


# --- PersistBench ----------------------------------------------------------


def adapt_persist(
    task: str,
    data_path: Path,
    row_index: int,
    raw: dict[str, Any],
) -> AdaptedSample:
    spec = task_registry.spec(task)
    memories = [str(m).strip() for m in (raw.get("memories") or [])]
    memories = [m for m in memories if m]
    if not memories:
        raise ValueError(f"{task} row {row_index} has no memories")
    if len(set(memories)) != len(memories):
        raise ValueError(f"{task} row {row_index} contains duplicate memories")

    query = str(raw.get("query") or "").strip()
    if not query:
        raise ValueError(f"{task} row {row_index} has an empty query")

    failure_type = str(raw.get("failure_type") or spec.subset).strip()
    if failure_type != spec.subset:
        raise ValueError(
            f"{task} row {row_index} declares failure_type={failure_type!r}, "
            f"expected {spec.subset!r} -- the judge prompt is selected from this"
        )

    # These rows carry no id upstream, so the id is derived from position. The
    # index is the 1-based line number, matching how the row is read from disk.
    sample_id = f"persistbench_{failure_type}_{row_index:04d}"

    benchmark_row = dict(raw)
    benchmark_row["sample_id"] = sample_id
    benchmark_row["task"] = task
    benchmark_row["given_memories"] = list(memories)
    benchmark_row["dialogue_context_turns"] = [
        {"speaker": "User", "content": f"User: {m}", "is_query": False}
        for m in memories
    ]

    return AdaptedSample(
        sample_id=sample_id,
        task=task,
        dataset=spec.dataset,
        subset=spec.subset,
        source_file=str(data_path),
        row_index=row_index,
        current_request=query,
        benchmark_row=benchmark_row,
        ingest_mode=RAW_INGEST,
        ingest_messages=tuple({"role": "user", "content": m} for m in memories),
        # Truncating to a fixed 10 would drop more than half the given memories on
        # 259 of the 500 rows (51.8%), which would deflate the leakage and
        # sycophancy opportunity and turn a fidelity question into a truncation
        # artefact. The depth is therefore the size of the given set.
        top_k=len(memories),
        meta={
            "memory_domain": raw.get("memory_domain"),
            "query_domain": raw.get("query_domain"),
            "given_memory_count": len(memories),
        },
    )


# --- loading ---------------------------------------------------------------


def data_path(task: str) -> Path:
    spec = task_registry.spec(task)
    root = paths.MEMTRAP_ROOT if spec.is_memtrap else paths.PERSIST_ROOT
    return root / spec.data_file


def load_samples(
    task: str,
    *,
    limit: int | None = None,
    indices: list[int] | None = None,
) -> list[AdaptedSample]:
    """Read one external task.

    ``indices`` selects 1-based line numbers (used by the pilot and by shard
    assignment); ``limit`` takes the first N rows. MemTrapBench files are JSON
    arrays, PersistBench files are JSONL.
    """
    spec = task_registry.spec(task)
    path = data_path(task)
    if not path.is_file():
        raise FileNotFoundError(f"External dataset file missing: {path}")

    if spec.is_memtrap:
        raw_rows = json.loads(path.read_text(encoding="utf-8"))
        enumerated = list(enumerate(raw_rows, start=1))
    else:
        enumerated = []
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if line.strip():
                    enumerated.append((number, json.loads(line)))

    if indices is not None:
        wanted = set(indices)
        enumerated = [(n, r) for n, r in enumerated if n in wanted]
    if limit is not None:
        enumerated = enumerated[:limit]

    adapt = adapt_memtrap if spec.is_memtrap else adapt_persist
    return [adapt(task, path, number, raw) for number, raw in enumerated]


def sample_count(task: str) -> int:
    """Total rows in the task's file, without adapting any of them.

    ``default_matrix`` calls this to size the dry-run call budget, so it must not
    pay for parsing every row into messages.
    """
    spec = task_registry.spec(task)
    path = data_path(task)
    if not path.is_file():
        return 0
    if spec.is_memtrap:
        return len(json.loads(path.read_text(encoding="utf-8")))
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_tasks(dataset: str | None = None) -> Iterator[str]:
    for task, spec in task_registry.EXTERNAL_TASKS.items():
        if dataset is None or spec.dataset == dataset:
            yield task


def parser_roundtrip_loss(task: str, *, limit: int | None = None) -> dict[str, Any]:
    """Audit helper: how badly the legacy text parser mangles this task's priors.

    Documented rather than worked around -- the ingest path bypasses the parser,
    but the number belongs in the reproducibility notes.
    """
    import sys

    baselines = str(paths.ROOT / "benchmark" / "baselines")
    if baselines not in sys.path:
        sys.path.insert(0, baselines)
    from common import parse_dialogue_to_messages  # type: ignore

    samples = load_samples(task, limit=limit)
    mangled = 0
    for sample in samples:
        reparsed = parse_dialogue_to_messages(sample.prior_text)
        expected = [dict(m) for m in sample.ingest_messages]
        if reparsed != expected:
            mangled += 1
    return {
        "task": task,
        "samples": len(samples),
        "mangled": mangled,
        "rate": (mangled / len(samples)) if samples else 0.0,
    }
