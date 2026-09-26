"""Writing an explicit message list into a memory layer.

The stock ingest loop (``_optimized_memory._build_toolkit_entry``) takes a
dialogue *string* and re-parses it into messages. Both external adapters already
know the exact message list -- MemTrapBench from ``context_history``, PersistBench
from the given memory set -- so they hand it over directly and this module writes
it. See ``dataset_adapters`` for why the string round-trip is not safe here.

Two modes, differing only in how a message becomes stored text:

``messages``
    Each layer's own ``add_message``. This is byte-for-byte the stock path and is
    what MemTrapBench uses: the source genuinely is a dialogue, so it should be
    ingested as one. A-MEM prefixes ``Speaker user says: `` and MemZero runs its
    LLM extraction, exactly as in the native MemSyco-Bench runs.

``raw``
    The memory string is written straight to the underlying store -- no speaker
    prefix, and ``infer=False`` so MemZero does not paraphrase it. PersistBench
    uses this because its memories are *given* strings that define the construct:
    a cross-domain leakage probe is only meaningful if the memory the system holds
    is the memory the benchmark specified.

On A-MEM and verbatim fidelity. A-MEM's ``add_note`` runs ``analyze_content`` and
``process_memory`` on every note, so it was expected to paraphrase the text it is
handed. Measured on PersistBench sycophancy, it does not: the note's ``content``
came back byte-identical for all 9 of 9 given memories, with the LLM output
landing in the derived fields (keywords, tags, context) instead. So raw mode is
verbatim for all three systems -- 9/9 on ``raw_content`` for A-MEM, MemZero and
NaiveRAG alike -- and the cross-domain construct holds for A-MEM too. What does
differ is ``used_content``, the layer's formatted view for the prompt
(``Content: ...`` / ``Memory: ...``), which is presentation, not storage, and is
identical across arms because both arms read the same frozen retrieval file.

This is measured rather than assumed because the opposite was the reasonable
expectation: see ``verify_raw_fidelity``, which the pilot gate and
``validate_external.py`` both run.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .run_config import INGEST_MODES, LLM_INGEST_SYSTEMS

#: Signature the memory layer's ingest hook is called with.
IngestFn = Callable[[Any, str, Sequence[dict[str, str]], Sequence[str]], dict[str, Any]]


def _timestamp_helper():
    """Import ``timestamp_for_turn`` from the baselines package.

    Imported lazily because it pulls in the vendored memory toolkit's sys.path
    setup, which must not happen at module import time.
    """
    from . import paths

    paths.ensure_benchmark_on_path()
    from common import timestamp_for_turn  # type: ignore

    return timestamp_for_turn


@dataclass(frozen=True)
class IngestPlan:
    """How one sample's messages should be written, plus the digest material."""

    mode: str
    messages: tuple[dict[str, str], ...]
    top_k: int | None = None
    sample_id: str = ""
    # An optional explicit identity lets a targeted repair use the same write
    # mechanics as ``raw`` while remaining in a separate cache namespace.
    # Empty-primary retrieval fallback must never reuse a normal raw store.
    label: str = ""

    def __post_init__(self) -> None:
        if self.mode not in INGEST_MODES:
            raise ValueError(f"Unknown ingest mode {self.mode!r}; known: {INGEST_MODES}")

    def identity(self) -> str:
        """Tag mixed into the store digest.

        Without this, a ``raw`` store and a ``messages`` store built from the same
        sample would share a digest and the second build would silently adopt the
        first one's directory -- wrong retrieved memories, no error.
        """
        return self.label or f"ingest={self.mode}"

    @property
    def turn_indices(self) -> tuple[int, ...]:
        return tuple(range(len(self.messages)))

    @property
    def hook(self) -> Callable[[Any, str], dict[str, Any]] | None:
        """The ``(layer, method) -> report`` callable ``_optimized_memory`` wants.

        Returns None for ``messages`` mode: that mode *is* the stock ingest loop,
        so letting it run means MemTrapBench rides the same code path as the
        native MemSyco-Bench tasks instead of a re-implementation.
        """
        if self.mode == "messages":
            return None
        return make_ingest_fn(self)


def _config_user_id(layer: Any) -> str:
    config = getattr(layer, "config", None)
    user_id = getattr(config, "user_id", "") if config is not None else ""
    if not user_id:
        raise ValueError(
            "Memory layer has no config.user_id; cannot scope an ingest to a user"
        )
    return str(user_id)


# --- mode: messages (stock path) ------------------------------------------


def ingest_messages(
    layer: Any,
    method: str,
    messages: Sequence[dict[str, str]],
    timestamps: Sequence[str],
) -> dict[str, Any]:
    """Insert chat turns through each layer's own ``add_message``.

    Deliberately a verbatim re-statement of the stock loop rather than a call
    into it, so the two remain independently readable when one changes.
    """
    for index, message in enumerate(messages):
        add_kwargs: dict[str, Any] = {"timestamp": timestamps[index]}
        if method == "NaiveRAG":
            add_kwargs["turn_index"] = index
        layer.add_message(dict(message), **add_kwargs)
    return {"inserted": len(messages), "mode": "messages", "method": method}


# --- mode: raw (direct store write) ---------------------------------------


def ingest_memories_raw(
    layer: Any,
    method: str,
    messages: Sequence[dict[str, str]],
    timestamps: Sequence[str],
) -> dict[str, Any]:
    """Write each message's content straight into the store, verbatim.

    Dispatches on the layer's structure rather than on a capability flag, because
    the three layers expose genuinely different low-level entry points:

    *   ``A-MEM``    -- ``AgenticMemorySystem.add_note`` (the prefix is added by
        the layer's own ``add_message``, so calling past it removes the prefix).
    *   ``MemZero``  -- ``mem0.Memory.add`` with ``infer=False``, which stores the
        text as-is instead of extracting facts from it.
    *   ``NaiveRAG`` -- ``mem0.Memory.add`` likewise; the layer's ``add_message``
        would otherwise prepend a ``User: `` label via ``_format_turn_chunk``.
    """
    user_id = _config_user_id(layer)
    inner = getattr(layer, "memory_layer", None)
    if inner is None:
        raise ValueError(f"Layer for method {method!r} has no .memory_layer to write to")

    for index, message in enumerate(messages):
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        timestamp = timestamps[index]

        if method == "A-MEM":
            inner.add_note(content, time=timestamp)
        elif method == "MemZero":
            inner.add(
                messages=content,
                user_id=user_id,
                metadata={"timestamp": timestamp},
                infer=False,
            )
        elif method == "NaiveRAG":
            inner.add(
                messages=[{"role": "user", "content": content}],
                user_id=user_id,
                infer=False,
                metadata={
                    "raw_role": "user",
                    "chunk_type": "given_memory",
                    "turn_index": index,
                    "timestamp": timestamp,
                },
            )
        else:
            raise ValueError(
                f"raw ingest has no implementation for method {method!r}; "
                f"known: A-MEM, MemZero, NaiveRAG"
            )

    return {
        "inserted": sum(1 for m in messages if str(m.get("content") or "").strip()),
        "mode": "raw",
        "method": method,
        "llm_ingest": method in LLM_INGEST_SYSTEMS,
    }


_INGEST_FNS: dict[str, IngestFn] = {
    "messages": ingest_messages,
    "raw": ingest_memories_raw,
}


def ingest_fn(mode: str) -> IngestFn:
    try:
        return _INGEST_FNS[mode]
    except KeyError:
        raise KeyError(f"Unknown ingest mode {mode!r}; known: {sorted(_INGEST_FNS)}") from None


def run_plan(layer: Any, method: str, plan: IngestPlan) -> dict[str, Any]:
    """Apply a plan to a freshly built layer and report what happened."""
    timestamps = [_timestamp_helper()(i) for i in plan.turn_indices]
    report = ingest_fn(plan.mode)(layer, method, plan.messages, timestamps)
    report.update({"sample_id": plan.sample_id, "planned": len(plan.messages)})
    return report


def make_ingest_fn(plan: IngestPlan) -> Callable[[Any, str], dict[str, Any]]:
    """Bind a plan into the ``(layer, method) -> report`` hook.

    ``_optimized_memory`` takes this hook instead of a plan so that it never has
    to import anything from this package, and never has to know how timestamps
    are derived -- both stay here, on the experiment side.
    """
    timestamps = [_timestamp_helper()(i) for i in plan.turn_indices]

    def _fn(layer: Any, method: str) -> dict[str, Any]:
        report = ingest_fn(plan.mode)(layer, method, plan.messages, timestamps)
        report.update({"sample_id": plan.sample_id, "planned": len(plan.messages)})
        return report

    return _fn


# --- read-back verification -----------------------------------------------


def verify_raw_fidelity(
    layer: Any,
    *,
    memories: Sequence[str],
    query: str,
    top_k: int,
) -> dict[str, Any]:
    """Check that ``raw`` ingest really stored the given strings.

    Runs the layer's own retrieval and compares normalized content. Used by the
    pilot gate and by ``validate_external.py``: the fidelity claim is the whole
    reason PersistBench is run in raw mode, so it is measured rather than assumed.

    Retrieval is a nearest-neighbour search, not a lookup, so a non-match is not
    automatically an error -- the returned counts are reported for inspection.
    """
    from .text_norm import normalize_for_match

    retrieved = layer.retrieve(query, k=top_k)
    wanted = {normalize_for_match(m) for m in memories}
    got = [normalize_for_match(str(item.get("content", ""))) for item in retrieved]
    matched = sum(1 for text in got if text in wanted)
    return {
        "given": len(memories),
        "retrieved": len(got),
        "verbatim_matches": matched,
        "unmatched_retrieved": [t for t in got if t and t not in wanted][:5],
    }
