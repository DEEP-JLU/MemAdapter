"""Validate the external-benchmark artefacts before any number from them is reported.

Five groups, each aimed at a failure that would not be visible in the numbers:

*   ``retrieval`` -- the frozen retrieval file's own integrity: row schema, unique and
    ordered ids, ``retrieved_count`` matching the list it summarises, and the depth
    bound (``top_k`` for MemTrapBench, the sample's own given-memory count for
    PersistBench, where ``top_k`` is ``None`` by design).
*   ``store`` -- a per-sample memory store was not silently rebuilt with different
    content. The digest is *recomputed* from the adapted sample and compared with the
    completion marker, the marker's message count and the prior hash; an A-MEM store is
    additionally required to still have its Chroma directory and pickled index, because
    a store whose digest matched but whose index was lost would keep retrieving -- just
    different memories.
*   ``alignment`` -- retrieval, generation and judging describe the same samples in the
    same order, and each arm names the same frozen retrieval file *by sha256*, not by
    path: a file that changed after an arm read it produces perfectly well-formed
    records of the wrong experiment.
*   ``arms`` -- the two arms are the same experiment. Shared config deep-equal, then
    per-record generation settings, then per-record memory text, then the judge's own
    protocol fields; and finally that the *only* difference is the final-answer prompt,
    which must be the dataset's official prompt on one side and stage three of the
    method on the other. Both arms' prompts are hashed from disk, so the check is on the
    bytes sent rather than on a name.
*   ``judge`` -- every recorded judgement is compliant with its own rubric family and is
    *recomputable*: the score in the record is re-derived from the judge's parsed reply
    using the same scoring function the judge used, so a hand-edited or misparsed score
    cannot survive into a table.

Read-only. A ``FAIL`` means do not report the cell; a ``WARN`` means report it with the
caveat named in the message. Both are printed; ``OK`` only under ``--verbose``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import build_retrieval, dataset_adapters, paths, run_config, task_registry
from .judges import common as judge_common
from .judges import judge_persist
from .text_norm import read_prompt_file

OK, WARN, FAIL = "ok", "warn", "fail"
GROUPS = ("retrieval", "store", "alignment", "arms", "judge")

#: Where the memory text lives, in priority order. The MemAdapter arm slims each
#: retrieved memory to ``{memory_id, memory_text}`` while the baseline keeps the frozen
#: five-key canonical object, so the arms can only be compared on the text itself.
MEMORY_TEXT_KEYS = ("memory_text", "content", "text", "raw_content")

#: The five keys ``build_retrieval.canonical_memories`` writes.
CANONICAL_KEYS = frozenset(
    {"memory_id", "content", "raw_content", "retrieval_rank", "metadata"}
)

#: The intervention, as paths relative to the repo root. Mirrored from
#: ``run_config.config_for_external_arm``; a copy that drifts shows up here as a FAIL
#: rather than as two arms that quietly intervene differently.
EXPECTED_ANSWER_PROMPT: dict[str, dict[str, str]] = {
    "baseline": {
        "memtrapbench": "external_benchmarks/rubrics/memtrap/prompt_user_mem.txt",
        "persistbench": "external_benchmarks/rubrics/persist/generator.txt",
    },
    "memadapter": {
        "memtrapbench": "MemAdapter/prompts/03_memory_use_guided_generation.txt",
        "persistbench": "MemAdapter/prompts/03_memory_use_guided_generation.txt",
    },
}

#: Ingest mode each dataset must have used. The mode is part of the store digest, so a
#: mismatch here means the store in ``save_dir`` was not built the way the row claims.
EXPECTED_INGEST = {"memtrapbench": "messages", "persistbench": "raw"}


# --- reporting -------------------------------------------------------------


@dataclass
class Report:
    """Findings, in check order, with per-check detail capped at print time.

    Per-row checks repeat once per sample; a broken cell can produce thousands of
    identical lines, which hides the finding instead of reporting it. The cap keeps the
    first few and says how many more there were.
    """

    findings: list[tuple[str, str, str, str]] = field(default_factory=list)

    def add(self, group: str, level: str, check: str, detail: str = "") -> None:
        self.findings.append((group, level, check, detail))

    def ok(self, group: str, check: str, detail: str = "") -> None:
        self.add(group, OK, check, detail)

    def warn(self, group: str, check: str, detail: str = "") -> None:
        self.add(group, WARN, check, detail)

    def fail(self, group: str, check: str, detail: str = "") -> None:
        self.add(group, FAIL, check, detail)

    def counts(self, group: str) -> tuple[int, int, int]:
        rows = [f[1] for f in self.findings if f[0] == group]
        return rows.count(OK), rows.count(WARN), rows.count(FAIL)

    @property
    def failed(self) -> bool:
        return any(f[1] == FAIL for f in self.findings)

    def render(self, limit: int, verbose: bool) -> str:
        levels = (FAIL, WARN, OK) if verbose else (FAIL, WARN)
        lines: list[str] = []
        for group in GROUPS:
            ok, warn, fail = self.counts(group)
            if not (ok or warn or fail):
                continue
            lines.append(f"[{group}] {ok} ok, {warn} warn, {fail} fail")
            for level in levels:
                shown: dict[str, int] = {}
                for _group, found, check, detail in self.findings:
                    if _group != group or found != level:
                        continue
                    count = shown.get(check, 0)
                    shown[check] = count + 1
                    if count < limit:
                        lines.append(f"    {level.upper():4} {check}: {detail}")
                    elif count == limit:
                        lines.append(
                            f"    {level.upper():4} {check}: (further occurrences "
                            f"suppressed; see the counts above)"
                        )
        ok, warn, fail = (
            sum(self.counts(g)[i] for g in GROUPS) for i in range(3)
        )
        lines.append(f"TOTAL {ok} ok, {warn} warn, {fail} fail")
        return "\n".join(lines)


# --- small helpers ---------------------------------------------------------


def rel(path: str | Path) -> str:
    """Repo-relative POSIX path where possible.

    Absolute paths here contain the repo's own name, which is not ASCII; printing one
    through a Windows console pipe mangles it beyond recognition and makes a finding
    impossible to act on. Relative paths avoid the encoding question entirely.
    """
    try:
        return Path(path).resolve().relative_to(paths.ROOT).as_posix()
    except (ValueError, OSError):
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl_strict(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read a JSONL file, reporting lines that do not parse.

    Deliberately *not* ``runner.read_jsonl``: the producers use that one, and a validator
    that skips exactly the lines they skip cannot see the gap they leave behind.
    """
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    if not path.is_file():
        return rows, [f"missing file {rel(path)}"]
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"line {number} is not JSON: {exc}")
                continue
            if not isinstance(payload, dict):
                problems.append(f"line {number} is not an object")
                continue
            rows.append(payload)
    return rows, problems


def ids_of(rows: Sequence[dict[str, Any]]) -> list[str]:
    return [str(row.get("sample_id")) for row in rows]


def duplicate_ids(ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    dupes: list[str] = []
    for sample_id in ids:
        if sample_id in seen and sample_id not in dupes:
            dupes.append(sample_id)
        seen.add(sample_id)
    return dupes


def memory_texts(memories: Iterable[Any]) -> list[str]:
    """The memory strings as the prompt would render them."""
    out: list[str] = []
    for memory in memories:
        if not isinstance(memory, dict):
            out.append("")
            continue
        text = ""
        for key in MEMORY_TEXT_KEYS:
            value = memory.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
        out.append(text)
    return out


def resolve_config_prompt(record_config: dict[str, Any]) -> Path | None:
    """The answer-prompt file a stored run config names, if it still exists."""
    for key in ("answer_prompt_file", "final_answer_prompt"):
        value = record_config.get(key)
        if not value:
            continue
        candidate = Path(str(value))
        if not candidate.is_file() and not candidate.is_absolute():
            candidate = paths.ROOT / candidate
        if candidate.is_file():
            return candidate
    return None


# --- cells -----------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    task: str
    dataset: str
    subset: str
    system: str
    spec: task_registry.TaskSpec

    @property
    def label(self) -> str:
        return f"{self.dataset}/{self.subset}/{self.system}"

    @property
    def retrieval_file(self) -> Path:
        return build_retrieval.merged_path(self.dataset, self.subset, self.system)


@dataclass
class ArmFacts:
    arm: str
    run_index: int
    directory: Path
    config: dict[str, Any] | None = None
    outputs: list[dict[str, Any]] = field(default_factory=list)
    judge: list[dict[str, Any]] = field(default_factory=list)

    @property
    def label(self) -> str:
        suffix = "" if self.run_index == 0 else f"/run{self.run_index}"
        return f"{self.arm}{suffix}"

    def shared(self) -> dict[str, Any]:
        """The 27 fields both arms must agree on, read out of the stored config."""
        if self.config is None:
            return {}
        return {name: self.config.get(name) for name in run_config.RunConfig.SHARED_FIELDS}


@dataclass
class CellFacts:
    retrieval_rows: list[dict[str, Any]] = field(default_factory=list)
    retrieval_sha: str = ""
    samples: dict[str, dataset_adapters.AdaptedSample] = field(default_factory=dict)
    arms: dict[str, ArmFacts] = field(default_factory=dict)


def arm_run_indices(cell: Cell) -> list[int]:
    """Repeat indices present on disk, always including 0.

    PersistBench draws three independent generations for its two safety classes, each
    into its own ``runN`` directory; everything else is a single run 0. Checking only 0
    would leave two thirds of the safety classes unvalidated.
    """
    indices = {0}
    for arm in run_config.ARMS:
        base = paths.arm_dir(cell.dataset, cell.subset, cell.system, arm, 0)
        if base.is_dir():
            for child in base.glob("run*"):
                if child.is_dir() and child.name[3:].isdigit():
                    indices.add(int(child.name[3:]))
    return sorted(indices)


def discover(args: argparse.Namespace) -> tuple[list[Cell], list[Cell]]:
    """Split every (task, system) into "has artefacts" and "not started"."""
    active: list[Cell] = []
    pending: list[Cell] = []
    for task in dataset_adapters.iter_tasks(None):
        spec = task_registry.spec(task)
        if args.dataset and spec.dataset not in args.dataset:
            continue
        if args.subset and spec.subset not in args.subset:
            continue
        for system in run_config.SYSTEMS:
            if args.system and system not in args.system:
                continue
            cell = Cell(task, spec.dataset, spec.subset, system, spec)
            present = cell.retrieval_file.is_file() or any(
                paths.arm_dir(cell.dataset, cell.subset, cell.system, arm, 0).is_dir()
                for arm in run_config.ARMS
            )
            (active if present else pending).append(cell)
    return active, pending


def load_arm(cell: Cell, arm: str, run_index: int, report: Report, group: str) -> ArmFacts:
    directory = paths.arm_dir(cell.dataset, cell.subset, cell.system, arm, run_index)
    facts = ArmFacts(arm=arm, run_index=run_index, directory=directory)
    config_path = directory / "run_config.json"
    if config_path.is_file():
        try:
            facts.config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - any unreadable config is the finding
            report.fail(group, "run_config_unreadable", f"{rel(config_path)}: {exc}")
    elif directory.is_dir():
        report.fail(group, "run_config_missing", rel(config_path))
    facts.outputs, problems = read_jsonl_strict(directory / "outputs.jsonl")
    for problem in problems:
        # A cell whose arm has not started is not a finding: the retrieval file alone
        # makes the cell "active", and the arms stage has not reached it yet. An arm
        # *directory* with an unreadable or absent output file is.
        if directory.is_dir():
            report.fail(group, "outputs_unreadable", f"{cell.label}/{facts.label}: {problem}")
    facts.judge, problems = read_jsonl_strict(directory / "judge.jsonl")
    for problem in problems:
        # A judge file that does not exist yet is normal mid-pipeline; the coverage
        # check in ``rep_judge`` is what decides whether that matters.
        if "missing file" not in problem:
            report.fail(group, "judge_unreadable", f"{cell.label}/{facts.label}: {problem}")
    return facts


# --- group: retrieval ------------------------------------------------------


def check_retrieval(cell: Cell, report: Report, row_limit: int) -> CellFacts:
    facts = CellFacts()
    path = cell.retrieval_file
    if not path.is_file():
        report.fail("retrieval", "retrieval_file_missing", rel(path))
        return facts
    facts.retrieval_sha = sha256_file(path)
    rows, problems = read_jsonl_strict(path)
    for problem in problems:
        report.fail("retrieval", "retrieval_row_unreadable", f"{cell.label}: {problem}")
    facts.retrieval_rows = rows
    if not rows:
        report.fail("retrieval", "retrieval_file_empty", rel(path))
        return facts

    expected = dataset_adapters.sample_count(cell.task)
    if len(rows) > expected:
        report.fail(
            "retrieval", "retrieval_rows_exceed_dataset",
            f"{cell.label}: {len(rows)} rows for {expected} samples in the source file",
        )
    elif len(rows) < expected:
        report.warn(
            "retrieval", "retrieval_file_partial",
            f"{cell.label}: {len(rows)}/{expected} rows -- pilot or in-progress freeze; "
            f"the arms may only cover what is here",
        )
    else:
        report.ok("retrieval", "retrieval_coverage", f"{cell.label}: {len(rows)} rows")

    dupes = duplicate_ids(ids_of(rows))
    if dupes:
        report.fail("retrieval", "retrieval_duplicate_ids", f"{cell.label}: {dupes[:5]}")

    ids = ids_of(rows)
    if ids != sorted(ids):
        report.warn(
            "retrieval", "retrieval_not_in_source_order",
            f"{cell.label}: first out-of-order id "
            f"{next((b for a, b in zip(ids, sorted(ids)) if a != b), '?')}",
        )

    checked = 0
    for row in rows:
        if row_limit and checked >= row_limit:
            break
        checked += 1
        sample_id = str(row.get("sample_id"))
        missing = sorted({"task", "current_request", "retrieved_memories", "retrieved_count",
                          "memory_config", "memory_system", "official_context_text",
                          "benchmark_row"} - set(row))
        if missing:
            report.fail("retrieval", "retrieval_row_keys", f"{sample_id}: missing {missing}")
            continue
        if str(row["task"]) != cell.task:
            report.fail("retrieval", "retrieval_row_task", f"{sample_id}: {row['task']!r}")
        if str(row["memory_system"]) != run_config.METHOD_FOR_SYSTEM[cell.system]:
            # The row carries the *method* label (``A-MEM``/``MemZero``/``NaiveRAG``),
            # which is what indexes ``MEMORY_LAYERS_MAPPING`` and what a reader of the
            # frozen file sees; the CLI system label is the shorter alias.
            report.fail(
                "retrieval", "retrieval_row_system",
                f"{sample_id}: {row['memory_system']!r} != "
                f"{run_config.METHOD_FOR_SYSTEM[cell.system]!r}",
            )
        memories = row["retrieved_memories"]
        if not isinstance(memories, list) or not memories:
            report.fail("retrieval", "retrieval_row_no_memories", sample_id)
            continue
        if int(row["retrieved_count"]) != len(memories):
            report.fail(
                "retrieval", "retrieval_count_mismatch",
                f"{sample_id}: retrieved_count={row['retrieved_count']} "
                f"but {len(memories)} memories",
            )
        bad = [m for m in memories if not CANONICAL_KEYS.issubset(m)]
        if bad:
            report.fail(
                "retrieval", "retrieval_memory_schema",
                f"{sample_id}: {len(bad)} memories without the canonical five keys",
            )
        if not str(row["official_context_text"]).strip():
            report.fail("retrieval", "retrieval_context_text_empty", sample_id)
        # The depth bound is per row, not per cell: PersistBench sets top_k to each
        # sample's own given-memory count (4-16), so a cell-level bound would both miss
        # a truncated sample and flag a generous one.
        config = row.get("memory_config") or {}
        bound = config.get("top_k")
        if bound is None:
            bound = len((row.get("benchmark_row") or {}).get("memories") or [])
        if isinstance(bound, int) and len(memories) > bound:
            report.fail(
                "retrieval", "retrieval_depth_exceeded",
                f"{sample_id}: {len(memories)} memories for top_k={bound}",
            )
        if not cell.spec.is_memtrap:
            # PersistBench's validity condition: the retrieval depth is the whole given
            # set. A smaller top_k would silently truncate over half the memories for the
            # 259 samples that carry more than ten.
            given = len((row.get("benchmark_row") or {}).get("memories") or [])
            if config.get("top_k") != given:
                report.fail(
                    "retrieval", "retrieval_top_k_not_given_set",
                    f"{sample_id}: top_k={config.get('top_k')} for {given} given memories",
                )
            if config.get("top_k_policy") != "per_sample_len_memories":
                report.fail(
                    "retrieval", "retrieval_top_k_policy",
                    f"{sample_id}: {config.get('top_k_policy')!r}",
                )
        elif config.get("top_k_policy") != "fixed":
            report.fail(
                "retrieval", "retrieval_top_k_policy",
                f"{sample_id}: {config.get('top_k_policy')!r}",
            )
    report.ok("retrieval", "retrieval_rows_checked", f"{cell.label}: {checked} rows")
    return facts


# --- group: store ----------------------------------------------------------


def check_stores(cell: Cell, facts: CellFacts, report: Report, limit: int) -> None:
    """Recompute each checked row's store digest and compare it with the marker.

    ``limit`` defaults low: this is the only group that touches the store directories,
    and the check is about provenance, so a sample of them proves the pipeline rather
    than every file in a 1,050-row cell.
    """
    rows = facts.retrieval_rows
    if limit:
        rows = rows[:limit]
    if not rows:
        return
    indices = sorted({int(row.get("row_index") or 0) for row in rows} - {0})
    if not indices:
        # ``row_index`` is part of the frozen row schema; without it the adapter cannot
        # be re-adapted and the digest cannot be recomputed at all.
        report.fail("store", "store_row_index_missing", f"{cell.label}: no row_index")
        return
    try:
        samples = {s.sample_id: s for s in dataset_adapters.load_samples(cell.task, indices=indices)}
    except Exception as exc:  # noqa: BLE001 - an unreadable dataset is the finding
        report.fail("store", "store_dataset_unreadable", f"{cell.label}: {exc}")
        return
    facts.samples = samples

    for row in rows:
        sample_id = str(row.get("sample_id"))
        sample = samples.get(sample_id)
        config = row.get("memory_config") or {}
        if sample is None:
            report.fail("store", "store_sample_absent", f"{cell.label}/{sample_id}")
            continue

        expected_digest = build_retrieval.expected_store_digest(sample)
        if config.get("marker_digest") != expected_digest:
            report.fail(
                "store", "store_digest_mismatch",
                f"{sample_id}: marker {str(config.get('marker_digest'))[:12]} != "
                f"recomputed {expected_digest[:12]} (the store was built from a "
                f"different message list than this row claims)",
            )
        else:
            report.ok("store", "store_digest", sample_id)

        if config.get("messages_count") != sample.messages_count:
            report.fail(
                "store", "store_messages_count",
                f"{sample_id}: {config.get('messages_count')} != {sample.messages_count}",
            )
        if config.get("marker_messages_count") != sample.messages_count:
            report.fail(
                "store", "store_marker_messages_count",
                f"{sample_id}: marker {config.get('marker_messages_count')} != "
                f"{sample.messages_count}",
            )
        if config.get("prior_sha256") != sample.prior_sha256():
            report.fail(
                "store", "store_prior_sha256",
                f"{sample_id}: the memory prior this row was built from is not the one "
                f"the adapter produces now",
            )
        if config.get("ingest_mode") != EXPECTED_INGEST[cell.dataset]:
            report.fail(
                "store", "store_ingest_mode",
                f"{sample_id}: {config.get('ingest_mode')!r} != {EXPECTED_INGEST[cell.dataset]!r}",
            )
        if config.get("ingest_mode") != sample.ingest_mode:
            report.fail(
                "store", "store_ingest_mode_adapter",
                f"{sample_id}: row {config.get('ingest_mode')!r} != "
                f"adapter {sample.ingest_mode!r}",
            )
        # ``top_k`` in the row is the depth the memory layer *used*, which resolves the
        # sample's override against the run config's default. So the invariant is not
        # "equals the sample's override" -- MemTrapBench has no override -- but "the
        # policy and the recorded depth agree with whether an override exists".
        policy = config.get("top_k_policy")
        per_sample = sample.top_k is not None
        if per_sample != (policy == "per_sample_len_memories"):
            report.fail(
                "store", "store_top_k_policy",
                f"{sample_id}: policy {policy!r} but the adapter's override is "
                f"{sample.top_k!r}",
            )
        if per_sample and config.get("top_k") != sample.top_k:
            report.fail(
                "store", "store_top_k",
                f"{sample_id}: recorded depth {config.get('top_k')} != "
                f"{sample.top_k} given memories",
            )
        if not isinstance(config.get("top_k"), int) or config["top_k"] <= 0:
            report.fail(
                "store", "store_top_k_schema",
                f"{sample_id}: top_k={config.get('top_k')!r}",
            )
        if str(config.get("store_status")) not in {"built", "reused"}:
            report.fail(
                "store", "store_status",
                f"{sample_id}: {config.get('store_status')!r}",
            )

        save_dir = Path(str(config.get("save_dir") or ""))
        if not save_dir.is_dir():
            report.fail("store", "store_dir_missing", f"{sample_id}: {rel(save_dir)}")
            continue
        marker = save_dir / ".memory_complete.json"
        if not marker.is_file():
            report.fail("store", "store_marker_missing", rel(marker))
            continue
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            report.fail("store", "store_marker_unreadable", f"{rel(marker)}: {exc}")
            continue
        if payload.get("digest") != expected_digest:
            report.fail(
                "store", "store_marker_digest",
                f"{sample_id}: marker file {str(payload.get('digest'))[:12]} != "
                f"recomputed {expected_digest[:12]}",
            )
        if payload.get("messages_count") != sample.messages_count:
            report.fail(
                "store", "store_marker_file_count",
                f"{sample_id}: {payload.get('messages_count')} != {sample.messages_count}",
            )
        if cell.system == "AMEM":
            # A-MEM keeps a Chroma directory plus a pickled note store; a "reused" store
            # that lost either would still answer a query, just with other memories.
            chroma = save_dir / "chroma"
            user_id = str(config.get("user_id") or "")
            pickle = save_dir / f"{user_id}.pkl"
            if not chroma.is_dir():
                report.fail("store", "store_amem_chroma_missing", f"{sample_id}: {rel(chroma)}")
            if not user_id or not pickle.is_file():
                report.fail("store", "store_amem_pickle_missing", f"{sample_id}: {rel(pickle)}")
    report.ok("store", "store_rows_checked", f"{cell.label}: {len(rows)} rows")


# --- group: alignment ------------------------------------------------------


def check_alignment(cell: Cell, facts: CellFacts, arm: ArmFacts, report: Report) -> None:
    tag = f"{cell.label}/{arm.label}"
    if not arm.outputs:
        # An arm directory with no outputs is a cell that has not generated yet; the
        # caller decides whether that is a gap (arms) or simply not started.
        return
    retrieval_ids = ids_of(facts.retrieval_rows)
    output_ids = ids_of(arm.outputs)
    if duplicate_ids(output_ids):
        report.fail("alignment", "outputs_duplicate_ids", f"{tag}: {duplicate_ids(output_ids)[:5]}")
    if output_ids != retrieval_ids[: len(output_ids)]:
        mismatch = next(
            (i for i, (a, b) in enumerate(zip(output_ids, retrieval_ids)) if a != b),
            min(len(output_ids), len(retrieval_ids)),
        )
        report.fail(
            "alignment", "outputs_not_retrieval_prefix",
            f"{tag}: first divergence at row {mismatch} "
            f"({output_ids[mismatch:mismatch+1]} vs {retrieval_ids[mismatch:mismatch+1]})",
        )
    else:
        report.ok("alignment", "outputs_are_retrieval_prefix", f"{tag}: {len(output_ids)} rows")

    if arm.config is not None:
        recorded = str(arm.config.get("retrieval_sha256") or "")
        if recorded != facts.retrieval_sha:
            report.fail(
                "alignment", "config_retrieval_sha256",
                f"{tag}: config {recorded[:12]} != file {facts.retrieval_sha[:12]} "
                f"({rel(cell.retrieval_file)})",
            )
        declared = str(arm.config.get("retrieval_path") or "")
        if declared and Path(declared).resolve() != cell.retrieval_file.resolve():
            report.fail(
                "alignment", "config_retrieval_path",
                f"{tag}: {rel(declared)} != {rel(cell.retrieval_file)}",
            )
        sample_count = arm.config.get("sample_count")
        if sample_count != len(facts.retrieval_rows):
            report.fail(
                "alignment", "config_sample_count",
                f"{tag}: config says {sample_count}, retrieval file has "
                f"{len(facts.retrieval_rows)} rows -- one of them changed after the other",
            )

    for row in arm.outputs:
        sample_id = str(row.get("sample_id"))
        recorded = row.get("retrieval_sha256")
        if recorded is not None and recorded != facts.retrieval_sha:
            report.fail(
                "alignment", "record_retrieval_sha256",
                f"{tag}/{sample_id}: {str(recorded)[:12]} != {facts.retrieval_sha[:12]}",
            )
        # Each arm's own count against its own list. The arms group compares the two
        # memory *lists*; without this, a count edited on one side alone would only be
        # visible if the other side happened to disagree with it.
        memories = row.get("retrieved_memories") or []
        if not memories:
            report.fail("alignment", "record_no_memories", f"{tag}/{sample_id}")
        elif int(row.get("retrieved_count") or 0) != len(memories):
            report.fail(
                "alignment", "record_retrieved_count",
                f"{tag}/{sample_id}: retrieved_count {row.get('retrieved_count')} != "
                f"{len(memories)} memories",
            )
    # MemTrapBench has no per-sample depth override, so the frozen file's depth is the
    # run config's, and any divergence means the file was written under another protocol.
    if arm.config is not None and cell.spec.is_memtrap and facts.retrieval_rows:
        declared_top_k = arm.config.get("top_k")
        mismatched = sorted(
            {
                int(row.get("memory_config", {}).get("top_k") or -1)
                for row in facts.retrieval_rows
            }
            - {declared_top_k}
        )
        if mismatched:
            report.fail(
                "alignment", "config_top_k",
                f"{tag}: retrieval file was built at top_k {mismatched}, config says "
                f"{declared_top_k}",
            )
    _check_record_configs(cell, arm, report)


def _check_record_configs(cell: Cell, arm: ArmFacts, report: Report) -> None:
    """Each record's embedded shared config must be the one on disk beside it.

    A record carries the config it was produced under, which is what makes a stage
    cache hit or a file copied between cells detectable at all: the values inside a
    record are otherwise self-consistent and perfectly plausible.
    """
    if arm.config is None:
        return
    expected = arm.shared()
    tag = f"{cell.label}/{arm.label}"
    for kind, rows in (("output", arm.outputs), ("judge", arm.judge)):
        for row in rows:
            embedded = row.get("config")
            if embedded is None:
                continue
            if embedded != expected:
                differing = sorted(
                    name for name in run_config.RunConfig.SHARED_FIELDS
                    if embedded.get(name) != expected.get(name)
                )
                report.fail(
                    "alignment", "record_config",
                    f"{tag}/{kind} {row.get('sample_id')}: config differs from "
                    f"run_config.json in {differing[:5] or 'key set'}",
                )
                break


# --- group: arms -----------------------------------------------------------


def check_arm_identity(
    cell: Cell, facts: CellFacts, pair: dict[str, ArmFacts], report: Report, pair_limit: int
) -> None:
    """The intervention is one prompt and nothing else."""
    baseline, memadapter = pair.get("baseline"), pair.get("memadapter")
    run_index = next(iter(pair.values())).run_index if pair else 0
    tag = f"{cell.label}/run{run_index}"
    if baseline is None or memadapter is None:
        report.warn(
            "arms", "arm_pair_incomplete",
            f"{cell.label}: only {sorted(a for a, f in pair.items() if f.outputs or f.config)} "
            f"present; the arm comparison cannot run yet",
        )
        return

    # (i) the stored shared config, field by field so the drifting field is named.
    left, right = baseline.shared(), memadapter.shared()
    if baseline.config is None or memadapter.config is None:
        report.fail("arms", "arm_config_missing", tag)
    else:
        diffs = [name for name in run_config.RunConfig.SHARED_FIELDS if left[name] != right[name]]
        if diffs:
            detail = "; ".join(f"{name}: {left[name]!r} != {right[name]!r}" for name in diffs[:5])
            report.fail("arms", "arm_shared_config", f"{tag}: {detail}")
        else:
            report.ok("arms", "arm_shared_config", f"{tag}: 27 shared fields equal")
        if baseline.config.get("shared_fingerprint") != memadapter.config.get("shared_fingerprint"):
            report.fail(
                "arms", "arm_shared_fingerprint",
                f"{tag}: {baseline.config.get('shared_fingerprint')} != "
                f"{memadapter.config.get('shared_fingerprint')}",
            )
        for arm, arm_facts in pair.items():
            stored = (arm_facts.config or {}).get("shared_fingerprint")
            recomputed = run_config.shared_fingerprint_of(arm_facts.shared())
            if stored != recomputed:
                report.fail(
                    "arms", "arm_shared_fingerprint_stale",
                    f"{tag}/{arm}: stored {str(stored)[:16]} != recomputed "
                    f"{recomputed[:16]} from the fields stored beside it",
                )
        if baseline.config.get("run_fingerprint") != memadapter.config.get("run_fingerprint"):
            report.fail(
                "arms", "arm_run_fingerprint",
                f"{tag}: {baseline.config.get('run_fingerprint')} != "
                f"{memadapter.config.get('run_fingerprint')}",
            )

    # The intervention itself: a different answer prompt on each side, each hashed from
    # the file on disk, each the file it is supposed to be.
    digests: dict[str, str] = {}
    for arm, arm_facts in pair.items():
        config = arm_facts.config or {}
        prompt_path = resolve_config_prompt(config)
        if prompt_path is None:
            report.fail("arms", "arm_answer_prompt_missing", f"{tag}/{arm}")
            continue
        digest = sha256_file(prompt_path)
        digests[arm] = digest
        recorded = str(config.get("answer_prompt_sha256") or "")
        if recorded != digest:
            report.fail(
                "arms", "arm_answer_prompt_sha256",
                f"{tag}/{arm}: config {recorded[:12]} != file {digest[:12]} "
                f"({rel(prompt_path)})",
            )
        expected = EXPECTED_ANSWER_PROMPT[arm][cell.dataset]
        actual = rel(prompt_path)
        if actual != expected:
            report.fail(
                "arms", "arm_answer_prompt_identity",
                f"{tag}/{arm}: uses {actual}, expected {expected}",
            )
    if len(digests) == 2 and digests["baseline"] == digests["memadapter"]:
        report.fail(
            "arms", "arm_intervention_absent",
            f"{tag}: both arms send the same answer prompt; the intervention is missing",
        )
    elif len(digests) == 2:
        report.ok("arms", "arm_intervention", f"{tag}: answer prompts differ")

    # (ii) per-record generation settings, paired by sample_id.
    paired = [(a, b) for a, b in zip(baseline.outputs, memadapter.outputs)
              if a.get("sample_id") == b.get("sample_id")]
    if not paired:
        report.warn("arms", "arm_records_unpaired", f"{tag}: no sample_id present on both sides")
    mismatched = 0
    checked = 0
    for left_row, right_row in paired:
        if pair_limit and checked >= pair_limit:
            break
        checked += 1
        for field in ("generation_model", "generation_base_url", "generation_temperature",
                      "generation_max_tokens"):
            if left_row.get(field) != right_row.get(field):
                mismatched += 1
                report.fail(
                    "arms", "arm_record_settings",
                    f"{tag}/{left_row.get('sample_id')}: {field} "
                    f"{left_row.get(field)!r} != {right_row.get(field)!r}",
                )
        if left_row.get("run_fingerprint") != right_row.get("run_fingerprint"):
            mismatched += 1
            report.fail(
                "arms", "arm_record_fingerprint",
                f"{tag}/{left_row.get('sample_id')}: "
                f"{left_row.get('run_fingerprint')} != {right_row.get('run_fingerprint')}",
            )
        # (iv) the memory text both arms were shown, which is the condition the two
        # answers are supposed to differ *under*. The baseline keeps the frozen
        # canonical objects and MemAdapter keeps only the text, so compare the texts.
        left_text = memory_texts(left_row.get("retrieved_memories") or [])
        right_text = memory_texts(right_row.get("retrieved_memories") or [])
        if left_text != right_text:
            mismatched += 1
            report.fail(
                "arms", "arm_record_memories",
                f"{tag}/{left_row.get('sample_id')}: {len(left_text)} vs {len(right_text)} "
                f"memories; first difference at "
                f"{next((i for i, (a, b) in enumerate(zip(left_text, right_text)) if a != b), '-')}",
            )
    if paired and not mismatched:
        report.ok("arms", "arm_records", f"{tag}: {checked} pairs identical in settings and memories")

    # (iii) the judge protocol: the judge is shared, so its records must agree on
    # everything the arms could differ in.
    judged_ids = set(ids_of(baseline.judge)) & set(ids_of(memadapter.judge))
    by_id_left = {str(r.get("sample_id")): r for r in baseline.judge}
    by_id_right = {str(r.get("sample_id")): r for r in memadapter.judge}
    judge_fields = ("judge_model", "judge_base_url", "judge_temperature", "judge_max_tokens",
                    "judge_prompt_mode", "rubric_sha256", "run_fingerprint")
    judge_mismatch = 0
    for sample_id in sorted(judged_ids)[: pair_limit or None]:
        for field in judge_fields:
            if by_id_left[sample_id].get(field) != by_id_right[sample_id].get(field):
                judge_mismatch += 1
                report.fail(
                    "arms", "arm_judge_field",
                    f"{tag}/{sample_id}: {field} "
                    f"{by_id_left[sample_id].get(field)!r} != "
                    f"{by_id_right[sample_id].get(field)!r}",
                )
        if by_id_left[sample_id].get("config") != by_id_right[sample_id].get("config"):
            judge_mismatch += 1
            report.fail("arms", "arm_judge_config", f"{tag}/{sample_id}: shared config differs")
    if judged_ids and not judge_mismatch:
        report.ok("arms", "arm_judge_protocol", f"{tag}: {len(judged_ids)} paired judgements agree")


# --- group: judge ----------------------------------------------------------


def check_judge(
    cell: Cell, arm: ArmFacts, report: Report, rubric: str, expected_dims: list[str],
    limit: int, verbose: bool,
) -> None:
    """Rubric compliance and recomputability of every recorded judgement."""
    tag = f"{cell.label}/{arm.label}"
    if not arm.judge:
        return
    low, high = judge_persist.expected_range(cell.spec)
    checked = 0
    for row in arm.judge:
        if limit and checked >= limit:
            break
        checked += 1
        sample_id = str(row.get("sample_id"))

        recorded_rubric = str(row.get("rubric_sha256") or "")
        if recorded_rubric != rubric:
            report.fail(
                "judge", "judge_rubric_sha256",
                f"{tag}/{sample_id}: record {recorded_rubric[:12]} != file {rubric[:12]}",
            )
        rubric_file = row.get("rubric_file")
        if rubric_file and rel(rubric_file) != rel(build_rubric_path(cell)):
            report.fail(
                "judge", "judge_rubric_file",
                f"{tag}/{sample_id}: {rel(rubric_file)} != {rel(build_rubric_path(cell))}",
            )
        if str(row.get("arm") or "") != arm.arm:
            report.fail(
                "judge", "judge_arm_label",
                f"{tag}/{sample_id}: record says arm {row.get('arm')!r}",
            )
        if int(row.get("run_index") or 0) != arm.run_index:
            report.fail(
                "judge", "judge_run_index",
                f"{tag}/{sample_id}: record says run {row.get('run_index')!r}",
            )
        if not str(row.get("final_answer") or "").strip():
            report.fail("judge", "judge_empty_answer", f"{tag}/{sample_id}")

        if cell.spec.is_memtrap:
            scores = row.get("dimension_scores")
            if not isinstance(scores, dict) or not scores:
                report.fail("judge", "judge_dimensions_missing", f"{tag}/{sample_id}")
                continue
            if sorted(scores) != sorted(expected_dims):
                report.fail(
                    "judge", "judge_dimensions",
                    f"{tag}/{sample_id}: {sorted(scores)} != {sorted(expected_dims)} "
                    f"(rubric {cell.spec.rubric_file})",
                )
            out_of_range = {k: v for k, v in scores.items() if not 0 <= v <= 5}
            if out_of_range:
                report.fail("judge", "judge_score_range", f"{tag}/{sample_id}: {out_of_range}")
            if row.get("dimension_count") != len(scores):
                report.fail(
                    "judge", "judge_dimension_count",
                    f"{tag}/{sample_id}: {row.get('dimension_count')} != {len(scores)}",
                )
            recomputed, problems = judge_common.score_memtrap_dimensions(
                row.get("dimensions") or {}, expected_dims
            )
            if problems:
                report.fail(
                    "judge", "judge_rescore",
                    f"{tag}/{sample_id}: recorded judgement no longer parses: {problems[0]}",
                )
            elif recomputed != scores:
                report.fail(
                    "judge", "judge_rescore",
                    f"{tag}/{sample_id}: stored {scores} != recomputed {recomputed}",
                )
            elif verbose:
                report.ok("judge", "judge_rescore", f"{tag}/{sample_id}")
        else:
            if row.get("score_scale_low") != low or row.get("score_scale_high") != high:
                report.fail(
                    "judge", "judge_score_scale",
                    f"{tag}/{sample_id}: {row.get('score_scale_low')}-"
                    f"{row.get('score_scale_high')} != {low}-{high}",
                )
            key = str(row.get("score_key") or "")
            if key not in {"score", "rating"}:
                report.fail("judge", "judge_score_key", f"{tag}/{sample_id}: {key!r}")
            elif key != cell.spec.score_key:
                # ``score_persist`` accepts either key within range, so this is a
                # deviation from the rubric's own field rather than an error.
                report.warn(
                    "judge", "judge_score_key_deviation",
                    f"{tag}/{sample_id}: scored from {key!r}, rubric asks for "
                    f"{cell.spec.score_key!r}",
                )
            score = row.get("score")
            if not isinstance(score, int) or not low <= score <= high:
                report.fail(
                    "judge", "judge_score_range",
                    f"{tag}/{sample_id}: {score!r} outside {low}-{high}",
                )
            recomputed, recomputed_key, problems = judge_persist.score_persist(
                row.get("judge_parsed") or {}, low=low, high=high
            )
            if problems:
                report.fail(
                    "judge", "judge_rescore",
                    f"{tag}/{sample_id}: recorded judgement no longer parses: {problems[0]}",
                )
            elif (recomputed, recomputed_key) != (score, key):
                report.fail(
                    "judge", "judge_rescore",
                    f"{tag}/{sample_id}: stored {score!r}/{key!r} != recomputed "
                    f"{recomputed!r}/{recomputed_key!r}",
                )
            elif verbose:
                report.ok("judge", "judge_rescore", f"{tag}/{sample_id}")
    report.ok("judge", "judge_rows_checked", f"{tag}: {checked} rows")


def build_rubric_path(cell: Cell) -> Path:
    return paths.RUBRICS_ROOT / cell.spec.rubric_dir / cell.spec.rubric_file


# --- driver ----------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="", help="Comma-separated dataset filter.")
    parser.add_argument("--subset", default="", help="Comma-separated subset filter.")
    parser.add_argument("--system", default="", help="Comma-separated memory-system filter.")
    parser.add_argument(
        "--check-arm-identity",
        action="store_true",
        help=(
            "Run only the arm-identity group. Named in the plan's validation table; "
            "without it every group runs."
        ),
    )
    parser.add_argument("--groups", default="", help="Comma-separated group selection.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap the rows checked per check (0 = all). Applies to row-wise checks.",
    )
    parser.add_argument(
        "--store-limit",
        type=int,
        default=25,
        help=(
            "Rows whose store is re-hashed and re-read per cell (0 = all). The store "
            "check is the only one that touches Chroma/Qdrant directories, so it is "
            "sampled by default."
        ),
    )
    parser.add_argument("--max-detail", type=int, default=5)
    parser.add_argument("--verbose", action="store_true", help="Print OK findings too.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_arm_identity:
        args.groups = "arms"
    selected = [g.strip() for g in args.groups.split(",") if g.strip()]
    for group in selected:
        if group not in GROUPS:
            raise SystemExit(f"unknown group {group!r}; choose from {', '.join(GROUPS)}")
    active_group = (lambda group: not selected or group in selected)

    report = Report()
    active, pending = discover(args)
    if pending:
        print(
            f"not started ({len(pending)} cell(s), not checked): "
            + ", ".join(f"{c.dataset}/{c.subset}/{c.system}" for c in pending[:6])
            + (" ..." if len(pending) > 6 else "")
        )

    for cell in active:
        facts = CellFacts()
        if active_group("retrieval") or active_group("store") or active_group("alignment"):
            facts = check_retrieval(cell, report, args.limit or 0) if active_group("retrieval") else _bare_rows(cell)
        # Arms are loaded whenever any group needs them: alignment needs the config and
        # outputs, arms needs both, and judge needs the judgements.
        needs_arms = any(active_group(g) for g in ("alignment", "arms", "judge"))
        arms: dict[str, ArmFacts] = {}
        if needs_arms:
            # Findings from loading an arm's files are attributed to whichever group the
            # caller is actually running, so ``--check-arm-identity`` does not report
            # them under ``alignment``, which it was not asked about.
            load_group = "alignment" if active_group("alignment") else "arms"
            for run_index in arm_run_indices(cell):
                for arm in run_config.ARMS:
                    arms[f"{arm}:{run_index}"] = load_arm(cell, arm, run_index, report, load_group)
        if active_group("store"):
            check_stores(cell, facts, report, args.store_limit)
        if active_group("alignment"):
            for key, arm in arms.items():
                if arm.directory.is_dir():
                    check_alignment(cell, facts, arm, report)
        if active_group("arms"):
            for run_index in sorted({arm.run_index for arm in arms.values()}):
                # Only arms that have actually started belong in the pair: an absent
                # directory is a cell the arms stage has not reached, and comparing a
                # real arm against an empty one would report the emptiness as drift.
                pair = {
                    arm: arms[f"{arm}:{run_index}"]
                    for arm in run_config.ARMS
                    if f"{arm}:{run_index}" in arms and arms[f"{arm}:{run_index}"].directory.is_dir()
                }
                if pair:
                    check_arm_identity(cell, facts, pair, report, args.limit or 0)
        if active_group("judge"):
            paths_missing = False
            rubric_path = build_rubric_path(cell)
            if not rubric_path.is_file():
                report.fail("judge", "rubric_file_missing", rel(rubric_path))
                paths_missing = True
            else:
                template = read_prompt_file(rubric_path)
                rubric = run_config.sha256_text(template)
                expected_dims = (
                    judge_common.declared_dimensions(template) if cell.spec.is_memtrap else []
                )
                for arm in arms.values():
                    if arm.judge:
                        check_judge(
                            cell, arm, report, rubric, expected_dims,
                            args.limit or 0, args.verbose,
                        )
            if paths_missing:
                continue
            for arm in arms.values():
                _report_coverage(cell, arm, report)

    print(report.render(args.max_detail, args.verbose))
    return 1 if report.failed else 0


def _bare_rows(cell: Cell) -> CellFacts:
    """Retrieval rows without running the retrieval checks (group filtered out)."""
    rows, problems = read_jsonl_strict(cell.retrieval_file)
    facts = CellFacts(retrieval_rows=[] if problems else rows)
    if facts.retrieval_rows:
        facts.retrieval_sha = sha256_file(cell.retrieval_file)
    return facts


def _report_coverage(cell: Cell, arm: ArmFacts, report: Report) -> None:
    """Judged/generated, with the same 0.98 publishing gate the judge itself applies."""
    total = len(arm.outputs)
    if not total:
        return
    done = len(arm.judge)
    ratio = done / total
    tag = f"{cell.label}/{arm.label}"
    if ratio < 0.98:
        report.fail(
            "judge", "judge_coverage",
            f"{tag}: {done}/{total} ({ratio:.3f}) -- below the 0.98 gate; do not report "
            f"this cell's aggregate",
        )
    else:
        report.ok("judge", "judge_coverage", f"{tag}: {done}/{total}")


if __name__ == "__main__":
    raise SystemExit(main())
