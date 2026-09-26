"""Frozen protocol for the baseline / MemAdapter runs on the external benchmarks.

The two arms must be indistinguishable except for the final-answer prompt, which
*is* the intervention. Everything else -- generation model and endpoint,
temperature, max_tokens, thinking setting, retrieval file, judge model and
rubric, sample set, generations per sample -- is resolved once here and copied
verbatim into both arms' ``run_config.json``.

Two design choices make the identity guarantee cheap to enforce:

*   The model client (``MemAdapter/model_client.py``) reads its temperature and
    token budget from the *environment* (``MEMADAPTER_TEMPERATURE``,
    ``MEMADAPTER_MAX_TOKENS``), not from a per-arm argument. Both arms therefore
    inherit the identical values as long as the launcher exports them once. This
    module records the resolved values and refuses to proceed if they drift.
*   Retrieval is frozen to one JSONL per ``(dataset, subset, system)`` and both
    arms read that same file, so the retrieved-memory condition is byte-identical
    by construction rather than by convention.

API keys are never read, echoed or hashed here -- only the *names* of the
variables that were set, so a run config can be archived with the paper.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import paths, task_registry

# --- arms -----------------------------------------------------------------

ARMS = ("baseline", "memadapter")
INTERVENTION_ARMS = ("anti_sycophancy", "self_recheck", "dynamic_partition", "memgate")
SUPPORTED_ARMS = ARMS + INTERVENTION_ARMS

#: Python method name under ``memories.MEMORY_LAYERS_MAPPING`` and the baselines
#: config loader, per memory-system label used on the command line.
METHOD_FOR_SYSTEM: dict[str, str] = {
    "AMEM": "A-MEM",
    "Mem0": "MemZero",
    "naiveRAG": "NaiveRAG",
}

METHOD_FOR_SYSTEM.update({"MemoryBank": "MemoryBank", "LightMem": "LightMem"})
SYSTEMS = tuple(METHOD_FOR_SYSTEM)
SUPPORTED_SYSTEMS = SYSTEMS

#: Memory-systems whose layers run an LLM while ingesting, and therefore cannot
#: be assumed to preserve a caller-supplied memory string verbatim.
LLM_INGEST_SYSTEMS = frozenset({"A-MEM", "MemZero"})

#: Ingest modes. ``messages`` inserts synthesized chat turns through the normal
#: dialogue path (MemTrapBench); ``raw`` writes the given memory strings straight
#: into the store without a speaker prefix or LLM re-extraction (PersistBench).
INGEST_MODES = ("messages", "raw", "raw_dialogue_fallback")

#: Repo-root env file holding the non-secret protocol pins. Loaded explicitly by
#: the launcher and by dev scripts; see :func:`load_env_file`.
DEFAULT_ENV_FILE = "config/.external_bench_20260917.env"

#: Environment variables the model client consumes. Recorded by name so the run
#: config stays archivable; values are recorded only for the non-secret ones.
MODEL_ENV_KEYS = (
    "MEMADAPTER_TEMPERATURE",
    "MEMADAPTER_MAX_TOKENS",
    "MEMADAPTER_ENABLE_REASONING",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
)

DEFAULT_TEMPERATURE = "0.7"
DEFAULT_MAX_TOKENS = "4096"

#: The one field that is *supposed* to differ between arms -- it is the intervention.
#: ``assert_arm_identity`` fails if the two arms share it, so both arms name it here
#: and neither invents its own value. The baseline arm answers with each dataset's own
#: official prompt, which is why it is keyed by dataset as well as by arm.
FINAL_ANSWER_PROMPTS = {
    "baseline": {
        "memtrapbench": "memtrap/prompt_user_mem.txt",
        "persistbench": "persist/generator.txt",
    },
    "memadapter": {
        "memtrapbench": "MemAdapter/prompts/03_memory_use_guided_generation.txt",
        "persistbench": "MemAdapter/prompts/03_memory_use_guided_generation.txt",
    },
}

#: Where each arm's prompt paths are rooted: the baseline arm's are the datasets'
#: own rubrics, the MemAdapter arm's ship with the method.
FINAL_ANSWER_ROOTS = {"baseline": paths.RUBRICS_ROOT, "memadapter": paths.ROOT}
for _arm in INTERVENTION_ARMS:
    FINAL_ANSWER_PROMPTS[_arm] = dict(FINAL_ANSWER_PROMPTS["baseline"])
    FINAL_ANSWER_ROOTS[_arm] = paths.RUBRICS_ROOT


def final_answer_prompt_path(arm: str, dataset: str) -> Path:
    """The file that actually produces ``arm``'s final answer on ``dataset``.

    ``make_config`` defaults ``answer_prompt_*`` to the dataset's official answer
    prompt, which is true of the baseline arm and false of the MemAdapter arm -- that
    prompt is never sent there. Resolving it per arm gives the field one meaning on
    both sides: the file whose bytes produced the final answer.
    """
    return FINAL_ANSWER_ROOTS[arm] / FINAL_ANSWER_PROMPTS[arm][dataset]


class ProtocolDrift(RuntimeError):
    """Raised when the resolved protocol does not match the frozen expectation."""


# --- hashing ---------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {path}")
    return sha256_bytes(path.read_bytes())


# --- resolved model identity ----------------------------------------------


def resolved_generation_model() -> str:
    """The generation model id as the client will actually send it."""
    return os.environ.get("DEEPSEEK_MODEL", "").strip() or "deepseek-v4-flash"


def generation_base_url() -> str:
    value = os.environ.get("DEEPSEEK_BASE_URL", "").strip()
    if not value:
        raise RuntimeError(
            "DEEPSEEK_BASE_URL must be supplied through the local environment; "
            "public repository code does not embed endpoints."
        )
    return value


def resolved_temperature() -> float:
    return float(os.environ.get("MEMADAPTER_TEMPERATURE") or DEFAULT_TEMPERATURE)


def resolved_max_tokens() -> int:
    return int(os.environ.get("MEMADAPTER_MAX_TOKENS") or DEFAULT_MAX_TOKENS)


def resolved_thinking() -> bool:
    """Whether DeepSeek reasoning mode is on.

    ``model_client`` disables thinking unless ``MEMADAPTER_ENABLE_REASONING`` is
    set, so an unset variable is a meaningful value ('off') rather than a gap.
    """
    raw = os.environ.get("MEMADAPTER_ENABLE_REASONING", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def load_env_file(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Load the ``KEY=VALUE`` protocol pins from a plain-text env file.

    ``override=False`` (the default) means the surrounding shell wins. That is
    the right precedence here because the file holds the non-secret protocol pins
    and never a credential: anything already exported is a deliberate operator
    choice, and silently clobbering it would defeat the point of pinning.

    Deliberately not ``python-dotenv``: the file is ours, the grammar needed is
    three lines, and keeping the parser visible means its quirks are too. Lines
    without ``=`` and lines starting with ``#`` are skipped. A bare ``KEY=`` sets
    the empty string, which ``MEMADAPTER_ENABLE_REASONING=`` relies on -- for
    ``model_client`` an unset-and-empty variable is a meaningful 'reasoning off'
    rather than a missing value.

    Returns the names it set, for logging. Values are never returned.
    """
    target = Path(path) if path is not None else paths.ROOT / DEFAULT_ENV_FILE
    if not target.is_file():
        raise FileNotFoundError(f"Protocol env file not found: {target}")

    applied: dict[str, str] = {}
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def env_file_conflicts(path: Path | None = None) -> list[str]:
    """Names of keys the env file pins differently from the live environment.

    :func:`load_env_file` lets the surrounding shell win, which is the right
    precedence -- but it wins silently, so a stale export can change the protocol
    with no signal at all. The model client makes that worse by falling back:
    ``ModelClient("DeepSeek")`` reads ``DEEPSEEK_API_KEY`` and then
    ``OPENAI_API_KEY``, so an unrelated gateway credential already in the shell is
    enough to produce a 401 that looks like a bad endpoint rather than a shadowed
    variable.

    Callers that must know whether they are running the pinned protocol use this and
    decide for themselves. Only names are returned; values are compared and dropped.
    """
    target = Path(path) if path is not None else paths.ROOT / DEFAULT_ENV_FILE
    if not target.is_file():
        return []
    conflicts: list[str] = []
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key in os.environ and os.environ[key] != value:
            conflicts.append(key)
    return sorted(conflicts)


def export_protocol_environment() -> dict[str, str]:
    """Pin the model client's knobs so both arms cannot drift apart.

    ``setdefault`` rather than assignment: an operator who deliberately exports a
    different value keeps it, and :func:`assert_protocol` then reports the drift
    instead of silently overwriting it.
    """
    os.environ.setdefault("MEMADAPTER_TEMPERATURE", DEFAULT_TEMPERATURE)
    os.environ.setdefault("MEMADAPTER_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    os.environ.setdefault("MEMADAPTER_ENABLE_REASONING", "")
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    return {key: os.environ.get(key, "") for key in MODEL_ENV_KEYS}


# --- the config ------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """One (dataset, subset, system, arm) cell of the experiment grid."""

    # --- identity of the cell ---
    dataset: str
    subset: str
    system: str
    arm: str
    run_index: int = 0

    # --- arm-independent condition: MUST match across arms ---
    generation_model: str = ""
    generation_base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 0
    thinking_enabled: bool = False
    judge_model: str = ""
    judge_base_url: str = ""
    judge_temperature: float = 0.0
    judge_max_tokens: int = 0
    judge_prompt_mode: str = ""
    k_generations: int = 1
    top_k: int | None = None
    top_k_policy: str = ""
    ingest_mode: str = "messages"
    dialogue_context_policy: str = "none"
    retrieval_file: str = ""
    retrieval_sha256: str = ""
    rubric_file: str = ""
    rubric_sha256: str = ""
    answer_prompt_file: str = ""
    answer_prompt_sha256: str = ""
    sample_count: int = 0
    memory_llm_model: str = ""
    embedding_model: str = ""
    tool_env: dict[str, str] = field(default_factory=dict)

    # --- arm-specific: this difference IS the intervention ---
    final_answer_prompt: str = ""

    def __post_init__(self) -> None:
        if self.arm not in SUPPORTED_ARMS:
            raise ValueError(f"Unknown arm {self.arm!r}; known: {SUPPORTED_ARMS}")
        if self.system not in METHOD_FOR_SYSTEM:
            raise ValueError(f"Unknown memory system {self.system!r}; known: {SYSTEMS}")
        if self.ingest_mode not in INGEST_MODES:
            raise ValueError(
                f"Unknown ingest mode {self.ingest_mode!r}; known: {INGEST_MODES}"
            )

    @property
    def method(self) -> str:
        return METHOD_FOR_SYSTEM[self.system]

    @property
    def task(self) -> str:
        return f"{'memtrap' if self.dataset == 'memtrapbench' else 'persist'}_{self.subset}"

    @property
    def spec(self) -> task_registry.TaskSpec:
        return task_registry.spec(self.task)

    @property
    def retrieval_path(self) -> Path:
        return paths.retrieval_dir(self.dataset, self.subset, self.system) / (
            self.retrieval_file or "retrieved.jsonl"
        )

    @property
    def output_dir(self) -> Path:
        return paths.arm_dir(
            self.dataset, self.subset, self.system, self.arm, self.run_index
        )

    # -- arm identity ---------------------------------------------------

    #: Fields compared when asserting the two arms are the same experiment.
    SHARED_FIELDS = (
        "dataset",
        "subset",
        "system",
        "run_index",
        "generation_model",
        "generation_base_url",
        "temperature",
        "max_tokens",
        "thinking_enabled",
        "judge_model",
        "judge_base_url",
        "judge_temperature",
        "judge_max_tokens",
        "judge_prompt_mode",
        "k_generations",
        "top_k",
        "top_k_policy",
        "ingest_mode",
        "dialogue_context_policy",
        "retrieval_file",
        "retrieval_sha256",
        "rubric_file",
        "rubric_sha256",
        "sample_count",
        "memory_llm_model",
        "embedding_model",
        "tool_env",
    )

    def shared(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.SHARED_FIELDS}

    def shared_fingerprint(self) -> str:
        return shared_fingerprint_of(self.shared())

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["method"] = self.method
        record["task"] = self.task
        record["shared_fingerprint"] = self.shared_fingerprint()
        record["run_fingerprint"] = run_fingerprint(self)
        record["retrieval_path"] = str(self.retrieval_path)
        record["output_dir"] = str(self.output_dir)
        return record

    def write(self, directory: str | Path | None = None) -> Path:
        target = Path(directory) if directory is not None else self.output_dir
        target.mkdir(parents=True, exist_ok=True)
        path = target / "run_config.json"
        path.write_text(
            json.dumps(self.to_record(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path


def shared_fingerprint_of(shared: dict[str, Any]) -> str:
    """``RunConfig.shared_fingerprint`` for a shared-field mapping, stored or live.

    Exposed as a free function so a validator can recompute the fingerprint from a
    ``run_config.json`` on disk: the stored digest proves nothing about the fields
    stored beside it unless somebody re-hashes them, and a hand-edited config is
    exactly the case where the two disagree. ``default=str`` is what lets a
    ``Path``-typed field and its recorded string hash to the same value.
    """
    return sha256_text(
        json.dumps(shared, sort_keys=True, ensure_ascii=False, default=str)
    )


def run_fingerprint(config: RunConfig) -> str:
    """Stable hash of everything that changes a run's semantics.

    Written into every record and into the stage cache so a cached stage cannot
    be replayed under a different dataset, system or retrieval file. Note this
    deliberately does NOT include the arm: the cache is per-output-directory and
    directories are already per-arm, while including the arm would make an
    otherwise-identical re-run look like a miss.
    """
    payload = {
        "dataset": config.dataset,
        "subset": config.subset,
        "system": config.system,
        "run_index": config.run_index,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "thinking_enabled": config.thinking_enabled,
        "top_k": config.top_k,
        "top_k_policy": config.top_k_policy,
        "ingest_mode": config.ingest_mode,
        "dialogue_context_policy": config.dialogue_context_policy,
        "retrieval_sha256": config.retrieval_sha256,
        "rubric_sha256": config.rubric_sha256,
        "generation_model": config.generation_model,
        "judge_model": config.judge_model,
    }
    return sha256_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    )


# --- verification ----------------------------------------------------------


def assert_protocol(config: RunConfig) -> None:
    """Fail loudly if the live environment no longer matches the frozen config."""
    live = {
        "temperature": resolved_temperature(),
        "max_tokens": resolved_max_tokens(),
        "thinking_enabled": resolved_thinking(),
        "generation_model": resolved_generation_model(),
        "generation_base_url": generation_base_url(),
    }
    drift = {
        key: (getattr(config, key), value)
        for key, value in live.items()
        if getattr(config, key) != value
    }
    if drift:
        lines = ", ".join(f"{k}: config={a!r} live={b!r}" for k, (a, b) in drift.items())
        raise ProtocolDrift(f"Protocol drift for {config.dataset}/{config.subset}/{config.system}: {lines}")


def assert_arm_identity(left: RunConfig, right: RunConfig) -> None:
    """The single hard requirement: the two arms are the same experiment."""
    if {left.arm, right.arm} != set(ARMS):
        raise ValueError(
            f"assert_arm_identity expects one {ARMS[0]} and one {ARMS[1]} arm, "
            f"got {left.arm!r} and {right.arm!r}"
        )
    a, b = left.shared(), right.shared()
    diffs: list[str] = []
    for name in left.SHARED_FIELDS:
        if a[name] != b[name]:
            diffs.append(f"  {name}: {a[name]!r} != {b[name]!r}")
    if diffs:
        raise AssertionError(
            "Arms differ on shared protocol fields:\n" + "\n".join(diffs)
        )
    if left.final_answer_prompt == right.final_answer_prompt:
        raise AssertionError(
            "Both arms use the same final-answer prompt; the intervention is absent "
            "(baseline must use the dataset's official answer prompt, MemAdapter must "
            "use prompts/03_memory_use_guided_generation.txt)."
        )
    if not left.final_answer_prompt or not right.final_answer_prompt:
        raise AssertionError("An arm is missing its final-answer prompt.")


def config_for_external_arm(
    *,
    retrieval_file: str | Path,
    system: str,
    arm: str,
    sample_count: int,
    run_index: int = 0,
    output_dir: str | Path | None = None,
) -> RunConfig:
    """Build, verify and write the run config for one external cell.

    Both arms call this, which is the point: the hard requirement is that the two
    arms are the same experiment, and two implementations of "derive the shared
    fields from the retrieval file" agreeing today is not the same as one
    implementation that cannot disagree. Everything arm-independent is read from the
    frozen retrieval file -- the file itself, the ingest mode, the memory model, the
    embedder -- so a field can only differ between arms if the file differs, and the
    file is a single frozen artefact.

    ``top_k`` is deliberately not passed to :func:`make_config`: it defaults to 10
    for MemTrapBench and to None ("per_sample_len_memories") for PersistBench, where
    each row's depth is its own memory-set size. Copying row 0's value would record
    one sample's depth as the whole cell's.

    The retrieval file must be the canonical one for the cell, and the config is
    written into the cell's canonical directory. Both restrictions exist so that
    ``retrieval_sha256`` describes the bytes actually read and so that a repeat
    cannot write its records into another repeat's stage cache.
    """
    rows = _read_jsonl_raw(Path(retrieval_file))
    if not rows:
        raise SystemExit(f"No retrieval rows found: {retrieval_file}")
    tasks = sorted({str(row.get("task") or "") for row in rows})
    if len(tasks) != 1:
        raise SystemExit(
            f"Retrieval file must hold exactly one task, found {len(tasks)}: {tasks[:5]}"
        )
    spec = task_registry.spec(tasks[0])

    # A targeted empty-retrieval repair can yield a mixed file: unchanged rows
    # retain their primary ingest mode while repaired rows carry the explicit
    # raw-dialogue fallback provenance.  The primary mode describes the cell;
    # the per-row fallback marker remains in the retrieval record and its audit
    # manifest.  Choosing row zero would make a repaired first row redefine the
    # protocol for all untouched rows.
    ingest_modes = {
        str((row.get("memory_config") or {}).get("ingest_mode") or "messages")
        for row in rows
    }
    primary_ingest_mode = (
        "messages" if "messages" in ingest_modes else sorted(ingest_modes)[0]
    )
    memory_config = next(
        ((row.get("memory_config") or {}) for row in rows
         if str((row.get("memory_config") or {}).get("ingest_mode") or "messages") == primary_ingest_mode),
        rows[0].get("memory_config") or {},
    )
    config = make_config(
        dataset=spec.dataset,
        subset=spec.subset,
        system=system,
        arm=arm,
        sample_count=sample_count,
        run_index=run_index,
        ingest_mode=primary_ingest_mode,
        memory_llm_model=str(memory_config.get("llm_model") or ""),
        embedding_model=str(memory_config.get("embedding_model") or ""),
        # The arm's own answer prompt, so ``answer_prompt_sha256`` describes the
        # bytes that produced the final answer on either arm (see
        # :func:`final_answer_prompt_path`). ``final_answer_prompt`` remains the
        # label, and stays out of SHARED_FIELDS because it is the intervention.
        answer_prompt_file=str(final_answer_prompt_path(arm, spec.dataset)),
        final_answer_prompt=FINAL_ANSWER_PROMPTS[arm][spec.dataset],
    )

    given = Path(retrieval_file).resolve()
    if given != config.retrieval_path.resolve():
        raise SystemExit(
            f"retrieval file must be the canonical file for this cell.\n"
            f"  given:     {given}\n"
            f"  canonical: {config.retrieval_path.resolve()}"
        )
    target = Path(output_dir).resolve() if output_dir is not None else config.output_dir.resolve()
    if target != config.output_dir.resolve():
        raise SystemExit(
            f"output directory does not match this cell's canonical directory.\n"
            f"  given:     {target}\n"
            f"  canonical: {config.output_dir.resolve()}\n"
            f"The directory encodes dataset/subset/system/arm/run-index; writing "
            f"elsewhere would mix repeats."
        )

    # Refuse to reuse a directory that was written under a different protocol: a
    # resumed run must not silently mix two temperatures into one cell.
    stored = target / "run_config.json"
    if stored.is_file():
        try:
            assert_config_matches_record(config, json.loads(stored.read_text(encoding="utf-8")))
        except ProtocolDrift as exc:
            stored_record = json.loads(stored.read_text(encoding="utf-8"))
            old_shared = {name: stored_record.get(name) for name in config.SHARED_FIELDS}
            new_shared = config.shared()
            drift = {name for name in config.SHARED_FIELDS if old_shared.get(name) != new_shared.get(name)}
            if os.environ.get("EXTERNAL_ALLOW_RETRIEVAL_CONTINUATION") == "1" and drift == {"retrieval_sha256"}:
                continuation = target / f"retrieval_continuation_{config.retrieval_sha256[:16]}.json"
                continuation.write_text(json.dumps({
                    "reason": "targeted_empty_retrieval_fallback",
                    "stored_retrieval_sha256": old_shared["retrieval_sha256"],
                    "current_retrieval_sha256": config.retrieval_sha256,
                    "unchanged_shared_fields": [name for name in config.SHARED_FIELDS if name != "retrieval_sha256"],
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                raise SystemExit(str(exc)) from exc
    # Preserve the original run_config as the primary protocol record on an
    # audited continuation; the continuation file above links it to the new
    # retrieval hash instead of silently rewriting history.
    if not stored.is_file():
        config.write(target)
    return config


def _read_jsonl_raw(path: Path) -> list[dict[str, Any]]:
    """Minimal JSONL reader, to keep this module free of heavier imports."""
    if not path.is_file():
        raise SystemExit(f"Retrieval file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def assert_config_matches_record(config: RunConfig, record: dict[str, Any]) -> None:
    """Check a run_config.json found on disk still describes this run."""
    stored = record.get("shared_fingerprint")
    if stored != config.shared_fingerprint():
        raise ProtocolDrift(
            f"Stored run_config for {config.dataset}/{config.subset}/"
            f"{config.system}/{config.arm} does not match the current protocol "
            f"(stored={stored}, current={config.shared_fingerprint()}). "
            "Use a fresh output directory rather than reusing this one."
        )


# --- reporting -------------------------------------------------------------


def dry_run_matrix(
    selected: list[RunConfig] | None = None,
) -> dict[str, Any]:
    """Print-ready call matrix: rows, stages per row, repeats, and totals."""
    if selected is None:
        selected = default_matrix()
    rows: list[dict[str, Any]] = []
    per_arm = {"baseline": 0, "memadapter": 0}
    for config in selected:
        spec = config.spec
        stages = 1 if config.arm == "baseline" else 3
        calls = config.sample_count * config.k_generations
        gen_calls = calls * stages
        judge_calls = calls
        per_arm[config.arm] += gen_calls + judge_calls
        rows.append(
            {
                "dataset": config.dataset,
                "subset": config.subset,
                "system": config.system,
                "arm": config.arm,
                "samples": config.sample_count,
                "k_generations": config.k_generations,
                "stages_per_sample": stages,
                "generation_calls": gen_calls,
                "judge_calls": judge_calls,
                "total_calls": gen_calls + judge_calls,
                "metric": spec.metric,
            }
        )
    return {
        "rows": rows,
        "totals": {
            "cells": len(rows),
            "generation_calls": sum(r["generation_calls"] for r in rows),
            "judge_calls": sum(r["judge_calls"] for r in rows),
            "total_calls": sum(r["total_calls"] for r in rows),
            "by_arm": per_arm,
        },
    }


def default_matrix(dataset: str | None = None) -> list[RunConfig]:
    """The full grid, used by ``--dry-run`` before any sample is touched."""
    from . import dataset_adapters

    configs: list[RunConfig] = []
    for task, spec in task_registry.EXTERNAL_TASKS.items():
        if dataset is not None and spec.dataset != dataset:
            continue
        sample_count = dataset_adapters.sample_count(task)
        for system in SYSTEMS:
            for arm in ARMS:
                configs.append(
                    make_config(
                        dataset=spec.dataset,
                        subset=spec.subset,
                        system=system,
                        arm=arm,
                        sample_count=sample_count,
                    )
                )
    return configs


def make_config(
    *,
    dataset: str,
    subset: str,
    system: str,
    arm: str,
    sample_count: int,
    run_index: int = 0,
    top_k: int | None = None,
    ingest_mode: str = "messages",
    judge_prompt_mode: str = "",
    memory_llm_model: str = "",
    embedding_model: str = "",
    answer_prompt_file: str = "",
    final_answer_prompt: str = "",
    tool_env: dict[str, str] | None = None,
) -> RunConfig:
    """Assemble one cell, resolving every arm-independent field from the env."""
    spec = task_registry.spec(f"{'memtrap' if dataset == 'memtrapbench' else 'persist'}_{subset}")
    if top_k is None:
        top_k = 10 if spec.is_memtrap else None
    if not judge_prompt_mode:
        # MemTrapBench's official judge prompt is a single user message; PersistBench
        # sends the rubric as a system message and memories+query+response as the user
        # message (src/benchmark/execution/judgment.py:327-339). Recorded because it
        # changes what the judge sees and is easy to get silently wrong.
        judge_prompt_mode = "single_user" if spec.is_memtrap else "system_plus_user"
    retrieval_file = "retrieved_top10.jsonl" if spec.is_memtrap else "retrieved_topk.jsonl"
    rubric_path = paths.RUBRICS_ROOT / spec.rubric_dir / spec.rubric_file
    answer_path = (
        Path(answer_prompt_file)
        if answer_prompt_file
        else paths.RUBRICS_ROOT
        / ("memtrap/prompt_user_mem.txt" if spec.is_memtrap else "persist/generator.txt")
    )
    retrieval_path = paths.retrieval_dir(dataset, subset, system) / retrieval_file
    retrieval_sha = ""
    if retrieval_path.is_file():
        retrieval_sha = sha256_file(retrieval_path)
    answer_sha = sha256_file(answer_path) if answer_path.is_file() else ""
    return RunConfig(
        dataset=dataset,
        subset=subset,
        system=system,
        arm=arm,
        run_index=run_index,
        generation_model=resolved_generation_model(),
        generation_base_url=generation_base_url(),
        temperature=resolved_temperature(),
        max_tokens=resolved_max_tokens(),
        thinking_enabled=resolved_thinking(),
        judge_model=resolved_generation_model(),
        judge_base_url=generation_base_url(),
        judge_temperature=0.0,
        judge_max_tokens=2000,
        judge_prompt_mode=judge_prompt_mode,
        k_generations=spec.k_generations,
        top_k=top_k,
        top_k_policy="fixed" if spec.is_memtrap else "per_sample_len_memories",
        ingest_mode=ingest_mode,
        dialogue_context_policy="none",
        retrieval_file=retrieval_file,
        retrieval_sha256=retrieval_sha,
        rubric_file=spec.rubric_file,
        rubric_sha256=sha256_file(rubric_path) if rubric_path.is_file() else "",
        answer_prompt_file=str(answer_path),
        answer_prompt_sha256=answer_sha,
        sample_count=sample_count,
        memory_llm_model=memory_llm_model,
        embedding_model=embedding_model,
        final_answer_prompt=final_answer_prompt,
        tool_env=dict(tool_env or {}),
    )
