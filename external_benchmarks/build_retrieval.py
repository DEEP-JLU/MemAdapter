"""Freeze the retrieved-memory condition for the external benchmarks.

One JSONL per ``(dataset, subset, memory system)``. Both method arms then read
that same file, so their retrieved-memory condition is byte-identical by
construction rather than by convention -- there is no way for the two arms to
drift apart on what memory they were given.

Two properties the run depends on:

**Append, per sample.** The native runners rewrite the whole output file after
every sample, which is quadratic; on this corpus that would be ~1,050 rewrites of
a growing multi-hundred-megabyte file. Rows are appended instead, flushed per
sample, so a killed process loses at most one sample.

**Shardable.** A single process is limited to one live memory store
(``BASELINE_OPT_MEMORY_CACHE_MAX_ENTRIES=1``), and on Windows the per-sample
Chroma/Qdrant handles are what forced sharding in the previous full run. Each
shard is an independent process writing its own part file; a supervisor can
restart one shard without touching the others, and ``merge`` refuses to produce
the final file until every expected sample is present exactly once.

Retrieval is driven through ``build_cached_baseline_context``, the caching path,
rather than the non-caching one: only the caching path calls ``save_memory()``,
and without it A-MEM's store cannot be reloaded, so an interrupted run would
have to rebuild every memory from scratch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from . import dataset_adapters as adapters
from . import paths, run_config, task_registry
from .memory_ingest import IngestPlan
from .paths import PACKAGE_ROOT

#: ``memory_id`` prefix per system, matching the frozen native retrieval files.
MEMORY_ID_PREFIX = {
    "AMEM": "amem", "Mem0": "memzero", "naiveRAG": "naiverag",
    "MemoryBank": "memorybank", "LightMem": "lightmem",
}

BUNDLED_SYSTEMS = frozenset({"MemoryBank", "LightMem"})

#: Where the vendored per-method JSON configs live, relative to ``benchmark/``.
VENDOR_CONFIG_DIR = Path("baselines") / "toolkit" / "vendor" / "configs"

#: Local (on-device) embedder provider name, which is spelled differently by the
#: two code paths that consume it. A-MEM validates against
#: ``Literal["sentence-transformers", "openai"]`` in ``layers/amem.py``, while the
#: mem0-based layers resolve ``huggingface`` in ``embeddings/factory.py``. Using
#: one name for both silently leaves the other on the OpenAI embedder, which then
#: posts to whatever ``OPENAI_BASE_URL`` happens to hold.
LOCAL_EMBEDDER_PROVIDER = {
    "A-MEM": "sentence-transformers",
    "MemZero": "huggingface",
    "NaiveRAG": "huggingface",
}

#: Embedding model used when neither the env nor the config names one.
DEFAULT_EMBEDDING_REPO = "BAAI/bge-m3"


class RetrievalIncomplete(RuntimeError):
    """Raised when a merge is attempted before every sample has been retrieved."""


def write_json_atomic(path: Path, value: Any) -> None:
    """Write a small provenance record without leaving a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


# --- paths ----------------------------------------------------------------


def output_name(spec: task_registry.TaskSpec) -> str:
    return "retrieved_top10.jsonl" if spec.is_memtrap else "retrieved_topk.jsonl"


def part_dir(dataset: str, subset: str, system: str) -> Path:
    return paths.retrieval_dir(dataset, subset, system) / "_parts"


def part_path(dataset: str, subset: str, system: str, shard: int) -> Path:
    return part_dir(dataset, subset, system) / f"shard{shard:02d}.jsonl"


def failure_path(dataset: str, subset: str, system: str, shard: int) -> Path:
    """Per-shard failure ledger, safe for parallel writers in other shards."""
    return part_dir(dataset, subset, system) / f"shard{shard:02d}.failures.jsonl"


def merged_path(dataset: str, subset: str, system: str) -> Path:
    spec = task_registry.spec(f"{'memtrap' if dataset == 'memtrapbench' else 'persist'}_{subset}")
    return paths.retrieval_dir(dataset, subset, system) / output_name(spec)


# --- config ---------------------------------------------------------------


def resolve_embedding_model() -> str:
    """Return an absolute local path to the embedding model, and export it.

    The stock vendor configs select the OpenAI embedder and name a model id
    (``text-embedding-3-small``), and the outer configs name ``baai/bge-m3``.
    Neither can be embedded here: the DeepSeek endpoint has no ``/embeddings``
    route at all (404), and the bge-m3 weights are not in the local HuggingFace
    cache by default, so a remote call fails and a local load finds no weights.

    The resolved path is written back to ``MEMORY_EMBEDDING_MODEL`` because
    ``_build_toolkit_entry`` overwrites the layer's ``retriever_name_or_path``
    from that variable -- so setting it here is what actually reaches the
    embedder.
    """
    configured = os.environ.get("MEMORY_EMBEDDING_MODEL", "").strip()
    if configured and Path(configured).exists() and Path(configured).is_dir():
        return configured

    repo = configured or DEFAULT_EMBEDDING_REPO
    try:
        from .prepare_embedder import ensure_local_weights
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "huggingface_hub is required to resolve the local embedding model"
        ) from exc
    try:
        resolved = str(ensure_local_weights(repo))
    except Exception as exc:
        raise RuntimeError(
            f"Embedding model {repo!r} is not usable locally ({type(exc).__name__}: {exc}). "
            f"Download it first, e.g. snapshot_download({repo!r}), or set "
            f"MEMORY_EMBEDDING_MODEL to an existing directory."
        ) from exc
    os.environ["MEMORY_EMBEDDING_MODEL"] = resolved
    return resolved


def materialize_vendor_config(method: str) -> Path:
    """Write a per-method vendor config that uses the on-device embedder.

    Derived from the stock config rather than hand-copied, so the only difference
    is the embedder provider and any future upstream change still flows through.
    ``config_loader`` exposes no way to set the provider, and editing the stock
    files in place would change behaviour for the five native tasks too, so the
    overlay is written here and referenced by absolute path.
    """
    stock = paths.ROOT / "benchmark" / VENDOR_CONFIG_DIR / f"{method}.json"
    if not stock.is_file():
        raise FileNotFoundError(f"Stock vendor config missing: {stock}")
    config = json.loads(stock.read_text(encoding="utf-8"))

    wanted = LOCAL_EMBEDDER_PROVIDER[method]
    stock_provider = config.get("embedder_provider")
    config["embedder_provider"] = wanted
    # A-MEM and MemZero ship 1536 dims for text-embedding-3-small; the local model
    # is 1024. ``_build_toolkit_entry`` overwrites this from the env, but leaving a
    # contradictory value in the file invites a silent mismatch if that ever stops.
    if "embedding_model_dims" in config:
        config["embedding_model_dims"] = 1024
    # Default to CPU, overriding MemZero's shipped "cuda".  This remains the
    # desktop-safe setting; a multi-GPU server may opt in with
    # MEMORY_EMBEDDER_DEVICE=cuda and pin workers via CUDA_VISIBLE_DEVICES.
    # The original desktop rationale is:
    #
    #   * Concurrency here is process count (``run_shard`` builds stores strictly
    #     serially), and every process loads its own embedder. On the GPU that is
    #     one CUDA context and one full weight copy each -- a pool of even three
    #     would not fit an 8 GB card, and the pool size is exactly the wall-clock
    #     lever for a 112.8 h serial job. CPU keeps the pool bound by host RAM
    #     instead, where the weights are mmap-shared (~1.0 GiB per extra process).
    #   * The earlier MemSyco runs pinned "cpu" for both mem0 layers already (see
    #     workspaces/deepseek/{naiverag,memzero}/configs/*_vendor.json), so this
    #     reproduces their setting rather than inventing one.
    #
    # The embedder is not the bottleneck either way: store building is dominated by
    # the layers' own LLM calls (A-MEM ~4.3 s per message, two calls per note), so
    # a faster embedder would barely move the wall clock.
    #
    # A-MEM is unaffected: its retriever constructs SentenceTransformerEmbedding-
    # Function without a device argument, which is CPU by default.
    if "use_gpu" in config:
        config["use_gpu"] = os.environ.get("MEMORY_EMBEDDER_DEVICE", "cpu").strip() or "cpu"

    target_dir = PACKAGE_ROOT / "vendor_configs"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{method}.json"
    payload = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    # Report on whether the overlay actually changed, not on whether it differs
    # from the stock file -- that comparison is true on every call by
    # construction, and would print a line per sample on a 1,050-sample run.
    if not target.is_file() or target.read_text(encoding="utf-8") != payload:
        target.write_text(payload, encoding="utf-8")
        print(
            f"[vendor-config] wrote {target.name}: embedder_provider "
            f"{stock_provider!r} -> {wanted!r}",
            flush=True,
        )
    return target


def build_eval_config(
    system: str,
    *,
    dataset: str,
    subset: str,
    top_k: int = 10,
):
    """Assemble the ``BaselineEvalConfig`` for one memory system.

    ``top_k`` here is only the fallback: PersistBench overrides the depth per
    sample, because its given memory sets range from 4 to 16 items and a fixed 10
    would truncate 259 of 500 rows.
    """
    method = run_config.METHOD_FOR_SYSTEM[system]
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    # The memory layers read MEMORY_* while generation reads DEEPSEEK_*.  They
    # must use the same DeepSeek account: merely falling back when MEMORY_API_KEY
    # is absent allows an unrelated inherited credential to bill memory building
    # elsewhere while the user watches the DEEPSEEK_API_KEY balance.  Embeddings
    # are local in this experiment, but keeping the two aliases aligned also
    # prevents an accidental future switch to a remote embedder from changing
    # accounts silently.  No credential value is logged or persisted here.
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        os.environ["MEMORY_API_KEY"] = deepseek_key
        os.environ["MEMORY_EMBEDDING_API_KEY"] = deepseek_key

    if system in BUNDLED_SYSTEMS:
        from .memory_systems import BaselineEvalConfig

        return BaselineEvalConfig(
            method=method,
            top_k=top_k,
            save_root=paths.store_dir(dataset, subset, system),
            api_key=os.environ.get("MEMORY_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=os.environ.get("MEMORY_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL"),
            llm_model=os.environ.get("MEMORY_LLM_MODEL") or run_config.resolved_generation_model(),
            embedding_model=os.environ.get("MEMORY_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_REPO,
            embedding_dims=int(os.environ.get("MEMORY_EMBEDDING_DIMS", "1024")),
            embedding_api_key=os.environ.get("MEMORY_EMBEDDING_API_KEY"),
            embedding_base_url=os.environ.get("MEMORY_EMBEDDING_BASE_URL"),
        )

    paths.ensure_benchmark_on_path()
    from baselines.config_loader import build_baseline_eval_config  # type: ignore
    resolve_embedding_model()
    config_path = paths.ROOT / "benchmark" / "baselines" / "configs" / f"{method}.json"
    return build_baseline_eval_config(
        method=method,
        baseline_config_path=config_path,
        # Redirect the vendor config to the local-embedder overlay. This is the
        # only way to change embedder_provider without editing the stock files.
        config_path=materialize_vendor_config(method),
        top_k=top_k,
        save_root=paths.store_dir(dataset, subset, system),
    )


# --- one sample -----------------------------------------------------------


def plan_for(sample: adapters.AdaptedSample) -> IngestPlan:
    return IngestPlan(
        mode=sample.ingest_mode,
        messages=sample.ingest_messages,
        top_k=sample.top_k,
        sample_id=sample.sample_id,
    )


def expected_store_digest(sample: adapters.AdaptedSample) -> str:
    """The store digest ``build_cached_baseline_context`` computes for ``sample``.

    Recomputed rather than read back, because the digest is what decides *both* which
    directory a store lives in and whether an existing one may be reused: a store whose
    marker digest disagrees with its sample is either holding another sample's memories
    or was built from a different message list, and neither is visible in the
    retrieved rows themselves. ``validate_external`` compares this against the
    completion marker on disk.

    The composition -- message keys plus the ingest identity, hashed by upstream's
    ``_sha1_parts`` -- mirrors the line in ``build_cached_baseline_context`` that
    produces it. It is spelled out here rather than factored out of there: that module
    is upstream's, and a helper added to it for our convenience would be a change to a
    frozen file. If the two ever disagree, this function's callers report every store
    as mismatched, which is a loud failure rather than a silent one.
    """
    benchmark_eval = str(paths.ROOT / "benchmark" / "evaluation")
    if benchmark_eval not in sys.path:
        sys.path.insert(0, benchmark_eval)
    from _optimized_memory import _message_key, _sha1_parts  # type: ignore

    plan = plan_for(sample)
    return _sha1_parts(
        [_message_key(dict(message)) for message in plan.messages] + [plan.identity()]
    )


def canonical_memories(system: str, retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize retrieved items to the frozen schema's five keys."""
    prefix = MEMORY_ID_PREFIX[system]
    out: list[dict[str, Any]] = []
    for index, memory in enumerate(retrieved, start=1):
        used = memory.get("used_content") or memory.get("content") or ""
        raw = memory.get("content") or used
        out.append(
            {
                "memory_id": f"{prefix}_{index:02d}",
                "content": str(used).strip(),
                "raw_content": str(raw).strip(),
                "retrieval_rank": index,
                "metadata": memory.get("metadata", {}),
            }
        )
    return out


def _marker_state(save_dir: str) -> tuple[dict[str, Any], float]:
    """Return the completion marker and the epoch it was last written to disk.

    The mtime is used rather than the ``created_at`` string inside the marker:
    that string is UTC (``time.gmtime``), and converting it back to an epoch with
    ``time.mktime`` would read it as local time -- an eight-hour error in this
    timezone, which silently labels every freshly built store as ``reused``. An
    mtime is an epoch already and carries no such ambiguity.
    """
    path = Path(save_dir) / ".memory_complete.json"
    if not path.is_file():
        return {}, 0.0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}, 0.0
    try:
        return payload, path.stat().st_mtime
    except OSError:
        return payload, 0.0


def retrieve_sample(
    sample: adapters.AdaptedSample,
    system: str,
    eval_config: Any,
    build_context,
    *,
    ingest_plan: IngestPlan | None = None,
) -> dict[str, Any]:
    """Retrieve for one sample and return the frozen-schema row."""
    plan = ingest_plan or plan_for(sample)
    started = time.time()
    if system in BUNDLED_SYSTEMS:
        from .memory_systems import build_context as build_bundled_context
        context = build_bundled_context(
            run_config.METHOD_FOR_SYSTEM[system],
            sample.prior_text,
            sample.current_request,
            eval_config,
            sample_key=sample.sample_id,
        )
    else:
        context = build_context(
            sample.prior_text,
            sample.current_request,
            eval_config,
            sample_key=sample.sample_id,
            ingest=plan,
        )
    marker, marker_mtime = _marker_state(context.save_dir)
    # ``started`` is sampled before the build, so a marker written by this call
    # always sorts at or after it while one left by an earlier run sorts strictly
    # before. No slack is allowed: a tolerance wide enough to absorb filesystem
    # timestamp granularity is also wide enough to call a second back-to-back
    # retrieval of the same sample a rebuild.
    created_at = str(marker.get("created_at") or "")
    store_status = (
        "managed_by_adapter" if system in BUNDLED_SYSTEMS
        else ("built" if marker_mtime >= started else "reused")
    )

    return {
        "sample_id": sample.sample_id,
        "task": sample.task,
        "source_file": sample.source_file,
        "row_index": sample.row_index,
        "current_request": sample.current_request,
        "memory_system": run_config.METHOD_FOR_SYSTEM[system],
        "memory_config": {
            "method": context.method,
            "top_k": context.top_k,
            "user_id": context.user_id,
            "save_dir": context.save_dir,
            "llm_model": os.environ.get("MEMORY_LLM_MODEL"),
            "embedding_model": os.environ.get("MEMORY_EMBEDDING_MODEL"),
            "embedding_dims": os.environ.get("MEMORY_EMBEDDING_DIMS"),
            "baseline_config_path": str(
                (PACKAGE_ROOT / "memory_systems" / "configs" / f"{run_config.METHOD_FOR_SYSTEM[system]}.json")
                if system in BUNDLED_SYSTEMS else
                (paths.ROOT / "benchmark" / "baselines" / "configs" / f"{run_config.METHOD_FOR_SYSTEM[system]}.json")
            ),
            # Both paths are recorded: the stock file is what the repo ships and
            # what a reader would look at first, but it is *not* what ran. Only
            # the overlay carries the local-embedder provider.
            "vendor_config_path": None if system in BUNDLED_SYSTEMS else str(
                paths.ROOT / "benchmark" / VENDOR_CONFIG_DIR
                / f"{run_config.METHOD_FOR_SYSTEM[system]}.json"
            ),
            "vendor_config_used": None if system in BUNDLED_SYSTEMS else str(
                materialize_vendor_config(run_config.METHOD_FOR_SYSTEM[system])
            ),
            # --- provenance added by this harness ---
            "ingest_mode": plan.mode,
            "ingest_identity": plan.identity(),
            "prior_sha256": sample.prior_sha256(),
            "messages_count": sample.messages_count,
            "store_status": store_status,
            "marker_written_at": created_at,
            "marker_messages_count": marker.get("messages_count"),
            "marker_digest": marker.get("digest"),
            "top_k_policy": "fixed" if sample.top_k is None else "per_sample_len_memories",
            "sample_meta": sample.meta,
        },
        "official_context_text": context.context_text,
        "retrieved_memories": canonical_memories(system, context.retrieved_memories),
        "retrieved_count": len(context.retrieved_memories),
        "benchmark_row": sample.benchmark_row,
    }


# --- sharding -------------------------------------------------------------


def shard_assignment(total: int, shard_count: int, shard_index: int) -> list[int]:
    """Contiguous 1-based row numbers for one shard.

    Contiguous rather than strided so that a merged file, before sorting, is
    already close to row order; it also keeps each shard's work localized to one
    region of the source file.
    """
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index {shard_index} out of range for {shard_count} shards")
    size = (total + shard_count - 1) // shard_count
    lo = shard_index * size + 1
    hi = min(total, lo + size - 1)
    return list(range(lo, hi + 1))


def _read_done(path: Path) -> dict[str, int]:
    """``sample_id -> row_index`` already written by a previous attempt."""
    done: dict[str, int] = {}
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A torn final line from a killed process: drop it by ignoring,
                # and the sample is simply redone.
                continue
            done[str(row.get("sample_id"))] = int(row.get("row_index") or 0)
    return done


def _read_group_done(directory: Path) -> dict[str, int]:
    """Return successes across every shard of one task/system group.

    A partial run may have been created with a different shard count and then
    resumed with a new count.  Checking only the current shard would assign a
    formerly completed sample to a different shard and pay to rebuild it.
    """
    done: dict[str, int] = {}
    for candidate in directory.glob("shard*.jsonl"):
        # ``shardNN.failures.jsonl`` has the same extension and carries the
        # original sample ID for diagnostics.  It is not a completed retrieval
        # and must never suppress a later retry.
        if candidate.name.endswith(".failures.jsonl"):
            continue
        done.update(_read_done(candidate))
    return done


def run_shard(
    *,
    task: str,
    system: str,
    shard_index: int,
    shard_count: int,
    limit: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    spec = task_registry.spec(task)
    total = adapters.sample_count(task)
    numbers = shard_assignment(total, shard_count, shard_index)
    if limit is not None:
        numbers = numbers[:limit]

    path = part_path(spec.dataset, spec.subset, system, shard_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use group-wide success state rather than only this shard.  This preserves
    # resume correctness when a pilot's shard layout differs from the formal
    # run's layout.
    done = _read_group_done(path.parent)
    todo = numbers
    if done:
        todo = [n for n in numbers if n not in _ids_for(task, numbers, done)]

    if not todo:
        return {
            "task": task, "system": system, "shard": shard_index,
            "assigned": len(numbers), "retrieved": 0, "skipped": len(numbers),
        }

    build_context = None if system in BUNDLED_SYSTEMS else _load_cached_builder()
    # top_k here is a fallback only; PersistBench overrides it per sample.
    eval_config = build_eval_config(
        system, dataset=spec.dataset, subset=spec.subset, top_k=10
    )

    samples = {s.row_index: s for s in adapters.load_samples(task, indices=todo)}
    written = 0
    failures: list[dict[str, Any]] = []
    failed_path = failure_path(spec.dataset, spec.subset, system, shard_index)
    with path.open("a", encoding="utf-8") as handle:
        for number in todo:
            sample = samples[number]
            try:
                row = retrieve_sample(sample, system, eval_config, build_context)
            except Exception as exc:
                # A bad sample must not strand the later rows in its shard.  The
                # durable success file stays the source of truth for resume, so a
                # subsequent bounded shard retry selects only this missing sample.
                failure = {
                    "task": task,
                    "system": system,
                    "shard": shard_index,
                    "row_index": number,
                    "sample_id": sample.sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                with failed_path.open("a", encoding="utf-8") as failed_handle:
                    failed_handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    failed_handle.flush()
                    os.fsync(failed_handle.fileno())
                print(
                    f"[retrieve] FAILED {task}/{system} shard{shard_index} "
                    f"row={number} {sample.sample_id}: {type(exc).__name__}; continuing",
                    flush=True,
                )
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written += 1
            if verbose or written % 25 == 0:
                print(
                    f"[retrieve] {task}/{system} shard{shard_index} "
                    f"{written}/{len(todo)} row={number} {sample.sample_id} "
                    f"n={row['retrieved_count']} store={row['memory_config']['store_status']}",
                    flush=True,
                )
    result = {
        "task": task, "system": system, "shard": shard_index,
        "assigned": len(numbers), "retrieved": written,
        "skipped": len(numbers) - len(todo), "failed": len(failures),
    }
    if failures:
        # The launcher's existing three-attempt policy now retries only rows absent
        # from the success file, after every later row has already been attempted.
        raise RuntimeError(
            f"{task}/{system} shard{shard_index}: {len(failures)} sample(s) failed "
            f"after later rows continued; details: {failed_path}"
        )
    return result


def _ids_for(task: str, numbers: list[int], done: dict[str, int]) -> set[int]:
    """Row indices among ``numbers`` whose sample_id is already on disk."""
    samples = adapters.load_samples(task, indices=numbers)
    return {s.row_index for s in samples if s.sample_id in done}


def _load_cached_builder():
    """Import and return ``build_cached_baseline_context``.

    Imported late: the module inserts the baselines package onto ``sys.path`` at
    import time, which must not happen before the process sets its environment.
    """
    benchmark_eval = str(paths.ROOT / "benchmark" / "evaluation")
    if benchmark_eval not in sys.path:
        sys.path.insert(0, benchmark_eval)
    from _optimized_memory import build_cached_baseline_context  # type: ignore

    return build_cached_baseline_context


# --- merge ----------------------------------------------------------------


def merge_shards(
    *,
    task: str,
    system: str,
    shard_count: int,
    require_complete: bool = True,
) -> dict[str, Any]:
    spec = task_registry.spec(task)
    expected = adapters.sample_count(task)
    rows: dict[str, tuple[tuple[float, int, int], dict[str, Any]]] = {}
    duplicates: list[dict[str, Any]] = []

    for shard in range(shard_count):
        path = part_path(spec.dataset, spec.subset, system, shard)
        if not path.is_file():
            # A shard can legitimately have no work after a resumed run: its
            # assigned samples may already be present in another shard part
            # from an earlier shard layout.  Completeness is determined below
            # from the union of sample IDs, not from the presence of an empty
            # shard file.
            continue
        try:
            file_mtime = path.stat().st_mtime
        except OSError:
            file_mtime = 0.0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = str(row["sample_id"])
                priority = (file_mtime, shard, line_number)
                if sid in rows:
                    previous_priority, _ = rows[sid]
                    duplicates.append({
                        "sample_id": sid,
                        "kept": "current" if priority > previous_priority else "previous",
                        "previous_shard": previous_priority[1],
                        "current_shard": shard,
                    })
                    if priority <= previous_priority:
                        continue
                rows[sid] = (priority, row)

    ordered = sorted((row for _, row in rows.values()), key=lambda r: int(r["row_index"]))
    if require_complete and len(ordered) != expected:
        missing = expected - len(ordered)
        raise RetrievalIncomplete(
            f"{task}/{system}: merged {len(ordered)} rows but the dataset has {expected} "
            f"({missing} missing); re-run the incomplete shards before merging"
        )

    target = merged_path(spec.dataset, spec.subset, system)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(target)
    audit_path = target.with_name("merge_duplicates_audit.json")
    write_json_atomic(audit_path, {
        "task": task,
        "system": system,
        "duplicate_records": len(duplicates),
        "policy": "newest part-file mtime, then shard index, then line number",
        "examples": duplicates[:20],
    })
    return {
        "task": task, "system": system,
        "rows": len(ordered), "expected": expected,
        "path": str(target), "sha256": run_config.sha256_file(target),
        "duplicate_records": len(duplicates),
        "duplicate_audit": str(audit_path),
    }


# --- cli ------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="External task name, e.g. memtrap_poison")
    parser.add_argument("--system", required=True, choices=tuple(run_config.SYSTEMS))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Cap rows (pilot)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--merge", action="store_true", help="Merge shard parts instead of retrieving")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="With --merge, write whatever exists (pilot only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Before anything reads the environment. The pins are load-bearing -- without
    # DEEPSEEK_BASE_URL the memory layers still resolve, and without the model id
    # they fail on first call -- so a missing file is an error, not a warning.
    run_config.load_env_file()
    protocol = run_config.export_protocol_environment()
    args = _parse_args(argv)
    spec = task_registry.spec(args.task)
    print(
        "[protocol] " + json.dumps(dict(sorted(protocol.items())), ensure_ascii=False),
        flush=True,
    )

    if args.merge:
        result = merge_shards(
            task=args.task, system=args.system, shard_count=args.shard_count,
            require_complete=not args.allow_partial,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    result = run_shard(
        task=args.task, system=args.system,
        shard_index=args.shard_index, shard_count=args.shard_count,
        limit=args.limit, verbose=args.verbose,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
