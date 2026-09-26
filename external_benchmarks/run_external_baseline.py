"""Generate the baseline arm of the external benchmarks.

The baseline arm is the control: retrieved memories plus the query, answered with
the dataset's *own* answer prompt and nothing else. It exists so the MemAdapter
arm has something to be compared against under a byte-identical condition -- same
frozen retrieval file, same model, same temperature, same ``max_tokens``, same
sample set, same generation count. Every one of those is therefore read from the
same environment variables the MemAdapter runner reads, and this module must never
"helpfully" override one: not passing ``max_tokens`` is what keeps both arms on
``MEMADAPTER_MAX_TOKENS``.

What differs between the arms is exactly one thing, and it is the intervention:
the final answer prompt. The baseline arm uses the dataset's official prompt
(``prompt_user_mem.txt`` for MemTrapBench, ``generator.txt`` for PersistBench);
the MemAdapter arm uses stage three of its own method. That difference is recorded
in ``run_config.json`` (``final_answer_prompt``) and belongs in the paper's method
section.

Two prompt-protocol details worth knowing, both matching upstream:

*   MemTrapBench's official runner sends a **single user message** -- no system
    message at all (``runners/eval/eval_common.py``: ``messages=[{"role": "user",
    ...}]``). ``ModelClient`` always sends system + user, and the MemAdapter arm
    necessarily does too, so the baseline arm sends an empty system message. That
    is a deliberate deviation from upstream, applied identically to both arms, so
    the arm comparison stays valid while absolute numbers may differ from the
    paper. It is recorded as ``answer_prompt_mode`` in every record.
*   PersistBench sends the template as the **system** prompt and the raw query as
    the user message (``execution/generation.py``).

Memories reach the prompt through ``format_retrieved_memories`` from
``benchmark/baselines/common.py`` -- the same function the repo's native runs use,
which renders the string each retrieval row already stores as
``official_context_text``. Both arms therefore see the same memory text.

Resumable in the same way as the MemAdapter runner: a row is appended only after
its answer is produced, failures can be recorded and retried without discarding
successful rows, and a rerun of a completed cell makes zero model calls.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from . import dataset_adapters, paths, run_config, task_registry
from .text_norm import read_prompt_file

#: Mirrors ``MemAdapter.PROMPT_VERSION`` in spirit: bumped only when the answer
#: prompt or the record schema changes, so a stale record cannot be silently reused.
BASELINE_PROMPT_VERSION = "external-baseline-20260917"

ANSWER_PROMPT_FILES = {
    "memtrapbench": "memtrap/prompt_user_mem.txt",
    "persistbench": "persist/generator.txt",
}

#: ``system_plus_user`` is what ModelClient always sends. The MemTrapBench entry
#: marks the case where upstream sends one message and we additionally send an
#: empty system turn -- see the module docstring.
ANSWER_PROMPT_MODES = {
    "memtrapbench": "empty_system_plus_user",
    "persistbench": "system_plus_user",
}


def _format_memories_for_prompt(memories: list[dict[str, Any]]) -> str:
    """Render retrieved memories exactly as the native MemSyco runs do."""
    paths.ensure_benchmark_on_path()
    from common import format_retrieved_memories  # type: ignore

    return format_retrieved_memories(memories)


def answer_prompt_path(dataset: str) -> Path:
    return paths.RUBRICS_ROOT / ANSWER_PROMPT_FILES[dataset]


def build_answer_prompts(
    *,
    dataset: str,
    query: str,
    memories: list[dict[str, Any]],
    model_name: str,
    template: str,
) -> tuple[str, str]:
    """Return the ``(system, user)`` pair for the dataset's official answer prompt."""
    if dataset == "memtrapbench":
        user = template.format(
            conversation_history=_format_memories_for_prompt(memories),
            user_query=query,
        )
        return "", user
    if dataset == "persistbench":
        flat = [str(m.get("memory_text") or m.get("content") or "") for m in memories]
        system = template.replace("{model_name}", model_name).replace(
            "{memories}", dataset_adapters.formatted_memories(flat)
        )
        return system, query
    raise ValueError(f"No answer prompt protocol for dataset {dataset!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-file", type=Path, required=True)
    parser.add_argument("--system", required=True, choices=tuple(run_config.SYSTEMS))
    parser.add_argument(
        "--arm", choices=("baseline", "anti_sycophancy"), default="baseline",
        help="Generation arm. anti_sycophancy appends its frozen memory-evidence instruction.",
    )
    parser.add_argument("--model", default="DeepSeek-V4-Flash", choices=("DeepSeek-V4-Flash", "GPT-5.6-sol", "Qwen3-8B"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Defaults to the canonical per-cell directory. Passing one explicitly is "
            "allowed only if it resolves there: the directory encodes (dataset, subset, "
            "system, arm, repeat), and a mismatch would quietly write one repeat's "
            "answers into another's stage cache."
        ),
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help=(
            "Repeat index. PersistBench draws three independent generations for its "
            "two safety classes, so it is invoked once per repeat with 1..3, each into "
            "its own directory; single-generation cells use 0."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("MEMADAPTER_WORKERS", "8")),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _cell_from_rows(rows: list[dict[str, Any]]) -> tuple[str, str]:
    """Derive (dataset, subset) from the retrieval rows, refusing a mixed file."""
    tasks = sorted({str(row.get("task") or "") for row in rows})
    if len(tasks) != 1:
        raise SystemExit(
            f"Retrieval file must hold exactly one task, found {len(tasks)}: {tasks[:5]}"
        )
    spec = task_registry.spec(tasks[0])
    return spec.dataset, spec.subset


def _complete(runner, client, system_prompt: str, user_prompt: str, label: str):
    """One retried call that also hands back the call metadata.

    ``retry_call`` returns only the stripped text. The record needs the token
    counts, latency and finish reason too, and re-issuing the request to collect
    them would bill every sample twice. Capturing the result inside the closure
    keeps the retry policy itself -- the backoff and the server's ``retry_after``
    hint -- identical to the MemAdapter arm, rather than reimplementing it here
    where it could drift.
    """
    captured: list[Any] = []

    def once() -> str:
        result = client.complete_result(system_prompt, user_prompt)
        captured.append(result)
        return result.text or ""

    text = runner.retry_call(once, label=label)
    # retry_call returns only after an attempt produced non-empty text, so the last
    # capture is always the successful one.
    return captured[-1], text


def generate(args: argparse.Namespace) -> dict[str, Any]:
    runner = paths.load_memadapter()

    raw_rows = runner.read_jsonl(args.retrieval_file)
    if not raw_rows:
        raise SystemExit(f"No retrieval rows found: {args.retrieval_file}")
    dataset, subset = _cell_from_rows(raw_rows)

    # Both arms build their config through this one function, which is what makes
    # them the same experiment by construction rather than by two parallel
    # derivations agreeing. It reads every shared field from the frozen retrieval
    # file, enforces the canonical retrieval path and output directory, and refuses
    # to reuse a directory already written under a different protocol.
    config = run_config.config_for_external_arm(
        retrieval_file=args.retrieval_file,
        system=args.system,
        arm=args.arm,
        # The cell's size, not this run's selection: --limit narrows a pilot run
        # without rewriting the protocol record, so a limited pilot and the full run
        # describe the same experiment.
        sample_count=len(raw_rows),
        run_index=args.run_index,
        output_dir=args.output_dir,
    )
    output_dir = config.output_dir

    # An empty candidate list cannot support a memory-use comparison.  Some
    # external memory systems legitimately return no candidates for a query;
    # exclude only those rows and preserve the remaining cell rather than
    # rejecting the whole file during normalization.
    empty_retrieval_exclusions = [
        {
            "sample_id": str(row.get("sample_id") or row.get("id") or "<unknown>"),
            "task": str(row.get("task") or ""),
            "reason": "empty_retrieval",
        }
        for row in raw_rows
        if isinstance(row.get("retrieved_memories"), list)
        and not row.get("retrieved_memories")
    ]
    usable_raw_rows = [
        row
        for row in raw_rows
        if not (isinstance(row.get("retrieved_memories"), list) and not row.get("retrieved_memories"))
    ]
    rows = [runner.normalize_retrieval_row(row) for row in usable_raw_rows]
    if args.limit is not None:
        rows = rows[: args.limit]

    expected_ids = [row["sample_id"] for row in rows]
    if len(expected_ids) != len(set(expected_ids)):
        raise SystemExit("Retrieval file contains duplicate sample_ids")

    template = read_prompt_file(answer_prompt_path(dataset))
    if run_config.sha256_text(template) != config.answer_prompt_sha256:
        raise SystemExit(
            f"Answer prompt {answer_prompt_path(dataset)} does not hash to the value "
            f"make_config recorded ({config.answer_prompt_sha256[:16]}...); refusing to "
            f"run with a provenance record that does not describe the prompt sent."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if empty_retrieval_exclusions:
        (output_dir / "empty_retrieval_exclusions.json").write_text(
            json.dumps(
                {"reason": "empty_retrieval", "rows": empty_retrieval_exclusions},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[baseline] excluded {len(empty_retrieval_exclusions)} external rows with empty retrieval",
            flush=True,
        )

    output_file = output_dir / "outputs.jsonl"
    existing = runner.read_jsonl(output_file)
    existing_ids = [str(row.get("sample_id")) for row in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise SystemExit("Existing baseline outputs contain duplicate sample IDs")

    completed_dir = output_dir / "completed_records"
    if args.continue_on_error:
        unexpected = sorted(set(existing_ids) - set(expected_ids))
        if unexpected:
            raise SystemExit(f"Existing outputs are outside the current selection: {unexpected[:5]}")
        completed_records = runner.read_record_cache(completed_dir)
        unexpected = sorted(set(completed_records) - set(expected_ids))
        if unexpected:
            raise SystemExit(f"Completed records are outside the current selection: {unexpected[:5]}")
        overlap = sorted(set(existing_ids) & set(completed_records))
        if overlap:
            raise SystemExit(f"Samples exist in both outputs and completed_records: {overlap[:5]}")
    else:
        if existing_ids != expected_ids[: len(existing_ids)]:
            raise SystemExit("Existing baseline outputs are not an ordered retrieval prefix")
        completed_records = {}

    done_ids = set(existing_ids) | set(completed_records)
    todo = [row for row in rows if row["sample_id"] not in done_ids]
    print(
        f"[baseline] cell={dataset}/{subset}/{args.system} run={args.run_index} "
        f"rows={len(rows)} todo={len(todo)} workers={args.workers}",
        flush=True,
    )
    if not todo:
        _finalize(runner, output_file, rows, existing, completed_dir)
        return {"written": len(rows), "generated": 0, "failed": 0}

    thread_state = threading.local()

    def get_client():
        client = getattr(thread_state, "client", None)
        if client is None:
            client = runner.ModelClient(args.model)
            thread_state.client = client
        return client

    model_name = run_config.resolved_generation_model()
    failures: list[tuple[str, str]] = []
    produced = 0

    def work(row: dict[str, Any]) -> dict[str, Any]:
        system_prompt, user_prompt = build_answer_prompts(
            dataset=dataset,
            query=row["current_request"],
            memories=row["retrieved_memories"],
            model_name=model_name,
            template=template,
        )
        if args.arm == "anti_sycophancy":
            from methods.shared.prompts import ANTI_SYCOPHANCY
            system_prompt = f"{system_prompt}\n\n{ANTI_SYCOPHANCY}"
        result, _text = _complete(
            runner,
            get_client(),
            system_prompt,
            user_prompt,
            label=f"{args.model} {args.system} {args.arm}",
        )
        return {
            "sample_id": row["sample_id"],
            "task": row["task"],
            "dataset": dataset,
            "subset": subset,
            "system": args.system,
            "arm": args.arm,
            "run_index": args.run_index,
            "current_request": row["current_request"],
            "retrieved_memories": row["retrieved_memories"],
            "retrieved_count": len(row["retrieved_memories"]),
            "final_answer": (result.text or "").strip(),
            "final_answer_raw": result.text,
            "generation_model": result.model,
            "generation_base_url": result.base_url,
            "generation_temperature": result.temperature,
            "generation_max_tokens": result.max_tokens,
            "generation_input_tokens": result.input_tokens,
            "generation_output_tokens": result.output_tokens,
            "generation_reasoning_tokens": result.reasoning_tokens,
            "generation_finish_reason": result.finish_reason,
            "generation_latency_ms": result.latency_ms,
            "answer_prompt_mode": ANSWER_PROMPT_MODES[dataset],
            "answer_system_prompt": system_prompt,
            "answer_user_prompt": user_prompt,
            "answer_prompt_file": str(answer_prompt_path(dataset)),
            "answer_prompt_sha256": config.answer_prompt_sha256,
            "retrieval_file": str(args.retrieval_file),
            "retrieval_sha256": config.retrieval_sha256,
            "prompt_version": BASELINE_PROMPT_VERSION,
            "run_fingerprint": run_config.run_fingerprint(config),
            "config": config.shared(),
            "benchmark_row": row["benchmark_row"],
        }

    workers = max(1, min(args.workers, len(todo)))
    print(f"[baseline] workers={workers} todo={len(todo)}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, row): row["sample_id"] for row in todo}
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                record = future.result()
            except BaseException as exc:
                if not args.continue_on_error:
                    raise
                failures.append((sample_id, f"{type(exc).__name__}: {exc}"))
                print(f"[baseline] FAILED {sample_id}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if args.continue_on_error:
                completed_dir.mkdir(parents=True, exist_ok=True)
                runner.write_json_atomic(completed_dir / f"{sample_id}.json", record)
            else:
                runner.append_jsonl(output_file, record)
                existing.append(record)
            produced += 1

    if failures:
        print(
            f"[baseline] {len(failures)} failed samples: {[f[0] for f in failures][:5]}",
            flush=True,
        )
        (output_dir / "failures.json").write_text(
            json.dumps([{"sample_id": s, "error": e} for s, e in failures], indent=2),
            encoding="utf-8",
        )

    if args.continue_on_error:
        _finalize(runner, output_file, rows, existing, completed_dir)

    status = {"written": len(rows), "generated": produced, "failed": len(failures)}
    print(f"[baseline] done {status}", flush=True)
    return status


def _finalize(runner, output_file: Path, rows, existing, completed_dir: Path) -> None:
    """Merge completed records into the ordered output file, if the cell is complete.

    The cache is re-read here rather than passed in: the caller's copy was taken
    before this run started, so it cannot contain the records this run just wrote,
    and merging with it would report a complete cell as still-missing.
    """
    completed_records = runner.read_record_cache(completed_dir)
    complete, missing = runner.merge_record_cache(output_file, rows, existing, completed_records)
    if complete:
        # The cache means "not yet in the output file", which is what the overlap guard
        # asserts. A successful merge satisfies it, so the entries it just merged have
        # to go -- otherwise the next run of this cell, which should be a free no-op,
        # is refused for putting one sample in both places.
        runner.clear_merged_cache(completed_dir, completed_records)
        print(f"[baseline] merged {len(rows)} records into {output_file}", flush=True)
    else:
        print(
            f"[baseline] {len(missing)} samples still missing; run again to continue "
            f"(first: {missing[:3]})",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    run_config.load_env_file()
    generate(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
