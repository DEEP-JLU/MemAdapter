"""Judge both arms of an external cell with the dataset's own rubric.

One runner for both arms, because the judge is part of the *shared* protocol: a second
implementation for the MemAdapter arm would be a second chance for the arms to differ in
something that is not the intervention. Records carry the same fields on both sides --
``judge_model``, ``judge_base_url``, ``judge_temperature``, ``judge_max_tokens``,
``rubric_sha256`` -- so ``validate_external --check-arm-identity`` can pair them by
``sample_id`` and compare directly.

Per dataset:

*   **MemTrapBench** -- one generation per sample, scored on the subset's own 0-5
    dimension rubric. The dimension list is read off the rubric rather than assumed:
    ``shared`` scores four dimensions, ``poison`` and ``number_game`` two each.
*   **PersistBench** -- K generations per sample (3 for the two safety classes, 1 for
    beneficial), each scored 1-5 (``score``) or 1-3 (``rating``). Every generation is
    judged: FR@1/2/3 is computed from the per-generation scores downstream, so
    aggregating here would throw away exactly what makes those numbers meaningful.

Resumable in the same way as the arms. A record is written only after its judgement
parses *and* validates; a judgement that cannot be read is retried on the next run
rather than recorded as a zero, because a fabricated zero enters the average as if the
judge had said it. A rerun of a complete cell makes zero model calls.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import threading
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from . import paths, run_config, task_registry
from .judges import common as judge_common
from .judges import judge_memtrap, judge_persist
from .text_norm import read_prompt_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-file", type=Path, required=True)
    parser.add_argument("--system", required=True, choices=tuple(run_config.SUPPORTED_SYSTEMS))
    parser.add_argument("--arm", required=True, choices=tuple(run_config.SUPPORTED_ARMS))
    parser.add_argument(
        "--run-index",
        type=int,
        default=0,
        help=(
            "Repeat index, matching the directory the arm wrote into: PersistBench's "
            "safety classes are generated 1..3 times, everything else once (0)."
        ),
    )
    parser.add_argument("--model", default="DeepSeek-V4-Flash", choices=("DeepSeek-V4-Flash", "GPT-5.6-sol", "Qwen3-8B"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("MEMADAPTER_WORKERS", "8")),
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=None,
        help=(
            "Override the frozen judge output budget. Omit for the protocol value "
            "from the eval config; only use this to clear a sample whose judgement "
            "was truncated by the budget (a judgement that cannot be read is retried "
            "rather than scored zero, so the cell stays unmerged until it parses)."
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def rubric_path_for(spec: task_registry.TaskSpec) -> Path:
    return paths.RUBRICS_ROOT / spec.rubric_dir / spec.rubric_file


def build_prompts(
    spec: task_registry.TaskSpec, record: dict[str, Any], template: str
) -> tuple[str, str]:
    if spec.is_memtrap:
        return judge_memtrap.build_prompts(record, template)
    return judge_persist.build_prompts(record, template)


def interpret(
    spec: task_registry.TaskSpec, text: str, template: str
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Dispatch to the dataset's parser; return ``(payload, record_extra, problems)``."""
    if spec.is_memtrap:
        parsed, scores, problems = judge_memtrap.interpret(text, template)
        extra: dict[str, Any] = {
            "dimensions": parsed,
            "dimension_scores": scores,
            "dimension_count": len(scores),
            "dimension_mean": (
                sum(scores.values()) / len(scores) if scores else None
            ),
        }
        return parsed, extra, problems

    low, high = judge_persist.expected_range(spec)
    parsed, score, key_used, problems = judge_persist.interpret(text, low=low, high=high)
    extra = {
        "judge_parsed": parsed,
        "score": score,
        "score_key": key_used,
        "score_scale_low": low,
        "score_scale_high": high,
    }
    return parsed, extra, problems


def judge(args: argparse.Namespace) -> dict[str, Any]:
    runner = paths.load_memadapter()

    config = run_config.config_for_external_arm(
        retrieval_file=args.retrieval_file,
        system=args.system,
        arm=args.arm,
        sample_count=len(runner.read_jsonl(args.retrieval_file)),
        run_index=args.run_index,
    )
    spec = config.spec

    # Applied *after* the rubric/retrieval provenance checks above, so a budget
    # override can never relax the frozen-protocol validation. Recorded per record
    # through config.judge_max_tokens, so an overridden judgement is distinguishable
    # from a protocol-budget one in the artifacts.
    if args.judge_max_tokens is not None:
        config = dataclasses.replace(config, judge_max_tokens=args.judge_max_tokens)
        print(
            f"[judge] judge_max_tokens overridden to {args.judge_max_tokens} "
            f"(protocol value is normally 2000)",
            flush=True,
        )

    # The judge is part of the frozen protocol, so the rubric it sends must be the one
    # the config hashed. read_prompt_file keeps the file's own bytes, which is what
    # makes this a check on the prompt sent rather than on a normalized copy of it.
    template = read_prompt_file(rubric_path_for(spec))
    if run_config.sha256_text(template) != config.rubric_sha256:
        raise SystemExit(
            f"Rubric {rubric_path_for(spec)} does not hash to the value the run config "
            f"recorded ({config.rubric_sha256[:16]}...); refusing to judge with a "
            f"provenance record that does not describe the prompt sent."
        )

    outputs_file = config.output_dir / "outputs.jsonl"
    records = runner.read_jsonl(outputs_file)
    if not records:
        raise SystemExit(
            f"No arm outputs to judge at {outputs_file}. Run the arm first."
        )
    if args.limit is not None:
        records = records[: args.limit]

    expected_ids = [str(row.get("sample_id")) for row in records]
    if len(expected_ids) != len(set(expected_ids)):
        raise SystemExit("Arm outputs contain duplicate sample_ids")

    for row in records:
        if str(row.get("arm") or "") != args.arm:
            raise SystemExit(
                f"{outputs_file} holds records from arm {row.get('arm')!r}, but "
                f"--arm {args.arm} was requested"
            )

    judge_file = config.output_dir / "judge.jsonl"
    existing = runner.read_jsonl(judge_file)
    existing_ids = [str(row.get("sample_id")) for row in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise SystemExit("Existing judge outputs contain duplicate sample IDs")

    completed_dir = config.output_dir / "completed_judge_records"
    if args.continue_on_error:
        unexpected = sorted(set(existing_ids) - set(expected_ids))
        if unexpected:
            raise SystemExit(f"Existing judge rows are outside this selection: {unexpected[:5]}")
        completed_records = runner.read_record_cache(completed_dir)
        unexpected = sorted(set(completed_records) - set(expected_ids))
        if unexpected:
            raise SystemExit(f"Completed judge records are outside this selection: {unexpected[:5]}")
        overlap = sorted(set(existing_ids) & set(completed_records))
        if overlap:
            raise SystemExit(f"Samples exist in both judge outputs and cache: {overlap[:5]}")
    else:
        if existing_ids != expected_ids[: len(existing_ids)]:
            raise SystemExit("Existing judge outputs are not an ordered prefix")
        completed_records = {}

    done_ids = set(existing_ids) | set(completed_records)
    todo = [row for row in records if str(row.get("sample_id")) not in done_ids]
    print(
        f"[judge] cell={config.dataset}/{config.subset}/{config.system}/{args.arm} "
        f"run={args.run_index} rows={len(records)} todo={len(todo)} workers={args.workers}",
        flush=True,
    )
    if not todo:
        _finalize(runner, judge_file, records, existing, completed_dir)
        report_coverage(records, runner.read_jsonl(judge_file))
        return {"judged": len(records), "generated": 0, "failed": 0}

    # ModelClient reads its sampling settings from the environment at construction and
    # offers no per-call temperature, so the judge's 0.0 has to be in place before the
    # first client exists. max_tokens is passed per call instead, where it is explicit.
    # Set *after* the config was built, deliberately: config_for_external_arm reads
    # MEMADAPTER_TEMPERATURE too, and the shared fingerprint the arms are compared on
    # must keep recording the generation temperature, not this one.
    os.environ["MEMADAPTER_TEMPERATURE"] = str(config.judge_temperature)

    thread_state = threading.local()

    def get_client():
        client = getattr(thread_state, "client", None)
        if client is None:
            client = runner.ModelClient(args.model)
            thread_state.client = client
        return client

    failures: list[tuple[str, str]] = []
    produced = 0
    run_id = runner.new_run_id('judge')
    run_started = runner.utc_now()

    def work_payload(row: dict[str, Any], recorder) -> dict[str, Any]:
        system_prompt, user_prompt = build_prompts(spec, row, template)
        result, text = judge_common.complete_capturing(
            runner,
            get_client(),
            system_prompt,
            user_prompt,
            label=f"{args.model} judge {spec.task}",
            max_tokens=config.judge_max_tokens,
            recorder=recorder,
        )
        _parsed, extra, problems = interpret(spec, text, template)
        if problems:
            # Not recorded anywhere as a score: the sample stays missing and the next
            # run retries it. See the module docstring.
            raise ValueError("; ".join(problems))
        return {
            "sample_id": row["sample_id"],
            "task": row["task"],
            "dataset": config.dataset,
            "subset": config.subset,
            "system": config.system,
            "arm": args.arm,
            "run_index": args.run_index,
            "current_request": row.get("current_request"),
            "final_answer": row.get("final_answer"),
            "judge_model": result.model,
            "judge_base_url": result.base_url,
            "judge_temperature": result.temperature,
            "judge_max_tokens": result.max_tokens,
            "judge_input_tokens": result.input_tokens,
            "judge_output_tokens": result.output_tokens,
            "judge_reasoning_tokens": result.reasoning_tokens,
            "judge_finish_reason": result.finish_reason,
            "judge_latency_ms": result.latency_ms,
            "judge_prompt_mode": config.judge_prompt_mode,
            "judge_system_prompt": system_prompt,
            "judge_user_prompt": user_prompt,
            "judge_raw": text,
            "rubric_file": str(rubric_path_for(spec)),
            "rubric_sha256": config.rubric_sha256,
            "prompt_version": judge_common.JUDGE_PROMPT_VERSION,
            "run_fingerprint": run_config.run_fingerprint(config),
            "config": config.shared(),
            **extra,
        }

    def work(row: dict[str, Any]) -> dict[str, Any]:
        recorder = runner.EfficiencyRecorder(output_dir=config.output_dir,
            sample_id=str(row['sample_id']), task=str(row['task']), method=args.arm,
            model=args.model, memory_system=args.system, phase='judge',
            run_id=run_id, run_started_at=run_started, base_url=config.judge_base_url)
        started = time.perf_counter()
        try:
            record = work_payload(row, recorder)
            recorder.finish_stage('judge', (time.perf_counter()-started)*1000, cache_hit=False)
            record['judge_efficiency'] = recorder.finish(success=True)
            return record
        except Exception as exc:
            recorder.finish_stage('judge', (time.perf_counter()-started)*1000, cache_hit=False)
            recorder.finish(success=False, error=exc)
            raise
        finally:
            recorder.persist()
            from efficiency import _safe_filename
            name = _safe_filename(str(row['sample_id'])) + '.json'
            archive = config.output_dir / 'efficiency_history' / run_id
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(config.output_dir / 'efficiency_samples' / name, archive / name)

    workers = max(1, min(args.workers, len(todo)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, row): str(row["sample_id"]) for row in todo}
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                record = future.result()
            except BaseException as exc:
                if not args.continue_on_error:
                    raise
                failures.append((sample_id, f"{type(exc).__name__}: {exc}"))
                print(f"[judge] FAILED {sample_id}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if args.continue_on_error:
                completed_dir.mkdir(parents=True, exist_ok=True)
                runner.write_json_atomic(completed_dir / f"{sample_id}.json", record)
            else:
                runner.append_jsonl(judge_file, record)
                existing.append(record)
            produced += 1

    if failures:
        (config.output_dir / "judge_failures.jsonl").write_text(
            json.dumps([{"sample_id": s, "error": e} for s, e in failures], indent=2),
            encoding="utf-8",
        )
        print(
            f"[judge] {len(failures)} failed judgements; run again to retry "
            f"(first: {[f[0] for f in failures][:3]})",
            flush=True,
        )

    if args.continue_on_error:
        _finalize(runner, judge_file, records, existing, completed_dir)

    report_coverage(records, runner.read_jsonl(judge_file))
    runner.rebuild_efficiency_artifacts(config.output_dir)
    status = {"judged": len(records), "generated": produced, "failed": len(failures)}
    print(f"[judge] done {status}", flush=True)
    return status


def _finalize(runner, judge_file: Path, records, existing, completed_dir: Path) -> None:
    """Merge completed judgements into the ordered judge file, if the cell is complete.

    The cache is re-read here rather than passed in: the caller's copy predates this
    run, and merging with it would report a complete cell as still missing.
    """
    completed_records = runner.read_record_cache(completed_dir)
    complete, missing = runner.merge_record_cache(judge_file, records, existing, completed_records)
    if complete:
        # See run_external_baseline: the cache holds what is *not* in the output file,
        # and a merge is what makes that true.
        runner.clear_merged_cache(completed_dir, completed_records)
        print(f"[judge] merged {len(records)} judgements into {judge_file}", flush=True)
    else:
        print(
            f"[judge] {len(missing)} samples still unjudged; run again to continue "
            f"(first: {missing[:3]})",
            flush=True,
        )


def report_coverage(records: list[dict[str, Any]], judged: list[dict[str, Any]]) -> None:
    """Print judged/total, and say plainly when the cell is too incomplete to report.

    The threshold is a gate on publishing a headline number, not on the run: a cell
    with missing judgements is still resumable, but a mean over the ones that happened
    to parse is not the dataset's number.
    """
    total = len(records)
    done = len(judged)
    coverage = done / total if total else 0.0
    line = f"[judge] coverage {done}/{total} ({coverage:.3f})"
    if coverage < 0.98:
        line += " -- below 0.98: do not report this cell's aggregate yet"
    print(line, flush=True)


def main(argv: list[str] | None = None) -> int:
    run_config.load_env_file()
    judge(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
