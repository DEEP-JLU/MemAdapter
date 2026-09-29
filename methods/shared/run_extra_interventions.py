"""Run post-retrieval comparison methods on one frozen retrieval file.

The four methods are Anti-Sycophancy, Self-ReCheck, Dynamic Partition, and
MemGate.  They deliberately implement a *retrieval-only* protocol: every model
call receives only ``current_request`` and the frozen ``retrieved_memories``.
In particular, this runner never reads, forwards, or serializes dialogue
context, query-session history, or task evidence.

Credentials and endpoint URLs must be supplied through the local environment.
They are neither loaded from a local secret file nor embedded in this module.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "methods" / "memadapter"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_memadapter as native
from efficiency import EfficiencyRecorder, new_run_id, rebuild_efficiency_artifacts, utc_now
from model_client import ModelClient
from prompts import ANTI_SYCOPHANCY, DYNAMIC_PARTITION, SELF_RECHECK


METHODS = ("anti_sycophancy", "self_recheck", "dynamic_partition", "memgate")
SYSTEMS = ("AMEM", "Mem0", "naiveRAG", "MemoryBank", "LightMem")


def configure_public_environment() -> None:
    """Pin the public DeepSeek-V4-Flash protocol without handling secrets.

    The key and endpoint remain caller-provided through ``DEEPSEEK_API_KEY`` and
    ``DEEPSEEK_BASE_URL``.  Rejecting a conflicting setting avoids a quiet change
    to the protocol while keeping all private deployment details out of source.
    """

    required = {
        "DEEPSEEK_MODEL": "DeepSeek-V4-Flash",
        "MEMADAPTER_TEMPERATURE": "0.2",
        "MEMADAPTER_MAX_TOKENS": "4096",
        "MEMADAPTER_ENABLE_REASONING": "false",
        "MEMADAPTER_API_MODE": "chat_completions",
    }
    for key, expected in required.items():
        actual = os.environ.get(key, "").strip()
        if actual and actual.lower() != expected.lower():
            raise RuntimeError(
                f"{key}={actual!r} conflicts with the public protocol; expected {expected!r}."
            )
        os.environ[key] = expected


def safe_error(exc: BaseException) -> str:
    """Return an error message with any environment credential redacted."""

    message = str(exc)
    for key, value in os.environ.items():
        if any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            if len(value) > 8:
                message = message.replace(value, "[REDACTED]")
    return message[:2000]


def memory_payload(memories: list[dict[str, Any]]) -> str:
    """Format only frozen memories, preserving IDs needed by selector methods."""

    return "\n\n".join(
        f"Memory {memory['memory_id']}: {memory['content']}" for memory in memories
    )


def normalized_memories(row: dict[str, Any]) -> list[dict[str, str]]:
    """Whitelisted retrieval-only view of a normalized retrieval row."""

    return [
        {
            "memory_id": str(memory["memory_id"]),
            "content": str(memory.get("memory_text") or memory.get("content") or ""),
        }
        for memory in row["retrieved_memories"]
    ]


def parse_selected_ids(text: str, memories: list[dict[str, str]], *, partition: bool) -> list[str]:
    data = json.loads(text)
    if partition:
        proposed = [
            str(memory_id)
            for group in data.get("groups", [])
            for memory_id in group.get("memory_ids", [])
        ]
        if len(proposed) != len(set(proposed)):
            raise ValueError("Dynamic Partition returned duplicate memory IDs")
        if set(proposed) != {memory["memory_id"] for memory in memories}:
            raise ValueError("Dynamic Partition must retain every supplied memory ID exactly once")
        return proposed
    proposed = {str(memory_id) for memory_id in data.get("keep_memory_ids", [])}
    known = {memory["memory_id"] for memory in memories}
    if not proposed <= known:
        raise ValueError("Self-ReCheck returned an unknown memory ID")
    return [memory["memory_id"] for memory in memories if memory["memory_id"] in proposed]


def call(client: ModelClient, recorder: EfficiencyRecorder, stage: str, system: str, user: str) -> str:
    started = time.perf_counter()
    try:
        for attempt in range(1, 4):
            request_started = utc_now()
            clock = time.perf_counter()
            try:
                result = client.complete_result(system, user)
            except Exception as exc:
                recorder.record_failure(
                    stage=stage,
                    attempt=recorder.next_attempt_number(stage),
                    request_started_at=request_started,
                    response_received_at=utc_now(),
                    latency_ms=(time.perf_counter() - clock) * 1000,
                    error=RuntimeError(safe_error(exc)),
                    model=client.model,
                    base_url=None,
                    temperature=client.temperature,
                    max_tokens=client.max_tokens,
                )
                if attempt == 3:
                    raise
                delay = 2**attempt
                recorder.add_retry_sleep(delay * 1000)
                time.sleep(delay)
            else:
                recorder.record_success(result, stage=stage, attempt=recorder.next_attempt_number(stage))
                if not result.text.strip():
                    raise ValueError("Empty model output")
                return result.text
    finally:
        recorder.finish_stage(
            stage,
            (time.perf_counter() - started) * 1000,
            cache_hit=False,
        )


def run_method(
    *, row: dict[str, Any], method: str, client: ModelClient, recorder: EfficiencyRecorder
) -> tuple[str, list[str], str | None, str]:
    """Return answer context using no inputs beyond query and frozen memories."""

    memories = normalized_memories(row)
    selected = [memory["memory_id"] for memory in memories]
    context = memory_payload(memories)
    intermediate: str | None = None
    extra = ""
    selector_input = json.dumps(
        {
            "question": str(row["current_request"]),
            "retrieved_memories": memories,
        },
        ensure_ascii=False,
    )
    if method == "anti_sycophancy":
        extra = ANTI_SYCOPHANCY
    elif method == "self_recheck":
        intermediate = call(client, recorder, "self_recheck", SELF_RECHECK, selector_input)
        selected = parse_selected_ids(intermediate, memories, partition=False)
        context = memory_payload([memory for memory in memories if memory["memory_id"] in selected])
    elif method == "dynamic_partition":
        for attempt in range(6):
            intermediate = call(client, recorder, "dynamic_partition", DYNAMIC_PARTITION, selector_input)
            try:
                selected = parse_selected_ids(intermediate, memories, partition=True)
                context = memory_payload(memories)
                break
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                if attempt == 5:
                    raise
    elif method == "memgate":
        recorder.finish_stage("memgate_filter", 0.0, cache_hit=False)
    else:
        raise ValueError(f"Unknown method: {method}")
    return context, selected, intermediate, extra


def build_answer_prompts(
    *, dataset: str, row: dict[str, Any], context: str, selected: list[str], model_name: str
) -> tuple[str, str]:
    """Build the ordinary answer prompt from the method's frozen-memory output."""

    if dataset == "memsyco":
        from methods.baseline.prompt_builder import build_official_baseline_system

        return (
            build_official_baseline_system(
                row["task"],
                {"official_context_text": context},
                model_name,
            ),
            str(row["current_request"]),
        )

    from external_benchmarks.run_external_baseline import answer_prompt_path, build_answer_prompts as external_prompts
    from external_benchmarks.text_norm import read_prompt_file

    memories = [
        {"memory_text": str(memory.get("memory_text") or memory.get("content") or "")}
        for memory in row["retrieved_memories"]
        if str(memory["memory_id"]) in selected
    ]
    return external_prompts(
        dataset="persistbench",
        query=str(row["current_request"]),
        memories=memories,
        model_name=model_name,
        template=read_prompt_file(answer_prompt_path("persistbench")),
    )


def generate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = native.read_jsonl(args.retrieval_file)
    if args.limit is not None:
        rows = rows[: args.limit]
    normalized_rows = [native.normalize_retrieval_row(row) for row in rows]
    ids = [str(row["sample_id"]) for row in normalized_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate retrieval IDs")

    config = {
        "dataset": args.dataset,
        "system": args.system,
        "method": args.method,
        "run_index": args.run_index,
        "retrieval_sha256": hashlib.sha256(args.retrieval_file.read_bytes()).hexdigest(),
        "model": "DeepSeek-V4-Flash",
        "temperature": 0.2,
        "max_tokens": 4096,
        "thinking": False,
        "input_protocol": "current_request_and_frozen_retrieved_memories_only",
    }
    config_path = output_dir / "intervention_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        if (output_dir / "outputs.jsonl").is_file():
            raise ValueError("Frozen run configuration mismatch")
    native.write_json_atomic(config_path, config)

    completed = native.read_record_cache(output_dir / "completed_records")
    done = {str(row["sample_id"]): row for row in native.read_jsonl(output_dir / "outputs.jsonl")}
    done.update(completed)
    todo = [row for row in normalized_rows if str(row["sample_id"]) not in done]
    run_id, run_started = new_run_id("generation"), utc_now()

    def work(row: dict[str, Any]) -> dict[str, Any]:
        client = ModelClient("DeepSeek-V4-Flash")
        recorder = EfficiencyRecorder(
            output_dir=output_dir,
            sample_id=str(row["sample_id"]),
            task=str(row["task"]),
            method=args.method,
            model="DeepSeek-V4-Flash",
            memory_system=args.system,
            phase="generation",
            run_id=run_id,
            run_started_at=run_started,
            base_url=None,
        )
        try:
            context, selected, intermediate, extra = run_method(
                row=row, method=args.method, client=client, recorder=recorder
            )
            system, user = build_answer_prompts(
                dataset=args.dataset,
                row=row,
                context=context,
                selected=selected,
                model_name=client.model,
            )
            if extra:
                system = f"{system}\n\n{extra}"
            answer = call(client, recorder, "answer", system, user)
            return {
                "sample_id": row["sample_id"],
                "task": row["task"],
                "dataset": args.dataset,
                "memory_system": args.system,
                "method": args.method,
                "run_index": args.run_index,
                "final_answer": answer,
                "final_context": context,
                "intermediate_output": intermediate,
                "selected_memory_ids": selected,
                "generation_model": client.model,
                "generation_temperature": client.temperature,
                "generation_max_tokens": client.max_tokens,
                "input_protocol": config["input_protocol"],
                "efficiency": recorder.finish(success=True),
            }
        except Exception as exc:
            recorder.finish(success=False, error=safe_error(exc))
            raise
        finally:
            recorder.persist()
            history = output_dir / "efficiency_history" / run_id
            history.mkdir(parents=True, exist_ok=True)
            from efficiency import _safe_filename

            name = f"{_safe_filename(str(row['sample_id']))}.json"
            shutil.copyfile(output_dir / "efficiency_samples" / name, history / name)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, row): row for row in todo}
        for future in as_completed(futures):
            row = futures[future]
            sample_id = str(row["sample_id"])
            try:
                record = future.result()
                native.write_json_atomic(
                    output_dir / "completed_records" / f"{hashlib.sha256(sample_id.encode()).hexdigest()}.json",
                    record,
                )
                done[sample_id] = record
            except Exception as exc:
                native.append_jsonl(
                    output_dir / "generation_failures.jsonl",
                    {
                        "sample_id": sample_id,
                        "method": args.method,
                        "error_type": type(exc).__name__,
                        "error": safe_error(exc),
                    },
                )
                if not args.continue_on_error:
                    raise

    native.write_jsonl_atomic(output_dir / "outputs.jsonl", [done[sample_id] for sample_id in ids if sample_id in done])
    native.write_json_atomic(
        output_dir / "coverage.json",
        {"expected": len(ids), "generated": len(done), "missing": [sample_id for sample_id in ids if sample_id not in done]},
    )
    rebuild_efficiency_artifacts(output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--retrieval-file", type=Path, required=True)
    parser.add_argument("--dataset", choices=("memsyco", "persistbench"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    configure_public_environment()
    generate(args)


if __name__ == "__main__":
    main()
