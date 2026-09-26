"""Efficiency instrumentation for generation and judge API calls.

``efficiency_calls.jsonl`` is rebuilt from ``efficiency_samples/<sample_id>.json``,
which holds the latest recorded run per phase. A run that makes no model calls -- the
stage cache answering every stage -- leaves the recorded entry alone, so a free rerun
cannot erase a cell's cost evidence. A rerun that *does* call the model replaces the
entry, so the files account for the runs recorded in them, not for every call a cell
has ever made.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _first_value(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        value = _value(obj, name)
        if value is not None:
            return value
    return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_usage(response: Any) -> dict[str, Any]:
    """Extract usage fields across OpenAI-compatible response variants."""

    usage = _value(response, "usage")
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cached_input_tokens": None,
            "reasoning_tokens": None,
            "usage_available": False,
        }
    prompt_details = _value(usage, "prompt_tokens_details") or _value(
        usage, "input_tokens_details"
    )
    completion_details = _value(usage, "completion_tokens_details") or _value(
        usage, "output_tokens_details"
    )
    input_tokens = _as_int(_first_value(usage, ("prompt_tokens", "input_tokens")))
    output_tokens = _as_int(_first_value(usage, ("completion_tokens", "output_tokens")))
    total_tokens = _as_int(_first_value(usage, ("total_tokens",)))
    cached_input_tokens = _as_int(_first_value(
        prompt_details or {}, ("cached_tokens", "cache_read_input_tokens")
    ))
    reasoning_tokens = _as_int(_first_value(
        completion_details or {}, ("reasoning_tokens", "thinking_tokens")
    ))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = int(input_tokens) + int(output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_input_tokens,
        "reasoning_tokens": reasoning_tokens,
        "usage_available": any(
            value is not None for value in (input_tokens, output_tokens, total_tokens)
        ),
    }


@dataclass(frozen=True)
class ModelCallResult:
    """Text plus metadata from one successful model API call."""

    text: str
    model: str
    base_url: str
    temperature: float | None
    max_tokens: int | None
    request_started_at: str
    response_received_at: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached_input_tokens: int | None
    reasoning_tokens: int | None
    usage_available: bool
    finish_reason: str | None


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return cleaned or uuid.uuid4().hex


def _sum_known(rows: list[dict[str, Any]], key: str) -> int | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return int(sum(values)) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 3)
    return round((ordered[middle - 1] + ordered[middle]) / 2, 3)


class EfficiencyRecorder:
    """Collect one sample's calls, stage timings, and aggregate metrics."""

    def __init__(
        self,
        *,
        output_dir: Path,
        sample_id: str,
        task: str,
        method: str,
        model: str,
        memory_system: str,
        phase: str,
        run_id: str,
        run_started_at: str,
        base_url: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.sample_id = sample_id
        self.task = task
        self.method = method
        self.model = model
        self.memory_system = memory_system
        self.phase = phase
        self.run_id = run_id
        self.run_started_at = run_started_at
        self.base_url = base_url
        self.sample_started_at = utc_now()
        self._sample_started_perf = time.perf_counter()
        self.calls: list[dict[str, Any]] = []
        self.stages: dict[str, dict[str, Any]] = {}
        self._stage_starts: dict[str, float] = {}
        self.finished_at: str | None = None
        self._finished = False
        self._summary: dict[str, Any] | None = None

    def start_stage(self, stage: str) -> None:
        self._stage_starts[stage] = time.perf_counter()

    def next_attempt_number(self, stage: str) -> int:
        """Return the next request number for a stage, including schema retries."""

        existing = [
            int(row.get("attempt") or 0)
            for row in self.calls
            if row.get("stage") == stage
        ]
        return max(existing, default=0) + 1

    def finish_stage(self, stage: str, wall_time_ms: float, cache_hit: bool) -> None:
        call_rows = [row for row in self.calls if row.get("stage") == stage]
        self.stages[stage] = self._aggregate_calls(
            call_rows,
            extra={
                "stage": stage,
                "stage_wall_time_ms": round(float(wall_time_ms), 3),
                "cache_hit": bool(cache_hit),
            },
        )

    def record_success(
        self,
        result: ModelCallResult,
        *,
        stage: str,
        attempt: int,
        retry_sleep_ms: float = 0.0,
    ) -> None:
        self.calls.append(
            self._call_row(
                stage=stage,
                attempt=attempt,
                success=True,
                request_started_at=result.request_started_at,
                response_received_at=result.response_received_at,
                latency_ms=result.latency_ms,
                retry_sleep_ms=retry_sleep_ms,
                model=result.model,
                base_url=result.base_url,
                temperature=result.temperature,
                max_tokens=result.max_tokens,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.total_tokens,
                cached_input_tokens=result.cached_input_tokens,
                reasoning_tokens=result.reasoning_tokens,
                usage_available=result.usage_available,
                finish_reason=result.finish_reason,
            )
        )

    def record_failure(
        self,
        *,
        stage: str,
        attempt: int,
        request_started_at: str,
        response_received_at: str,
        latency_ms: float,
        error: BaseException,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        cached_input_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        usage_available: bool = False,
        finish_reason: str | None = None,
        retry_sleep_ms: float = 0.0,
    ) -> None:
        self.calls.append(
            self._call_row(
                stage=stage,
                attempt=attempt,
                success=False,
                request_started_at=request_started_at,
                response_received_at=response_received_at,
                latency_ms=latency_ms,
                retry_sleep_ms=retry_sleep_ms,
                model=model,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                error_type=type(error).__name__,
                error=str(error),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_input_tokens=cached_input_tokens,
                reasoning_tokens=reasoning_tokens,
                usage_available=usage_available,
                finish_reason=finish_reason,
            )
        )

    def add_retry_sleep(self, milliseconds: float) -> None:
        if self.calls:
            self.calls[-1]["retry_sleep_ms"] = round(
                float(self.calls[-1].get("retry_sleep_ms") or 0) + float(milliseconds),
                3,
            )

    def add_retry_gap(self, stage: str, milliseconds: float) -> None:
        """Attach an observed inter-request wait after a failed call."""

        if not self.calls:
            return
        last = self.calls[-1]
        if last.get("stage") == stage and not last.get("success") and milliseconds > 0:
            last["retry_sleep_ms"] = round(
                float(last.get("retry_sleep_ms") or 0) + float(milliseconds),
                3,
            )

    def record_external_attempt(
        self,
        *,
        stage: str = "judge",
        request: dict[str, Any],
        response: Any = None,
        error: BaseException | None = None,
        request_started_at: str,
        response_received_at: str,
        latency_ms: float,
        attempt: int,
    ) -> None:
        usage = extract_usage(response) if response is not None else {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cached_input_tokens": None,
            "reasoning_tokens": None,
            "usage_available": False,
        }
        if error is None:
            self.calls.append(
                self._call_row(
                    stage=stage,
                    attempt=attempt,
                    success=True,
                    request_started_at=request_started_at,
                    response_received_at=response_received_at,
                    latency_ms=latency_ms,
                    retry_sleep_ms=0.0,
                    model=str(request.get("model") or self.model),
                    base_url=self.base_url,
                    temperature=request.get("temperature"),
                    max_tokens=request.get("max_tokens"),
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    total_tokens=usage["total_tokens"],
                    cached_input_tokens=usage["cached_input_tokens"],
                    reasoning_tokens=usage["reasoning_tokens"],
                    usage_available=usage["usage_available"],
                    finish_reason=_value(
                        (_value(response, "choices") or [None])[0], "finish_reason"
                    ),
                )
            )
        else:
            self.calls.append(
                self._call_row(
                    stage=stage,
                    attempt=attempt,
                    success=False,
                    request_started_at=request_started_at,
                    response_received_at=response_received_at,
                    latency_ms=latency_ms,
                    retry_sleep_ms=0.0,
                    model=str(request.get("model") or self.model),
                    base_url=self.base_url,
                    temperature=request.get("temperature"),
                    max_tokens=request.get("max_tokens"),
                    error_type=type(error).__name__,
                    error=str(error),
                )
            )

    def _call_row(self, **values: Any) -> dict[str, Any]:
        row = {
            "run_id": self.run_id,
            "sample_id": self.sample_id,
            "task": self.task,
            "method": self.method,
            "model_family": self.model,
            "memory_system": self.memory_system,
            "phase": self.phase,
        }
        row.update(values)
        for key in ("latency_ms", "retry_sleep_ms"):
            if isinstance(row.get(key), (int, float)):
                row[key] = round(float(row[key]), 3)
        return row

    @staticmethod
    def _aggregate_calls(
        calls: list[dict[str, Any]], *, extra: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        values = {
            "input_tokens": _sum_known(calls, "input_tokens"),
            "output_tokens": _sum_known(calls, "output_tokens"),
            "total_tokens": _sum_known(calls, "total_tokens"),
            "cached_input_tokens": _sum_known(calls, "cached_input_tokens"),
            "reasoning_tokens": _sum_known(calls, "reasoning_tokens"),
            "api_time_ms": round(
                sum(float(row.get("latency_ms") or 0) for row in calls), 3
            ),
            "retry_wait_ms": round(
                sum(float(row.get("retry_sleep_ms") or 0) for row in calls), 3
            ),
            "api_call_count": len(calls),
            "successful_api_call_count": sum(bool(row.get("success")) for row in calls),
            "failed_api_call_count": sum(not bool(row.get("success")) for row in calls),
            "usage_missing_call_count": sum(
                bool(row.get("success")) and not bool(row.get("usage_available"))
                for row in calls
            ),
        }
        retries_by_stage: dict[str, int] = {}
        for row in calls:
            stage = str(row.get("stage") or "unknown")
            retries_by_stage[stage] = max(
                retries_by_stage.get(stage, 0), int(row.get("attempt") or 1) - 1
            )
        values["retry_count"] = sum(retries_by_stage.values())
        if extra:
            values.update(extra)
        return values

    def finish(self, *, success: bool, error: BaseException | str | None = None) -> dict[str, Any]:
        if self._finished and self._summary is not None:
            return self._summary
        self.finished_at = utc_now()
        elapsed_ms = (time.perf_counter() - self._sample_started_perf) * 1000
        summary = self._aggregate_calls(self.calls)
        summary.update(
            {
                "run_id": self.run_id,
                "run_started_at": self.run_started_at,
                "sample_id": self.sample_id,
                "task": self.task,
                "method": self.method,
                "model_family": self.model,
                "memory_system": self.memory_system,
                "phase": self.phase,
                "sample_started_at": self.sample_started_at,
                "sample_finished_at": self.finished_at,
                "sample_wall_time_ms": round(elapsed_ms, 3),
                "success": bool(success),
                "error": None if error is None else str(error),
                "stages": self.stages,
            }
        )
        self._summary = summary
        self._finished = True
        return summary

    def persist(self) -> None:
        if self._summary is None:
            raise RuntimeError("EfficiencyRecorder.finish() must be called before persist()")
        directory = self.output_dir / "efficiency_samples"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_safe_filename(self.sample_id)}.json"
        existing: dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except json.JSONDecodeError:
                existing = {}
        phases = existing.get("phases")
        if not isinstance(phases, dict):
            phases = {}
        previous = phases.get(self.phase)
        if not self.calls and isinstance(previous, dict) and previous.get("calls"):
            # This phase made no model calls, which in practice means the stage cache
            # answered every stage -- a rerun of a finished cell. Writing the entry
            # would replace the calls this cell really made with an empty list, so the
            # rerun that is supposed to be a free no-op would erase the cell's cost
            # evidence: efficiency_calls.jsonl is rebuilt from these entries, and a
            # zero-call replay has nothing to contribute to it. Keep the recorded run.
            return
        phases[self.phase] = {"sample": self._summary, "calls": self.calls}
        payload = {"sample_id": self.sample_id, "phases": phases}
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


class TracedOpenAI:
    """Minimal OpenAI client proxy that traces chat completion calls."""

    def __init__(
        self,
        client: Any,
        recorder: EfficiencyRecorder,
        *,
        stage: str = "judge",
        api_mode: str | None = None,
    ) -> None:
        self._client = client
        self._recorder = recorder
        configured_mode = (
            api_mode
            or os.environ.get("JUDGE_API_MODE", "")
            or os.environ.get("OPENAI_API_MODE", "")
        ).lower().replace("-", "_")
        self._api_mode = "responses" if configured_mode in {"response", "responses"} else "chat_completions"
        self.chat = _TracedChat(
            client.chat,
            recorder,
            stage=stage,
            client=client,
            api_mode=self._api_mode,
        )
        self.responses = _TracedResponses(client.responses, recorder, stage=stage)


class _TracedChat:
    def __init__(
        self,
        chat: Any,
        recorder: EfficiencyRecorder,
        *,
        stage: str,
        client: Any,
        api_mode: str,
    ) -> None:
        self.completions = _TracedCompletions(
            chat.completions,
            recorder,
            stage=stage,
            client=client,
            api_mode=api_mode,
        )


def _responses_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        for part in content or []:
            value = getattr(part, "text", None)
            if value is None and isinstance(part, dict):
                value = part.get("text")
            if value:
                parts.append(str(value))
    return "".join(parts).strip()


class _TracedResponses:
    def __init__(self, responses: Any, recorder: EfficiencyRecorder, *, stage: str) -> None:
        self._responses = responses
        self._recorder = recorder
        self._stage = stage

    def create(self, **request: Any) -> Any:
        started_at = utc_now()
        started_perf = time.perf_counter()
        attempt = len(self._recorder.calls) + 1
        try:
            response = self._responses.create(**request)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self._recorder.record_external_attempt(
                stage=self._stage,
                request=request,
                error=exc,
                request_started_at=started_at,
                response_received_at=utc_now(),
                latency_ms=(time.perf_counter() - started_perf) * 1000,
                attempt=attempt,
            )
            raise
        self._recorder.record_external_attempt(
            stage=self._stage,
            request=request,
            response=response,
            request_started_at=started_at,
            response_received_at=utc_now(),
            latency_ms=(time.perf_counter() - started_perf) * 1000,
            attempt=attempt,
        )
        return response


class _TracedCompletions:
    def __init__(
        self,
        completions: Any,
        recorder: EfficiencyRecorder,
        *,
        stage: str,
        client: Any,
        api_mode: str,
    ) -> None:
        self._completions = completions
        self._recorder = recorder
        self._stage = stage
        self._client = client
        self._api_mode = api_mode
        self._last_call_finished_perf: float | None = None
        self._last_call_failed = False

    def create(self, **request: Any) -> Any:
        if self._api_mode == "responses":
            return self._create_via_responses(request)
        started_at = utc_now()
        started_perf = time.perf_counter()
        if self._last_call_failed and self._last_call_finished_perf is not None:
            self._recorder.add_retry_gap(
                self._stage,
                (started_perf - self._last_call_finished_perf) * 1000,
            )
        attempt = len(self._recorder.calls) + 1
        try:
            response = self._completions.create(**request)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self._recorder.record_external_attempt(
                stage=self._stage,
                request=request,
                error=exc,
                request_started_at=started_at,
                response_received_at=utc_now(),
                latency_ms=(time.perf_counter() - started_perf) * 1000,
                attempt=attempt,
            )
            self._last_call_finished_perf = time.perf_counter()
            self._last_call_failed = True
            raise
        self._recorder.record_external_attempt(
            stage=self._stage,
            request=request,
            response=response,
            request_started_at=started_at,
            response_received_at=utc_now(),
            latency_ms=(time.perf_counter() - started_perf) * 1000,
            attempt=attempt,
        )
        self._last_call_finished_perf = time.perf_counter()
        self._last_call_failed = False
        return response

    def _create_via_responses(self, request: dict[str, Any]) -> Any:
        source = dict(request)
        messages = source.pop("messages", [])
        response_request: dict[str, Any] = {}
        if "model" in source:
            response_request["model"] = source.pop("model")

        system_parts = []
        input_messages = []
        for message in messages:
            if not isinstance(message, dict):
                input_messages.append(message)
                continue
            if str(message.get("role", "")).lower() == "system":
                system_parts.append(str(message.get("content", "")))
            else:
                input_messages.append(message)
        if system_parts:
            response_request["instructions"] = "\n\n".join(system_parts)
        if len(input_messages) == 1 and isinstance(input_messages[0], dict):
            if str(input_messages[0].get("role", "")).lower() == "user":
                response_request["input"] = input_messages[0].get("content", "")
            else:
                response_request["input"] = input_messages
        else:
            response_request["input"] = input_messages

        if "temperature" in source:
            response_request["temperature"] = source.pop("temperature")
        if "max_tokens" in source:
            response_request["max_output_tokens"] = source.pop("max_tokens")
        if "response_format" in source:
            response_request["text"] = {"format": source.pop("response_format")}
        if "reasoning_effort" in source:
            response_request["reasoning"] = {
                "effort": source.pop("reasoning_effort")
            }
        # extra_body is a Chat Completions extension and has no portable
        # Responses equivalent; the GPT gateway does not need it here.
        source.pop("extra_body", None)
        for key in ("top_p", "presence_penalty", "frequency_penalty", "store"):
            if key in source:
                response_request[key] = source.pop(key)
        response_request.update(source)

        started_at = utc_now()
        started_perf = time.perf_counter()
        if self._last_call_failed and self._last_call_finished_perf is not None:
            self._recorder.add_retry_gap(
                self._stage,
                (started_perf - self._last_call_finished_perf) * 1000,
            )
        attempt = len(self._recorder.calls) + 1
        try:
            raw_response = self._client.responses.create(**response_request)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self._recorder.record_external_attempt(
                stage=self._stage,
                request=response_request,
                error=exc,
                request_started_at=started_at,
                response_received_at=utc_now(),
                latency_ms=(time.perf_counter() - started_perf) * 1000,
                attempt=attempt,
            )
            self._last_call_finished_perf = time.perf_counter()
            self._last_call_failed = True
            raise

        text = _responses_text(raw_response)
        finish_reason = getattr(raw_response, "status", None)
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text, reasoning_content=""),
                    finish_reason=finish_reason,
                )
            ],
            usage=getattr(raw_response, "usage", None),
            output_text=text,
            raw_response=raw_response,
        )
        self._recorder.record_external_attempt(
            stage=self._stage,
            request=response_request,
            response=response,
            request_started_at=started_at,
            response_received_at=utc_now(),
            latency_ms=(time.perf_counter() - started_perf) * 1000,
            attempt=attempt,
        )
        self._last_call_finished_perf = time.perf_counter()
        self._last_call_failed = False
        return response


_ARTIFACT_LOCK = threading.Lock()


def _read_sample_files(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    directory = output_dir / "efficiency_samples"
    if not directory.is_dir():
        return samples, calls
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        phase_payloads = payload.get("phases")
        if not isinstance(phase_payloads, dict):
            continue
        sample_row: dict[str, Any] = {"sample_id": payload.get("sample_id"), "phases": {}}
        for phase, phase_data in phase_payloads.items():
            if not isinstance(phase_data, dict):
                continue
            sample = phase_data.get("sample")
            if isinstance(sample, dict):
                sample_row["phases"][phase] = sample
            phase_calls = phase_data.get("calls")
            if isinstance(phase_calls, list):
                calls.extend(row for row in phase_calls if isinstance(row, dict))
        if sample_row["phases"]:
            samples.append(sample_row)
    calls.sort(key=lambda row: str(row.get("request_started_at") or ""))
    samples.sort(key=lambda row: str(row.get("sample_id") or ""))
    return samples, calls


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def _aggregate_run(
    rows: list[dict[str, Any]], *, include_task_summary: bool = True
) -> dict[str, Any]:
    completed = [row for row in rows if row.get("success")]
    sample_times = [float(row.get("sample_wall_time_ms") or 0) for row in completed]
    started = [
        str(row.get("run_started_at") or row.get("sample_started_at"))
        for row in rows
        if row.get("run_started_at") or row.get("sample_started_at")
    ]
    finished = [str(row.get("sample_finished_at")) for row in rows if row.get("sample_finished_at")]
    wall_ms: float | None = None
    if started and finished:
        try:
            wall_ms = round(
                (
                    datetime.fromisoformat(max(finished))
                    - datetime.fromisoformat(min(started))
                ).total_seconds()
                * 1000,
                3,
            )
        except ValueError:
            wall_ms = None
    calls: list[dict[str, Any]] = []
    for row in rows:
        calls.extend(row.get("_calls", []))
    aggregate = EfficiencyRecorder._aggregate_calls(calls)
    sample_token_values = {
        key: [
            float(row[key])
            for row in completed
            if isinstance(row.get(key), (int, float))
        ]
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    aggregate.update(
        {
            "sample_count": len(rows),
            "completed_sample_count": len(completed),
            "failed_sample_count": len(rows) - len(completed),
            "mean_sample_wall_time_ms": _mean(sample_times),
            "median_sample_wall_time_ms": _median(sample_times),
            "p95_sample_wall_time_ms": _p95(sample_times),
            "mean_input_tokens_per_sample": _mean(sample_token_values["input_tokens"]),
            "median_input_tokens_per_sample": _median(sample_token_values["input_tokens"]),
            "p95_input_tokens_per_sample": _p95(sample_token_values["input_tokens"]),
            "mean_output_tokens_per_sample": _mean(sample_token_values["output_tokens"]),
            "median_output_tokens_per_sample": _median(sample_token_values["output_tokens"]),
            "p95_output_tokens_per_sample": _p95(sample_token_values["output_tokens"]),
            "mean_tokens_per_sample": _mean(sample_token_values["total_tokens"]),
            "median_tokens_per_sample": _median(sample_token_values["total_tokens"]),
            "p95_tokens_per_sample": _p95(sample_token_values["total_tokens"]),
            "batch_wall_time_ms": wall_ms,
            "samples_per_second": (
                round(len(completed) / (wall_ms / 1000), 6)
                if wall_ms and wall_ms > 0
                else None
            ),
            "output_tokens_per_second": (
                round(float(aggregate["output_tokens"]) / (wall_ms / 1000), 6)
                if wall_ms and wall_ms > 0 and aggregate["output_tokens"] is not None
                else None
            ),
            "total_tokens_per_second": (
                round(float(aggregate["total_tokens"]) / (wall_ms / 1000), 6)
                if wall_ms and wall_ms > 0 and aggregate["total_tokens"] is not None
                else None
            ),
        }
    )
    if include_task_summary:
        aggregate["task_summary"] = {}
        for task in sorted({str(row.get("task")) for row in rows}):
            task_rows = [row for row in rows if str(row.get("task")) == task]
            aggregate["task_summary"][task] = _aggregate_run(
                task_rows, include_task_summary=False
            )
    return aggregate


def rebuild_efficiency_artifacts(output_dir: Path) -> dict[str, Any]:
    """Rebuild flat JSONL files and aggregate summary from per-sample records."""

    with _ARTIFACT_LOCK:
        samples, calls = _read_sample_files(output_dir)
        _write_jsonl(output_dir / "efficiency_samples.jsonl", samples)
        _write_jsonl(output_dir / "efficiency_calls.jsonl", calls)

        phase_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for sample_row in samples:
            for phase, sample in sample_row.get("phases", {}).items():
                if not isinstance(sample, dict):
                    continue
                phase_sample = dict(sample)
                phase_sample["_calls"] = [
                    call
                    for call in calls
                    if call.get("sample_id") == sample.get("sample_id")
                    and call.get("phase") == phase
                    and call.get("run_id") == sample.get("run_id")
                ]
                phase_rows.setdefault((phase, str(sample.get("run_id"))), []).append(phase_sample)

        runs = []
        for (phase, run_id), rows in sorted(phase_rows.items()):
            summary = _aggregate_run(rows)
            summary.update(
                {
                    "phase": phase,
                    "run_id": run_id,
                    "model_family": rows[0].get("model_family") if rows else None,
                    "method": rows[0].get("method") if rows else None,
                    "memory_system": rows[0].get("memory_system") if rows else None,
                }
            )
            runs.append(summary)
        payload = {
            "schema_version": 1,
            "retrieval": {
                "reused": True,
                "time_included": False,
                "tokens_included": False,
            },
            "runs": runs,
        }
        path = output_dir / "efficiency_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return payload


def new_run_id(phase: str) -> str:
    return f"{phase}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


__all__ = [
    "EfficiencyRecorder",
    "ModelCallResult",
    "TracedOpenAI",
    "extract_usage",
    "new_run_id",
    "rebuild_efficiency_artifacts",
    "utc_now",
]
