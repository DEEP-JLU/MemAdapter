"""Machinery shared by the two external judge families.

The two datasets judge differently in almost every respect, so this module holds only
the parts that must be identical for the arms to stay comparable:

*   **the call** -- one retried ``ModelClient.complete_result`` whose metadata is kept.
    ``retry_call`` returns only text, and re-issuing a request to recover the token
    counts would bill every sample twice;
*   **JSON recovery** -- each dataset's own ladder of strategies, because upstream's
    tolerance for a fenced or prose-wrapped object is part of the protocol being
    reproduced, not an implementation detail. Being *stricter* here would fail samples
    the paper's numbers include; being *looser* would make our numbers better than the
    paper's for a reason that is not the method;
*   **the record shape** -- both arms' judge records carry the same keys, so
    ``validate_external`` can pair them by ``sample_id`` and compare field by field.

What to *ask* is not here: that is the rubrics, and each family fills its own
placeholders in :mod:`judge_memtrap` / :mod:`judge_persist`.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: Bumped only when a judge prompt or the judge record schema changes, so a stale
#: judge record cannot be silently reused after a rubric edit.
JUDGE_PROMPT_VERSION = "external-judge-20260917"

#: ``dimension_1_factual_correctness``, ``dimension_2_relevance_purity``, ...
#: Parsed generically because the three MemTrapBench rubrics do not share a
#: dimension set: ``shared`` asks for four, ``poison`` and ``number_game`` for two
#: each, under different names.
DIMENSION_RE = re.compile(r"^dimension_(\d+)_(.+)$")


def complete_capturing(
    runner,
    client,
    system_prompt: str,
    user_prompt: str,
    label: str,
    *,
    max_tokens: int | None = None,
    recorder=None,
):
    # NOTE: ``recorder`` is threaded in by ``run_external_judge.work_payload`` so the
    # judge records the same efficiency traces as the generation arms. The body below
    # already implements that path; the parameter was missing from this signature,
    # which made every recorder-enabled judgement raise
    # ``TypeError: complete_capturing() got an unexpected keyword argument 'recorder'``
    # and left the whole cell unmerged (12,594 pending judgements on 2026-09-21).
    """One retried call that also hands back the call metadata.

    ``retry_call`` returns only the stripped text. The record needs the token counts,
    latency and finish reason too, and re-issuing the request to collect them would
    bill every sample twice. Capturing the result inside the closure keeps the retry
    policy itself -- the backoff and the server's ``retry_after`` hint -- identical to
    the generation arms, rather than reimplemented here where it could drift.

    ``max_tokens`` is left unset by the generation arms on purpose: not passing it is
    what keeps both of them on the same ``MEMADAPTER_MAX_TOKENS``. The judge passes its
    own, because the judge budget is part of the judging protocol.
    """
    captured: list[Any] = []

    def once() -> str:
        import time
        from datetime import datetime, timezone
        start = time.perf_counter()
        at = datetime.now(timezone.utc).isoformat()
        attempt = recorder.next_attempt_number('judge') if recorder else 1
        try:
            result = client.complete_result(system_prompt, user_prompt, max_tokens=max_tokens)
        except Exception as exc:
            if recorder:
                recorder.record_failure(stage='judge', attempt=attempt,
                    request_started_at=at, response_received_at=datetime.now(timezone.utc).isoformat(),
                    latency_ms=(time.perf_counter()-start)*1000, error=exc,
                    model=client.model, base_url=client.base_url, temperature=client.temperature,
                    max_tokens=max_tokens or client.max_tokens,
                    **{key: getattr(exc, '_efficiency_'+key, None) for key in
                       ('input_tokens','output_tokens','total_tokens','cached_input_tokens','reasoning_tokens','finish_reason')},
                    usage_available=bool(getattr(exc, '_efficiency_usage_available', False)))
            raise
        if recorder:
            recorder.record_success(result, stage='judge', attempt=attempt)
        captured.append(result)
        return result.text or ""

    if recorder:
        text = runner.retry_call_traced(lambda _attempt: once(), recorder=recorder,
                                       stage='judge', label=label)
    else:
        text = runner.retry_call(once, label=label)
    # retry_call returns only after an attempt produced non-empty text, so the last
    # capture is always the successful one.
    return captured[-1], text


def strip_think_blocks(text: str) -> tuple[str, str]:
    """Split a judge's reasoning trace off its answer, as PersistBench does.

    Upstream keeps the trace (``extracted_reasoning_content``) and parses only what
    remains, so a judge that narrates inside ``<think>`` before emitting its JSON is
    parsed normally rather than counted as a failure.
    """
    thinking = [
        block.strip()
        for block in re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    ]
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return stripped, "\n".join(thinking)


def extract_json_memtrap(raw: str) -> tuple[dict[str, Any] | None, str]:
    """MemTrapBench's recovery: strip a code fence, then parse. One step, no rescue.

    Mirrors upstream ``eval_common.parse_judge_output`` including its willingness to
    give up -- there, a judge that wraps the object in prose is a failed judgement, and
    reproducing that is the point.
    """
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(cleaned)
    except Exception as exc:
        return None, f"not JSON after fence stripping: {type(exc).__name__}"
    if not isinstance(parsed, dict):
        return None, f"parsed to {type(parsed).__name__}, not an object"
    return parsed, ""


def extract_json_persist(content: str) -> tuple[dict[str, Any] | None, str]:
    """PersistBench's recovery ladder: as-is, first object in the text, code block.

    Mirrors upstream ``utils.extract_json_from_response``, whose middle step is a
    one-level-deep balanced-object match -- enough for ``{"reasoning": ..., "score":
    ...}`` and deliberately not a full JSON scanner.
    """
    content = content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed, ""

    inline = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content, re.DOTALL)
    if inline:
        try:
            parsed = json.loads(inline.group(0))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed, ""

    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if block:
        try:
            parsed = json.loads(block.group(1))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed, ""

    return None, "no JSON object found (as-is, inline, or fenced)"


def declared_dimensions(template: str) -> list[str]:
    """The dimension keys a MemTrapBench rubric asks for, in the order it lists them.

    Read off the rubric rather than hard-coded: the judge must answer exactly the
    dimensions its own prompt defines, and a rubric edit that added one should fail the
    check rather than be silently ignored.
    """
    return re.findall(r'"(dimension_\d+_[a-z_]+)"\s*:', template)


def score_memtrap_dimensions(
    parsed: dict[str, Any],
    expected: list[str],
    *,
    low: int = 0,
    high: int = 5,
) -> tuple[dict[str, int], list[str]]:
    """Validate a parsed MemTrapBench judgement against the rubric's dimension list.

    Returns ``({dimension: score}, problems)``. Nothing is coerced: a judge that
    answered the wrong number of dimensions, used another scale or wrote a
    non-integer yields a problem and no score, so the caller can retry the sample.
    Recording a fabricated 0 would enter the average as if the judge had said it.
    """
    problems: list[str] = []
    found = {key for key in parsed if isinstance(key, str) and DIMENSION_RE.match(key)}
    missing = [key for key in expected if key not in found]
    extra = sorted(found - set(expected))
    if missing:
        problems.append(f"missing dimensions: {missing}")
    if extra:
        problems.append(f"unexpected dimensions: {extra}")

    scores: dict[str, int] = {}
    for key in expected:
        info = parsed.get(key)
        if not isinstance(info, dict):
            problems.append(f"{key}: not an object")
            continue
        value = info.get("score")
        # bool is an int subclass, and `true` is not a score.
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(f"{key}: score {value!r} is not an integer")
            continue
        if not low <= value <= high:
            problems.append(f"{key}: score {value} outside {low}-{high}")
            continue
        scores[key] = value
    return scores, problems


def score_persist(
    parsed: dict[str, Any],
    *,
    low: int,
    high: int,
) -> tuple[int | None, str, list[str]]:
    """Validate a parsed PersistBench judgement; return ``(score, key, problems)``.

    Upstream reads ``score or rating`` and needs a ``reasoning`` string
    (``execution/judgment.py``). ``rating`` is the beneficial class's key, and it is
    normalised into ``score`` here because upstream's own analysis reads ``score``
    only (``analysis/persistbench/failure_rates.extract_score``) -- without the
    alias, the whole beneficial class would silently score as missing.
    """
    problems: list[str] = []
    reasoning = parsed.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        problems.append("missing 'reasoning'")

    key = "score" if parsed.get("score") is not None else "rating"
    raw = parsed.get(key)
    if raw is None:
        problems.append(f"neither 'score' nor 'rating' present in {sorted(parsed)}")
        return None, key, problems
    if isinstance(raw, bool):
        problems.append(f"{key} {raw!r} is not a score")
        return None, key, problems
    try:
        value = int(raw)
    except (TypeError, ValueError):
        problems.append(f"{key} {raw!r} is not an integer")
        return None, key, problems
    if not low <= value <= high:
        problems.append(f"{key} {value} outside {low}-{high}")
        return None, key, problems
    return value, key, problems
