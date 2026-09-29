from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Callable, TypeVar


_T = TypeVar("_T")
_QUERY_RETRY_LOGGER = logging.getLogger("memory.query_retry")


def _is_retryable_embedding_query_error(exc: BaseException) -> bool:
    """Return whether an embedding-query failure is likely transient."""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status == 429 or isinstance(status, int) and 500 <= status < 600:
        return True

    message = str(exc).lower()
    transient_markers = (
        "connection error",
        "connection reset",
        "connection refused",
        "connection aborted",
        "remote protocol error",
        "remoteprotocolerror",
        "read timeout",
        "readtimeout",
        "connect timeout",
        "connecttimeout",
        "timed out",
        "timeout",
        "rate limit",
        "rate_limit",
        "too many requests",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "internal server error",
        "server disconnected",
        "http 429",
        "status code 429",
        "status_code=429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
    )
    return any(marker in message for marker in transient_markers)


def retry_embedding_query(
    operation: Callable[[], _T],
    *,
    method: str,
    sleep: Callable[[float], None] = time.sleep,
) -> _T:
    """Run a retrieval query with configurable exponential-backoff retries.

    Environment variables:
      MEMORY_QUERY_MAX_ATTEMPTS (default 5, including the first attempt)
      MEMORY_QUERY_RETRY_BASE_SECONDS (default 1)
      MEMORY_QUERY_RETRY_MULTIPLIER (default 4)
      MEMORY_QUERY_RETRY_MAX_SECONDS (default 16)
    """
    max_attempts = max(1, int(os.getenv("MEMORY_QUERY_MAX_ATTEMPTS", "5")))
    base_delay = max(0.0, float(os.getenv("MEMORY_QUERY_RETRY_BASE_SECONDS", "1")))
    multiplier = max(1.0, float(os.getenv("MEMORY_QUERY_RETRY_MULTIPLIER", "4")))
    max_delay = max(0.0, float(os.getenv("MEMORY_QUERY_RETRY_MAX_SECONDS", "16")))

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= max_attempts or not _is_retryable_embedding_query_error(exc):
                raise
            delay = min(max_delay, base_delay * (multiplier ** (attempt - 1)))
            _QUERY_RETRY_LOGGER.warning(
                "[memory-query-retry] method=%s attempt=%d/%d error=%s: %s sleep=%.2fs",
                method,
                attempt,
                max_attempts,
                type(exc).__name__,
                exc,
                delay,
            )
            sleep(delay)

    raise AssertionError("unreachable")


def parse_dialogue_to_messages(prior_dialogue: str) -> list[dict[str, str]]:
    """Best-effort conversion from benchmark dialogue text to chat messages."""
    text = (prior_dialogue or "").strip()
    if not text:
        return []

    messages: list[dict[str, str]] = []
    current_role: str | None = None
    current_lines: list[str] = []
    role_re = re.compile(r"^\s*(User|Assistant|System)\s*:\s*(.*)$", re.IGNORECASE)

    def flush() -> None:
        nonlocal current_role, current_lines
        content = "\n".join(line for line in current_lines).strip()
        if content:
            messages.append({"role": current_role or "user", "content": content})
        current_role = None
        current_lines = []

    for raw_line in re.split(r"\n+", text):
        line = raw_line.strip()
        if not line:
            flush()
            continue
        match = role_re.match(line)
        if match:
            flush()
            current_role = match.group(1).lower()
            current_lines = [match.group(2).strip()]
        else:
            if current_role is None:
                current_role = "user"
            current_lines.append(line)
    flush()

    return messages or [{"role": "user", "content": text}]


def format_retrieved_memories(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "[NO RETRIEVED MEMORIES]"
    parts: list[str] = []
    for i, mem in enumerate(memories, start=1):
        used = mem.get("used_content") or mem.get("content") or ""
        parts.append(f"### Memory {i}:\n{str(used).strip()}")
    return "\n\n".join(parts).strip()


def sample_user_id(method: str, sample_key: str | int | None, prior_dialogue: str, user_question: str) -> str:
    h = hashlib.sha1()
    h.update(method.encode("utf-8"))
    h.update(b"\0")
    h.update(str(sample_key or "").encode("utf-8"))
    h.update(b"\0")
    h.update(prior_dialogue.encode("utf-8"))
    h.update(b"\0")
    h.update(user_question.encode("utf-8"))
    return "pm_" + h.hexdigest()[:24]


def timestamp_for_turn(idx: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(1_700_000_000 + idx))


def jsonable_memories(memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(json.dumps(memories, ensure_ascii=False, default=str))
