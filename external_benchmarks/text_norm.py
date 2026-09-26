"""Text normalization for fidelity checks.

Used to decide whether a retrieve-then-compare check should count as a match.
Deliberately conservative: whitespace differences are collapsed because stores
routinely re-wrap text, but case and punctuation are preserved, so a store that
paraphrased a memory cannot pass as verbatim.
"""

from __future__ import annotations

import re
from pathlib import Path

_WHITESPACE = re.compile(r"\s+")


def read_prompt_file(path: Path) -> str:
    """Read a prompt with its line terminators intact.

    ``Path.read_text`` normalises CRLF to LF, so a prompt read that way is not the
    prompt the file holds -- and the run config (via ``sha256_file``) hashes the
    file's bytes. The two disagree exactly when it matters, on the record meant to
    prove which prompt was used. Reading with ``newline=""`` makes the string the
    file's own characters, so its hash and the recorded hash are the same number.
    MemTrapBench's shipped files are CRLF (verified byte-identical to upstream by
    ``audit_rubrics``), so this preserves them rather than silently rewriting them.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def normalize_for_match(text: str) -> str:
    """Collapse whitespace runs and strip, preserving case and punctuation."""
    return _WHITESPACE.sub(" ", str(text or "")).strip()


def truncate(text: str, limit: int = 160, *, ellipsis: str = "...") -> str:
    """Shorten for logs and reports without cutting mid-word where avoidable."""
    text = normalize_for_match(text)
    if len(text) <= limit:
        return text
    return text[: limit - len(ellipsis)].rstrip() + ellipsis
