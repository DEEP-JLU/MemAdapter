"""MemTrapBench judge: fill the subset's rubric, parse its dimensions.

Two details are copied from upstream rather than reinvented, because both are visible
in the prompt that is actually sent:

*   **the placeholder family** (``runners/eval/eval_common.py::DATASET_CONFIGS``).
    ``poison`` writes ``{query}``/``{response}`` where the other rubrics write
    ``{final_trigger}``/``{assistant_response}``, so the filler is selected by the
    subset's ``judge_type`` instead of one uniform set of names.
*   **the substitution order**, which is observable when a substituted value itself
    contains a brace token -- a matrix problem's ``{ij}`` subscripts, a LaTeX body.
    Upstream replaces ``{final_trigger}`` first, then the poison/number-game fields,
    and ``{assistant_response}`` last, so a token inside the model's own answer stays
    literal. Reordering these changes the prompt for exactly the samples where it is
    hardest to notice.

The judge is sent a **single user message** with no system turn, as upstream does.
``ModelClient`` always sends system + user, so the system turn is empty here too; the
MemAdapter arm's judge is invoked the same way, which keeps the two arms' judge
protocols identical even though neither matches upstream's message list exactly.
"""

from __future__ import annotations

import re
from typing import Any

from .. import task_registry
from .common import (
    declared_dimensions,
    extract_json_memtrap,
    score_memtrap_dimensions,
)

#: Placeholders each rubric family declares. Validated against the *raw* template
#: before substitution: scanning the filled prompt would flag brace tokens that came
#: from the injected content as unresolved fields, which is the bug upstream's own
#: comment describes.
FILLABLE = {
    "shared": {"{final_trigger}", "{assistant_response}"},
    "number_game": {"{final_trigger}", "{assistant_response}", "{gold_standard}"},
    "poison": {
        "{final_trigger}",
        "{assistant_response}",
        "{query}",
        "{response}",
        "{poisoned_fact}",
        "{objective_truth}",
    },
}

#: Only lowercase-and-underscore tokens match, as upstream's own scan does; that is
#: what keeps ``{ij}`` and ``{aligned}`` in the injected text from being read as
#: unfilled fields.
PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")


def build_judge_prompt(
    template: str,
    benchmark_row: dict[str, Any],
    response: str,
    judge_type: str,
    *,
    fallback_trigger: str = "",
) -> str:
    """Fill one rubric. Mirrors upstream ``eval_common.build_judge_prompt`` exactly."""
    final_trigger = str(benchmark_row.get("final_trigger") or fallback_trigger or "")
    fillable = FILLABLE.get(judge_type, FILLABLE["shared"])
    unresolved = sorted(set(PLACEHOLDER_RE.findall(template)) - fillable)
    if unresolved:
        raise ValueError(f"Unresolved judge placeholders: {', '.join(unresolved)}")

    prompt = template.replace("{final_trigger}", final_trigger)
    if judge_type == "poison":
        prompt = (
            prompt.replace("{query}", final_trigger)
            .replace("{poisoned_fact}", str(benchmark_row.get("poisoned_fact") or "N/A"))
            .replace(
                "{objective_truth}", str(benchmark_row.get("objective_truth") or "N/A")
            )
            .replace("{response}", response)
        )
    elif judge_type == "number_game":
        prompt = prompt.replace(
            "{gold_standard}", str(benchmark_row.get("gold_standard") or "")
        )
    return prompt.replace("{assistant_response}", response)


def build_prompts(record: dict[str, Any], template: str) -> tuple[str, str]:
    """``(system, user)`` for one judgement: empty system, rubric as the user turn."""
    spec = task_registry.spec(str(record["task"]))
    user = build_judge_prompt(
        template,
        record.get("benchmark_row") or {},
        str(record.get("final_answer") or ""),
        spec.judge_type,
        fallback_trigger=str(record.get("current_request") or ""),
    )
    return "", user


def interpret(text: str, template: str) -> tuple[dict[str, Any], dict[str, int], list[str]]:
    """Parse one judge reply into ``(parsed, {dimension: score}, problems)``.

    The fence-stripping step is upstream's, deliberately without the ``<think>``
    handling PersistBench needs: MemTrapBench's own parser gives up on a reply it
    cannot read, and treating that as anything other than a failed judgement would
    quietly improve our numbers relative to the paper's.
    """
    expected = declared_dimensions(template)
    parsed, error = extract_json_memtrap(text)
    if parsed is None:
        return {"raw_judge": text}, {}, [error]
    scores, problems = score_memtrap_dimensions(parsed, expected)
    return parsed, scores, problems
