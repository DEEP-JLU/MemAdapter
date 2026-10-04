"""Reusable three-stage MemAdapter orchestration."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
PROMPT_DIR = ROOT / "prompts"
ABLATION_PROMPT_DIR = ROOT / "ablations" / "prompts"
CallModel = Callable[..., str]
StageComplete = Callable[[str, str], None]
StageStart = Callable[[str], None]
StageFinish = Callable[[str, float, bool], None]
def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def load_ablation_prompt(name: str) -> str:
    """Load a versioned prompt owned by an ablation configuration."""

    return (ABLATION_PROMPT_DIR / name).read_text(encoding="utf-8")


def load_prompt_parts(name: str) -> tuple[str, str]:
    """Split a prompt into system instructions and its user template."""

    text = load_prompt(name)
    marker = "USER PROMPT TEMPLATE" if "USER PROMPT TEMPLATE" in text else "INPUT FORMAT"
    if marker not in text:
        raise ValueError(f"Prompt {name} has no system/user separator")
    system_part, user_part = text.split(marker, maxsplit=1)
    system_prompt = system_part.removeprefix("SYSTEM PROMPT").strip()
    user_template = user_part.strip()
    if not system_prompt or not user_template:
        raise ValueError(f"Prompt {name} has an empty system prompt or user template")
    return system_prompt, user_template


def load_ablation_prompt_parts(name: str) -> tuple[str, str]:
    """Split an ablation prompt into its system prompt and user template."""

    text = load_ablation_prompt(name)
    marker = "USER PROMPT TEMPLATE" if "USER PROMPT TEMPLATE" in text else "INPUT FORMAT"
    if marker not in text:
        raise ValueError(f"Ablation prompt {name} has no system/user separator")
    system_part, user_part = text.split(marker, maxsplit=1)
    system_prompt = system_part.removeprefix("SYSTEM PROMPT").strip()
    user_template = user_part.strip()
    if not system_prompt or not user_template:
        raise ValueError(f"Ablation prompt {name} has an empty system prompt or user template")
    return system_prompt, user_template


def normalize_memories(memories: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in memories:
        if not isinstance(item, dict):
            raise TypeError("Each retrieved memory must be a JSON object")
        memory_id = item.get("memory_id", item.get("id"))
        memory_text = item.get("memory_text", item.get("content", item.get("text")))
        if memory_id is None or memory_text is None:
            raise ValueError("Each memory must contain memory_id and memory_text/content")
        normalized.append({"memory_id": str(memory_id), "memory_text": str(memory_text)})
    return normalized


def build_stage1_input(memories: list[dict[str, Any]]) -> str:
    return json.dumps(normalize_memories(memories), ensure_ascii=False, indent=2)


def build_stage2_input(*, current_query: str, memories: list[dict[str, Any]],
                       boundary_cards: dict[str, Any]) -> dict[str, str]:
    return {
        "CURRENT_QUERY": current_query,
        "RETRIEVED_MEMORIES": json.dumps(normalize_memories(memories), ensure_ascii=False, indent=2),
        "MEMORY_BOUNDARY_CARDS": json.dumps(boundary_cards, ensure_ascii=False, indent=2),
    }


def build_stage2_direct_input(*, current_query: str, memories: list[dict[str, Any]]) -> dict[str, str]:
    """Build Stage-2 input when the boundary-induction module is ablated.

    It intentionally exposes only raw retrieval plus task context.  This prevents
    Stage 2+3 from receiving an invented empty ``memory_boundary_cards`` object,
    which would otherwise look like a meaningful Stage-1 result.
    """

    return {
        "CURRENT_QUERY": current_query,
        "RETRIEVED_MEMORIES": json.dumps(normalize_memories(memories), ensure_ascii=False, indent=2),
    }


def build_stage3_input(*, current_query: str, memories: list[dict[str, Any]],
                       use_instructions: dict[str, Any]) -> dict[str, str]:
    return {
        "CURRENT_QUERY": current_query,
        "RETRIEVED_MEMORIES": json.dumps(normalize_memories(memories), ensure_ascii=False, indent=2),
        "MEMORY_USE_INSTRUCTIONS": json.dumps(use_instructions, ensure_ascii=False, indent=2),
    }


def build_stage3_direct_input(*, current_query: str, memories: list[dict[str, Any]]) -> dict[str, str]:
    """Build the direct generation input for the Stage-3-only ablation."""

    return {
        "CURRENT_QUERY": current_query,
        "RETRIEVED_MEMORIES": json.dumps(normalize_memories(memories), ensure_ascii=False, indent=2),
    }


def render_prompt(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    unresolved = re.findall(r"\{\{([A-Z_]+)\}\}", rendered)
    if unresolved:
        raise ValueError(f"Unresolved prompt placeholders: {unresolved}")
    return rendered


def parse_json_response(raw: str, *, stage: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"{stage} must return valid JSON: {exc}") from exc
        try:
            value = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as nested:
            raise ValueError(f"{stage} must return valid JSON: {nested}") from nested
    if not isinstance(value, dict):
        raise ValueError(f"{stage} must return a JSON object")
    return value


def run_json_stage(
    call: Callable[[], str],
    validate: Callable[[dict[str, Any]], None],
    *,
    stage: str,
    attempts: int = 3,
) -> tuple[str, dict[str, Any]]:
    """Retry the full model-call, parse, and schema-validation sequence."""

    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            raw = call()
            parsed = parse_json_response(raw, stage=stage)
            validate(parsed)
            return raw, parsed
        except (ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt < attempts:
                continue
    raise RuntimeError(
        f"{stage} failed after {attempts} attempts: {last_error}"
    ) from last_error


def _require_text(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def validate_stage1(result: dict[str, Any], memories: list[dict[str, Any]]) -> None:
    expected = [item["memory_id"] for item in normalize_memories(memories)]
    cards = result.get("memory_boundary_cards")
    if not isinstance(cards, list):
        raise ValueError("Stage 1 response is missing memory_boundary_cards")
    actual = [card.get("memory_id") for card in cards if isinstance(card, dict)]
    if actual != expected:
        raise ValueError("Stage 1 must preserve every memory_id in the original order")
    for card in cards:
        if not isinstance(card, dict):
            raise ValueError("Stage 1 memory boundary card must be an object")
        factors = card.get("key_factors")
        # The legacy full-pipeline prompt emits compact ``key_factors`` cards;
        # the method-aligned induction prompt used by stage1-baseline emits the
        # richer counterfactual schema. Accept either contract, but never an
        # unstructured card.
        if factors is not None:
            if not isinstance(factors, list) or any(
                not isinstance(value, str) or not value.strip() for value in factors
            ):
                raise ValueError("Stage 1 key_factors must be a list of strings")
        else:
            task_types = card.get("task_types")
            settings = card.get("counterfactual_settings")
            assessments = card.get("setting_use_assessments")
            if not isinstance(task_types, list) or not task_types:
                raise ValueError("Method-aligned Stage 1 requires non-empty task_types")
            if not isinstance(settings, list) or not settings:
                raise ValueError("Method-aligned Stage 1 requires non-empty counterfactual_settings")
            if not isinstance(assessments, list) or not assessments:
                raise ValueError("Method-aligned Stage 1 requires non-empty setting_use_assessments")
        rules = card.get("conditional_use_rules")
        if not isinstance(rules, list) or not rules or any(
            not isinstance(value, str) or not value.strip() for value in rules
        ):
            raise ValueError("Stage 1 conditional_use_rules must be a non-empty list of strings")
        _require_text(card.get("boundary_summary"), "Stage 1 boundary_summary")


def validate_stage2(result: dict[str, Any], memories: list[dict[str, Any]]) -> None:
    expected = [item["memory_id"] for item in normalize_memories(memories)]
    assignments = result.get("memory_use_instructions")
    if not isinstance(assignments, list):
        raise ValueError("Stage 2 response is missing memory_use_instructions")
    # Stage 2 may omit IDs when the prompt supplies a fixed positional schema.
    # Reattach the immutable retrieval IDs only when every positional entry is
    # present; no model-provided identifier is trusted or invented.
    if len(assignments) == len(expected) and all(isinstance(item, dict) and not item.get("memory_id") for item in assignments):
        assignments = [{**item, "memory_id": memory_id} for item, memory_id in zip(assignments, expected)]
        result["memory_use_instructions"] = assignments
    actual = [item.get("memory_id") for item in assignments if isinstance(item, dict)]
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("Stage 2 must preserve every memory_id exactly once")
    # The role assignments are independent per memory.  Canonicalize their
    # presentation order to the retrieval order, while rejecting omissions,
    # duplicates, and invented IDs above.
    by_id = {item["memory_id"]: item for item in assignments if isinstance(item, dict)}
    result["memory_use_instructions"] = [by_id[memory_id] for memory_id in expected]
    assignments = result["memory_use_instructions"]
    for item in assignments:
        if not isinstance(item, dict):
            raise ValueError("Stage 2 memory role assignment must be an object")
        _require_text(item.get("use_instruction"), "Stage 2 use_instruction")
        _require_text(item.get("rationale"), "Stage 2 rationale")


def run_three_stage(*, memories: list[dict[str, Any]], current_query: str,
                    call_model: CallModel,
                    stage_raw_cache: dict[str, str] | None = None,
                    on_stage_complete: StageComplete | None = None,
                    on_stage_start: StageStart | None = None,
                    on_stage_finish: StageFinish | None = None) -> dict[str, Any]:
    """Run the three stages and return parsed and raw intermediate results."""

    normalized = normalize_memories(memories)
    cached = stage_raw_cache or {}

    stage1_system, stage1_template = load_prompt_parts(
        "01_counterfactual_induction_method_aligned.txt"
    )
    stage1_user = render_prompt(stage1_template, {"RETRIEVED_MEMORIES": build_stage1_input(normalized)})
    stage1_raw = cached.get("stage1_raw", "")
    stage1_cached = bool(stage1_raw)
    stage_started = time.perf_counter()
    if on_stage_start:
        on_stage_start("stage1")
    try:
        if stage1_raw:
            boundary_cards = parse_json_response(stage1_raw, stage="Stage 1 cache")
            validate_stage1(boundary_cards, normalized)
        else:
            stage1_raw, boundary_cards = run_json_stage(
                lambda: call_model(stage1_system, stage1_user, json_mode=True, efficiency_stage="stage1"),
                lambda parsed: validate_stage1(parsed, normalized),
                stage="Stage 1",
            )
            if on_stage_complete:
                on_stage_complete("stage1_raw", stage1_raw)
    finally:
        if on_stage_finish:
            on_stage_finish("stage1", (time.perf_counter() - stage_started) * 1000, stage1_cached)

    stage2_system, stage2_template = load_prompt_parts(
        "02_context_aware_reflection_method_aligned.txt"
    )
    stage2_user = render_prompt(
        stage2_template,
        build_stage2_input(
            current_query=current_query,
            memories=normalized,
            boundary_cards=boundary_cards,
        ),
    )
    stage2_raw = cached.get("stage2_raw", "")
    stage2_cached = bool(stage2_raw)
    stage_started = time.perf_counter()
    if on_stage_start:
        on_stage_start("stage2")
    try:
        if stage2_raw:
            use_instructions = parse_json_response(stage2_raw, stage="Stage 2 cache")
            validate_stage2(use_instructions, normalized)
        else:
            stage2_raw, use_instructions = run_json_stage(
                lambda: call_model(stage2_system, stage2_user, json_mode=True, efficiency_stage="stage2"),
                lambda parsed: validate_stage2(parsed, normalized),
                stage="Stage 2",
            )
            if on_stage_complete:
                on_stage_complete("stage2_raw", stage2_raw)
    finally:
        if on_stage_finish:
            on_stage_finish("stage2", (time.perf_counter() - stage_started) * 1000, stage2_cached)

    stage3_system, stage3_template = load_prompt_parts("03_memory_use_guided_generation.txt")
    stage3_user = render_prompt(
        stage3_template,
        build_stage3_input(
            current_query=current_query,
            memories=normalized,
            use_instructions=use_instructions,
        ),
    )
    # Qwen stage-1/2-only studies intentionally stop here.  The normal and
    # ablation conditions leave this unset and keep the complete three-stage
    # contract unchanged.
    if os.environ.get("MEMADAPTER_SKIP_STAGE3", "").lower() in {"1", "true", "yes"}:
        return {
            "retrieved_memories": normalized,
            "memory_boundary_cards": boundary_cards,
            "memory_use_instructions": use_instructions,
            "memory_role_assignments": use_instructions,
            "stage1_raw": stage1_raw,
            "stage2_raw": stage2_raw,
            "stage3_raw": "",
            "final_answer": "",
        }

    stage3_raw = cached.get("stage3_raw", "")
    stage3_cached = bool(stage3_raw)
    stage_started = time.perf_counter()
    if on_stage_start:
        on_stage_start("stage3")
    try:
        if not stage3_raw:
            stage3_raw = call_model(stage3_system, stage3_user, efficiency_stage="stage3")
            if not stage3_raw.strip():
                raise ValueError("Stage 3 returned an empty final answer")
            if on_stage_complete:
                on_stage_complete("stage3_raw", stage3_raw)
    finally:
        if on_stage_finish:
            on_stage_finish("stage3", (time.perf_counter() - stage_started) * 1000, stage3_cached)
    return {
        "retrieved_memories": normalized,
        "memory_boundary_cards": boundary_cards,
        "memory_use_instructions": use_instructions,
        "memory_role_assignments": use_instructions,
        "stage1_raw": stage1_raw,
        "stage2_raw": stage2_raw,
        "stage3_raw": stage3_raw,
        "final_answer": stage3_raw.strip(),
    }


ABLATION_VARIANTS = (
    "stage1-baseline",
    "stage1-stage2-baseline",
)


def run_ablation(*, variant: str, memories: list[dict[str, Any]], current_query: str,
                 call_model: CallModel,
                 stage_raw_cache: dict[str, str] | None = None,
                 on_stage_complete: StageComplete | None = None,
                 on_stage_start: StageStart | None = None,
                 on_stage_finish: StageFinish | None = None) -> dict[str, Any]:
    """Run a pre-registered MemAdapter ablation without fabricating skipped stages.

    ``stage1-baseline`` runs only counterfactual induction, then hands its
    boundary cards to the direct baseline answer generator. ``stage1-stage2-
    baseline`` retains counterfactual induction and context-aware reflection,
    then hands the resulting use instructions to the direct baseline answer
    generator; neither condition calls the Stage-3 role-guided generator.
    Returned keys remain compatible with the result/judge pipeline; skipped
    stages are represented as empty values rather than synthetic output.
    """

    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unsupported ablation variant: {variant}")
    normalized = normalize_memories(memories)
    cached = stage_raw_cache or {}

    if variant == "stage1-baseline":
        # This condition deliberately uses the method-aligned Stage-1 prompt
        # requested for the isolated induction study, rather than the full
        # pipeline's boundary-induction prompt.
        stage1_system, stage1_template = load_prompt_parts(
            "01_counterfactual_induction_method_aligned.txt"
        )
        stage1_user = render_prompt(
            stage1_template, {"RETRIEVED_MEMORIES": build_stage1_input(normalized)}
        )
        stage1_raw = cached.get("stage1_raw", "")
        stage1_cached = bool(stage1_raw)
        started = time.perf_counter()
        if on_stage_start:
            on_stage_start("stage1")
        try:
            if stage1_raw:
                boundary_cards = parse_json_response(stage1_raw, stage="Stage 1 cache")
                validate_stage1(boundary_cards, normalized)
            else:
                stage1_raw, boundary_cards = run_json_stage(
                    lambda: call_model(
                        stage1_system,
                        stage1_user,
                        json_mode=True,
                        efficiency_stage="stage1",
                    ),
                    lambda parsed: validate_stage1(parsed, normalized),
                    stage="Stage 1",
                )
                if on_stage_complete:
                    on_stage_complete("stage1_raw", stage1_raw)
        finally:
            if on_stage_finish:
                on_stage_finish("stage1", (time.perf_counter() - started) * 1000, stage1_cached)

        baseline_system, baseline_template = load_ablation_prompt_parts(
            "stage1_then_baseline_generation.txt"
        )
        baseline_user = render_prompt(
            baseline_template,
            {
                **build_stage3_direct_input(
                    current_query=current_query,
                    memories=normalized,
                ),
                "MEMORY_BOUNDARY_CARDS": json.dumps(
                    boundary_cards, ensure_ascii=False, indent=2
                ),
            },
        )
        baseline_raw = cached.get("stage3_raw", "")
        baseline_cached = bool(baseline_raw)
        started = time.perf_counter()
        if on_stage_start:
            on_stage_start("baseline")
        try:
            if not baseline_raw:
                baseline_raw = call_model(
                    baseline_system, baseline_user, efficiency_stage="baseline"
                )
                if not baseline_raw.strip():
                    raise ValueError("Baseline answer generator returned an empty final answer")
                if on_stage_complete:
                    # Keep the cache key compatible with the generic result pipeline.
                    on_stage_complete("stage3_raw", baseline_raw)
        finally:
            if on_stage_finish:
                on_stage_finish("baseline", (time.perf_counter() - started) * 1000, baseline_cached)
        return {
            "retrieved_memories": normalized,
            "memory_boundary_cards": boundary_cards,
            "memory_use_instructions": [],
            "memory_role_assignments": [],
            "stage1_raw": stage1_raw,
            "stage2_raw": "",
            "stage3_raw": baseline_raw,
            "final_answer": baseline_raw.strip(),
        }

    if variant == "stage1-stage2-baseline":
        # Preserve the user-provided method-aligned prompts verbatim. Only the
        # transport between their structured outputs and the baseline answerer
        # is owned by this ablation.
        stage1_system, stage1_template = load_prompt_parts(
            "01_counterfactual_induction_method_aligned.txt"
        )
        stage1_user = render_prompt(
            stage1_template, {"RETRIEVED_MEMORIES": build_stage1_input(normalized)}
        )
        stage1_raw = cached.get("stage1_raw", "")
        stage1_cached = bool(stage1_raw)
        started = time.perf_counter()
        if on_stage_start:
            on_stage_start("stage1")
        try:
            if stage1_raw:
                boundary_cards = parse_json_response(stage1_raw, stage="Stage 1 cache")
                validate_stage1(boundary_cards, normalized)
            else:
                stage1_raw, boundary_cards = run_json_stage(
                    lambda: call_model(
                        stage1_system,
                        stage1_user,
                        json_mode=True,
                        efficiency_stage="stage1",
                    ),
                    lambda parsed: validate_stage1(parsed, normalized),
                    stage="Stage 1",
                )
                if on_stage_complete:
                    on_stage_complete("stage1_raw", stage1_raw)
        finally:
            if on_stage_finish:
                on_stage_finish("stage1", (time.perf_counter() - started) * 1000, stage1_cached)

        stage2_system, stage2_template = load_prompt_parts(
            "02_context_aware_reflection_method_aligned.txt"
        )
        stage2_user = render_prompt(
            stage2_template,
            build_stage2_input(
                current_query=current_query,
                memories=normalized,
                boundary_cards=boundary_cards,
            ),
        )
        stage2_raw = cached.get("stage2_raw", "")
        stage2_cached = bool(stage2_raw)
        started = time.perf_counter()
        if on_stage_start:
            on_stage_start("stage2")
        try:
            if stage2_raw:
                use_instructions = parse_json_response(stage2_raw, stage="Stage 2 cache")
                validate_stage2(use_instructions, normalized)
            else:
                stage2_raw, use_instructions = run_json_stage(
                    lambda: call_model(
                        stage2_system,
                        stage2_user,
                        json_mode=True,
                        efficiency_stage="stage2",
                    ),
                    lambda parsed: validate_stage2(parsed, normalized),
                    stage="Stage 2",
                )
                if on_stage_complete:
                    on_stage_complete("stage2_raw", stage2_raw)
        finally:
            if on_stage_finish:
                on_stage_finish("stage2", (time.perf_counter() - started) * 1000, stage2_cached)

        baseline_system, baseline_template = load_ablation_prompt_parts(
            "stage1_stage2_then_baseline_generation.txt"
        )
        baseline_user = render_prompt(
            baseline_template,
            build_stage3_input(
                current_query=current_query,
                memories=normalized,
                use_instructions=use_instructions,
            ),
        )
        baseline_raw = cached.get("stage3_raw", "")
        baseline_cached = bool(baseline_raw)
        started = time.perf_counter()
        if on_stage_start:
            on_stage_start("baseline")
        try:
            if not baseline_raw:
                baseline_raw = call_model(
                    baseline_system, baseline_user, efficiency_stage="baseline"
                )
                if not baseline_raw.strip():
                    raise ValueError("Baseline answer generator returned an empty final answer")
                if on_stage_complete:
                    on_stage_complete("stage3_raw", baseline_raw)
        finally:
            if on_stage_finish:
                on_stage_finish("baseline", (time.perf_counter() - started) * 1000, baseline_cached)
        return {
            "retrieved_memories": normalized,
            "memory_boundary_cards": boundary_cards,
            "memory_use_instructions": use_instructions,
            "memory_role_assignments": use_instructions,
            "stage1_raw": stage1_raw,
            "stage2_raw": stage2_raw,
            "stage3_raw": baseline_raw,
            "final_answer": baseline_raw.strip(),
        }

    def execute_final(system: str, user: str) -> tuple[str, str]:
        stage3_raw = cached.get("stage3_raw", "")
        stage3_cached = bool(stage3_raw)
        started = time.perf_counter()
        if on_stage_start:
            on_stage_start("stage3")
        try:
            if not stage3_raw:
                stage3_raw = call_model(system, user, efficiency_stage="stage3")
                if not stage3_raw.strip():
                    raise ValueError("Stage 3 returned an empty final answer")
                if on_stage_complete:
                    on_stage_complete("stage3_raw", stage3_raw)
        finally:
            if on_stage_finish:
                on_stage_finish("stage3", (time.perf_counter() - started) * 1000, stage3_cached)
        return stage3_raw, stage3_raw.strip()

    if variant == "stage3-only":
        stage3_system, stage3_template = load_ablation_prompt_parts("stage3_only_direct_generation.txt")
        stage3_user = render_prompt(
            stage3_template,
            build_stage3_direct_input(
                current_query=current_query,
                memories=normalized,
            ),
        )
        stage3_raw, final_answer = execute_final(stage3_system, stage3_user)
        return {
            "retrieved_memories": normalized,
            "memory_boundary_cards": [],
            "memory_use_instructions": [],
            "memory_role_assignments": [],
            "stage1_raw": "",
            "stage2_raw": "",
            "stage3_raw": stage3_raw,
            "final_answer": final_answer,
        }

    stage2_system, stage2_template = load_ablation_prompt_parts("stage2_direct_reflection.txt")
    stage2_user = render_prompt(
        stage2_template,
        build_stage2_direct_input(
            current_query=current_query,
            memories=normalized,
        ),
    )
    stage2_raw = cached.get("stage2_raw", "")
    stage2_cached = bool(stage2_raw)
    started = time.perf_counter()
    if on_stage_start:
        on_stage_start("stage2")
    try:
        if stage2_raw:
            use_instructions = parse_json_response(stage2_raw, stage="Stage 2 cache")
            validate_stage2(use_instructions, normalized)
        else:
            stage2_raw, use_instructions = run_json_stage(
                lambda: call_model(stage2_system, stage2_user, json_mode=True, efficiency_stage="stage2"),
                lambda parsed: validate_stage2(parsed, normalized),
                stage="Stage 2",
            )
            if on_stage_complete:
                on_stage_complete("stage2_raw", stage2_raw)
    finally:
        if on_stage_finish:
            on_stage_finish("stage2", (time.perf_counter() - started) * 1000, stage2_cached)

    stage3_system, stage3_template = load_ablation_prompt_parts("stage3_after_direct_reflection.txt")
    stage3_user = render_prompt(
        stage3_template,
        build_stage3_input(
            current_query=current_query,
            memories=normalized,
            use_instructions=use_instructions,
        ),
    )
    stage3_raw, final_answer = execute_final(stage3_system, stage3_user)
    return {
        "retrieved_memories": normalized,
        "memory_boundary_cards": [],
        "memory_use_instructions": use_instructions,
        "memory_role_assignments": use_instructions,
        "stage1_raw": "",
        "stage2_raw": stage2_raw,
        "stage3_raw": stage3_raw,
        "final_answer": final_answer,
    }


__all__ = [
    "ABLATION_VARIANTS", "CallModel", "build_stage1_input", "build_stage2_direct_input",
    "build_stage2_input", "build_stage3_direct_input", "build_stage3_input", "load_ablation_prompt",
    "load_ablation_prompt_parts", "load_prompt", "load_prompt_parts", "normalize_memories",
    "parse_json_response", "render_prompt", "run_ablation", "run_json_stage", "run_three_stage",
    "validate_stage1", "validate_stage2",
]
