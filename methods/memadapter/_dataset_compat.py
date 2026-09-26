"""Translate canonical release rows to the evaluator's internal shape."""

from __future__ import annotations

from typing import Any


def _dialogue_turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    turns = []
    for message in row["dialogue"]:
        role = str(message["role"]).strip().lower()
        speaker = "User" if role == "user" else "Assistant"
        turns.append(
            {
                "speaker": speaker,
                "content": f"{speaker}: {str(message['content']).strip()}",
                "is_query": False,
            }
        )
    return turns


def _extraction(row: dict[str, Any]) -> dict[str, Any]:
    memories = []
    for index, memory in enumerate(row["memory"]["items"], start=1):
        memories.append(
            {
                "memory_id": index,
                "memory_type": memory["type"],
                "content": memory["content"],
                "confidence": "high",
                "status": memory["status"],
            }
        )
    return {"memories": memories}


def _memory_item(row: dict[str, Any], status: str) -> str:
    for item in row["memory"]["items"]:
        if item["status"] == status:
            return str(item["content"])
    return ""


def to_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return rows already in evaluator shape unchanged; adapt canonical rows."""
    if "dialogue" not in row or "evaluation" not in row:
        return row

    task = row["task"]
    evaluation = row["evaluation"]
    rubric = evaluation["rubric"]
    metadata = row.get("metadata") or {}
    memories = row["memory"]["items"]
    adapted: dict[str, Any] = {
        "id": row["id"],
        "source_query_id": metadata.get("source_id", row["id"]),
        "question_type": metadata.get("subtype", task),
        "dialogue_context_turns": _dialogue_turns(row),
        "extraction": _extraction(row),
        "original_memory": memories[0]["content"] if memories else "",
        "session_meta": {"topic": metadata.get("topic", "")},
        "context_match_ok": True,
        "source_file": f"data/{task}.jsonl",
        "task_type": task,
    }

    if task == "personalized_memory_use":
        adapted.update(
            {
                "applicability": "applicable",
                "generated_question": {
                    "question": row["question"],
                    "reference_answer": evaluation["reference_answer"],
                    "preference_basis": rubric["expected_behavior"],
                    "why_correct": rubric["correct_reason"],
                },
            }
        )
    elif task == "valid_memory_selection":
        adapted.update(
            {
                "update_type": metadata.get("subtype", "update"),
                "old_preference": _memory_item(row, "outdated"),
                "updated_preference": _memory_item(row, "current"),
                "user_question": row["question"],
                "reference_answer": evaluation["reference_answer"],
                "updated_preference_basis": rubric["expected_behavior"],
                "old_preference_trap": rubric["failure_behavior"],
            }
        )
    elif task == "memory_evidence_conflict":
        adapted.update(
            {
                "applicability": "applicable",
                "correct_answer": evaluation["reference_answer"],
                "generated_question": {
                    "question": row["question"],
                    "reference_answer": evaluation["reference_answer"],
                    "preference_answer": evaluation["preference_aligned_answer"],
                    "open_answer_target": rubric["expected_behavior"],
                    "memory_interference": rubric["failure_behavior"],
                    "evidence_basis": rubric["supporting_evidence"],
                    "preference_supporting_evidence": rubric["preference_supporting_evidence"],
                    "decisive_evidence": rubric["decisive_evidence"],
                    "target_tradeoff": rubric["tradeoff"],
                    "evaluation_policy": rubric["evaluation_policy"],
                },
            }
        )
    elif task == "contextual_scope_control":
        adapted.update(
            {
                "applicability": "partially_applicable",
                "correct_behavior": "subject_aware_partial_application",
                "generated_question": {
                    "question": row["question"],
                    "reference_answer": evaluation["reference_answer"],
                    "expected_behavior": rubric["expected_behavior"],
                    "memory_trap": rubric["failure_behavior"],
                    "scope_conflict": rubric["scope_conflict"],
                    "scope_label": rubric["scope_label"],
                    "acceptable_memory_use": rubric["acceptable_memory_use"],
                    "required_scope_limit": rubric["scope_limits"],
                    "overgeneralization_failure": rubric["overgeneralization_failure"],
                    "underuse_failure": rubric["underuse_failure"],
                },
            }
        )
    elif task == "objective_fact_judgment":
        source = metadata.get("source") or {}
        open_rubric = {
            key: value
            for key, value in rubric.items()
            if key not in {"supporting_evidence", "failure_behavior"}
        }
        adapted.update(
            {
                "applicability": "applicable",
                "source_dataset": source.get("dataset"),
                "source_split": source.get("split"),
                "source_row_idx": source.get("row_index"),
                "source_question": source.get("question"),
                "generated_question": {
                    "question": row["question"],
                    "source_question": source.get("question"),
                    "source_url": source.get("url"),
                    "preference_answer": evaluation["preference_aligned_answer"],
                    "reference_answer": evaluation["reference_answer"],
                    "objective_fact_basis": rubric["supporting_evidence"],
                    "preference_interference": rubric["failure_behavior"],
                    "open_answer_rubric": open_rubric,
                },
            }
        )
    else:
        raise ValueError(f"Unsupported canonical task: {task!r}")

    return adapted
