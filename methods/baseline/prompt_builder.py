"""Build the official MemSyco baseline prompt from a frozen retrieval row."""

from __future__ import annotations

import importlib
import inspect
import os
from typing import Any


def build_official_baseline_system(task: str, row: dict[str, Any], model_name: str) -> str:
    module = importlib.import_module(
        "_objective_base" if task == "objective_fact_judgment" else f"task_{task}"
    )
    answer_system_prompt = getattr(module, "answer_system_prompt")
    kwargs: dict[str, Any] = {}
    if "context_label" in inspect.signature(answer_system_prompt).parameters:
        kwargs["context_label"] = "Retrieved memories from earlier conversation"
    return answer_system_prompt(
        model_name,
        os.environ.get("EVAL_CURRENT_DATE", "2025-06-01"),
        row["official_context_text"],
        **kwargs,
    )
