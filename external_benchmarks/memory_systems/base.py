from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BaselineEvalConfig:
    method: str
    top_k: int = 10
    config_path: Path | None = None
    save_root: Path = REPO_ROOT / "output_data" / "baseline_eval_memory"
    api_key: str | None = None
    base_url: str | None = None
    llm_model: str | None = None
    embedding_model: str | None = None
    embedding_dims: int | None = None
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None


@dataclass(frozen=True)
class BaselineContext:
    context_text: str
    retrieved_memories: list[dict[str, Any]]
    user_id: str
    save_dir: str
    method: str
    top_k: int
