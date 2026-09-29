from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

from .base import BaselineContext, BaselineEvalConfig
from .common import (
    format_retrieved_memories,
    jsonable_memories,
    parse_dialogue_to_messages,
    retry_embedding_query,
    sample_user_id,
    timestamp_for_turn,
)
from .vendor_paths import prepare_lightmem_import_path


METHOD = "LightMem"


def _logging_enabled() -> bool:
    return os.getenv("LIGHTMEM_LOG", "").lower() in {"1", "true", "yes", "on"} or os.getenv(
        "MEMORY_METHOD_LOG", ""
    ).lower() in {"1", "true", "yes", "on"}


def _configure_logging() -> None:
    if not _logging_enabled():
        return
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("LightMemory").setLevel(logging.INFO)


def build_context(
    prior_dialogue: str,
    user_question: str,
    eval_config: BaselineEvalConfig,
    *,
    sample_key: str | int | None = None,
) -> BaselineContext:
    prepare_lightmem_import_path()
    _configure_logging()

    from lightmem.memory.lightmem import LightMemory  # type: ignore

    method = eval_config.method
    user_id = sample_user_id(method, sample_key, prior_dialogue, user_question)
    save_dir = str((eval_config.save_root / method / user_id).resolve())
    config = _load_native_lightmem_config(
        eval_config, user_id=user_id, save_dir=save_dir)
    lightmem = LightMemory.from_config(config)
    messages = parse_dialogue_to_messages(prior_dialogue)

    if _is_full_method(method):
        ingest_full_lightmem(lightmem, messages)
    retrieved_strings = retry_embedding_query(
        lambda: lightmem.retrieve(user_question, limit=eval_config.top_k),
        method=method,
    )
    retrieved = [
        {"content": text, "metadata": {"rank": i}, "used_content": text}
        for i, text in enumerate(retrieved_strings, start=1)
    ]
    return BaselineContext(
        context_text=format_retrieved_memories(retrieved),
        retrieved_memories=jsonable_memories(retrieved),
        user_id=user_id,
        save_dir=save_dir,
        method=method,
        top_k=eval_config.top_k,
    )


def _is_full_method(method: str) -> bool:
    return method == METHOD


def retrieval_only_lightmem_config(config: dict[str, Any]) -> dict[str, Any]:
    """Strip ingest-only components so disk hits avoid reloading local models."""
    slim = copy.deepcopy(config)
    slim["pre_compress"] = False
    slim.pop("pre_compressor", None)
    slim["topic_segment"] = False
    slim.pop("topic_segmenter", None)
    slim["precomp_topic_shared"] = False
    slim["metadata_generate"] = False
    slim["text_summary"] = False
    slim.pop("summary_retriever", None)
    logging_cfg = slim.setdefault("logging", {})
    logging_cfg["level"] = "WARNING"
    logging_cfg["file_enabled"] = False
    logging_cfg["console_enabled"] = False
    return slim


def _load_native_lightmem_config(eval_config: BaselineEvalConfig, *, user_id: str, save_dir: str) -> dict[str, Any]:
    if eval_config.config_path is not None:
        with Path(eval_config.config_path).open("r", encoding="utf-8") as f:
            config = json.load(f)
        if not isinstance(config, dict):
            raise ValueError(
                f"LightMem config must be a JSON object: {eval_config.config_path}")
    else:
        embedding_model = eval_config.embedding_model or os.environ.get(
            "MEMORY_EMBEDDING_MODEL",
            "BAAI/bge-m3",
        )
        embedding_dims = int(eval_config.embedding_dims or os.environ.get(
            "MEMORY_EMBEDDING_DIMS", "1536"))
        embedder_provider = os.environ.get("MEMORY_EMBEDDER_PROVIDER", "huggingface")
        embedding_api_key = eval_config.embedding_api_key or eval_config.api_key or os.environ.get(
            "OPENAI_API_KEY")
        if not embedding_api_key:
            embedder_provider = os.environ.get(
                "MEMORY_EMBEDDER_PROVIDER", "huggingface")
            embedding_model = eval_config.embedding_model or os.environ.get(
                "MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            embedding_dims = int(eval_config.embedding_dims or os.environ.get(
                "MEMORY_EMBEDDING_DIMS", "384"))

        text_embedder_configs: dict[str, Any] = {
            "model": embedding_model,
            "embedding_dims": embedding_dims,
        }
        if embedder_provider == "huggingface":
            text_embedder_configs["model_kwargs"] = {
                "device": os.environ.get("MEMORY_EMBEDDING_DEVICE", "cpu")}

        config = _default_full_lightmem_config(
            eval_config,
            user_id=user_id,
            save_dir=save_dir,
            embedder_provider=embedder_provider,
            text_embedder_configs=text_embedder_configs,
            embedding_dims=embedding_dims,
        )

    _patch_native_lightmem_config(
        config, eval_config, user_id=user_id, save_dir=save_dir)
    return config


def _default_raw_lightmem_config(
    eval_config: BaselineEvalConfig,
    *,
    user_id: str,
    save_dir: str,
    embedder_provider: str,
    text_embedder_configs: dict[str, Any],
    embedding_dims: int,
) -> dict[str, Any]:
    return {
        "pre_compress": False,
        "topic_segment": False,
        "messages_use": "hybrid",
        "metadata_generate": False,
        "text_summary": False,
        "memory_manager": _memory_manager_config(eval_config),
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": embedder_provider,
            "configs": text_embedder_configs,
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": _qdrant_retriever_config(user_id, save_dir, embedding_dims),
        "update": "offline",
        "logging": {
            "level": "WARNING",
            "file_enabled": False,
            "console_enabled": False,
        },
    }


def _default_full_lightmem_config(
    eval_config: BaselineEvalConfig,
    *,
    user_id: str,
    save_dir: str,
    embedder_provider: str,
    text_embedder_configs: dict[str, Any],
    embedding_dims: int,
) -> dict[str, Any]:
    llmlingua_model = os.environ.get(
        "LIGHTMEM_LLMLINGUA_MODEL_PATH",
        os.environ.get("LLMLINGUA_MODEL_PATH",
                       "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"),
    )
    llmlingua_device = os.environ.get("LIGHTMEM_LLMLINGUA_DEVICE", os.environ.get(
        "MEMORY_EMBEDDING_DEVICE", "cuda:1"))

    config = {
        "pre_compress": _env_bool("LIGHTMEM_FULL_PRE_COMPRESS", True),
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": llmlingua_model,
                    "device_map": llmlingua_device,
                    "use_llmlingua2": True,
                },
            },
        },
        "topic_segment": True,
        "precomp_topic_shared": True,
        "topic_segmenter": {
            "model_name": "llmlingua-2",
            "configs": {
                "model_name": llmlingua_model,
                "device_map": llmlingua_device,
            },
        },
        "messages_use": os.environ.get("LIGHTMEM_FULL_MESSAGES_USE", "hybrid"),
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": _memory_manager_config(eval_config),
        "extract_threshold": float(os.environ.get("LIGHTMEM_FULL_EXTRACT_THRESHOLD", "0.1")),
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": embedder_provider,
            "configs": text_embedder_configs,
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": _qdrant_retriever_config(user_id, save_dir, embedding_dims),
        "summary_retriever": _qdrant_retriever_config(f"{user_id}_summary", str(Path(save_dir) / "summaries"), embedding_dims),
        "update": "offline",
        "logging": {
            "level": "WARNING",
            "file_enabled": False,
            "console_enabled": False,
        },
    }
    if not config["pre_compress"]:
        config.pop("pre_compressor", None)
        config["precomp_topic_shared"] = False
    return config


def _memory_manager_config(eval_config: BaselineEvalConfig) -> dict[str, Any]:
    model = eval_config.llm_model or os.environ.get(
        "MEMORY_LLM_MODEL", "DeepSeek-V4-Flash")
    manager_provider = os.environ.get("LIGHTMEM_MEMORY_MANAGER", "deepseek")
    configs: dict[str, Any] = {
        "model": model,
        "api_key": eval_config.api_key or os.environ.get("OPENAI_API_KEY", ""),
        "max_tokens": int(os.environ.get("LIGHTMEM_MANAGER_MAX_TOKENS", "4096")),
        "temperature": float(os.environ.get("LIGHTMEM_MANAGER_TEMPERATURE", "0.2")),
        "thinking": False,
    }
    base_url = eval_config.base_url or os.environ.get(
        "OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    if manager_provider == "deepseek":
        configs["deepseek_base_url"] = base_url
    else:
        configs["openai_base_url"] = base_url
    return {
        "model_name": manager_provider,
        "configs": configs,
    }


def _qdrant_retriever_config(collection_name: str, path: str, embedding_dims: int) -> dict[str, Any]:
    return {
        "model_name": "qdrant",
        "configs": {
            "collection_name": collection_name,
            "embedding_model_dims": embedding_dims,
            "path": path,
            "on_disk": True,
        },
    }


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _patch_native_lightmem_config(
    config: dict[str, Any],
    eval_config: BaselineEvalConfig,
    *,
    user_id: str,
    save_dir: str,
) -> None:
    if _logging_enabled():
        logging_cfg = config.setdefault("logging", {})
        logging_cfg["level"] = "INFO"
        logging_cfg["console_enabled"] = True
        logging_cfg["console_level"] = "INFO"
        logging_cfg["file_enabled"] = False
        logging_cfg["suppress_loggers"] = []

    retriever = config.setdefault(
        "embedding_retriever", {}).setdefault("configs", {})
    retriever["collection_name"] = user_id
    retriever["path"] = save_dir
    retriever.setdefault("on_disk", True)

    manager = config.setdefault("memory_manager", {})
    manager_cfg = manager.setdefault("configs", {})
    if eval_config.llm_model:
        manager_cfg["model"] = eval_config.llm_model
    api_key = (eval_config.api_key or "").strip()
    base_url = (eval_config.base_url or "").strip()
    if api_key:
        manager_cfg["api_key"] = api_key
    if base_url:
        if manager.get("model_name") == "deepseek":
            manager_cfg["deepseek_base_url"] = base_url
        else:
            manager_cfg["openai_base_url"] = base_url

    if eval_config.embedding_model or eval_config.embedding_dims is not None:
        text_embedder = config.setdefault(
            "text_embedder", {}).setdefault("configs", {})
        if eval_config.embedding_model:
            text_embedder["model"] = eval_config.embedding_model
        if eval_config.embedding_dims is not None:
            text_embedder["embedding_dims"] = eval_config.embedding_dims
            retriever["embedding_model_dims"] = eval_config.embedding_dims
            summary = config.get("summary_retriever")
            if isinstance(summary, dict):
                summary.setdefault("configs", {})[
                    "embedding_model_dims"] = eval_config.embedding_dims

    embedding_api_key = (eval_config.embedding_api_key or "").strip()
    embedding_base_url = (eval_config.embedding_base_url or "").strip()
    if embedding_api_key or embedding_base_url:
        text_embedder_parent = config.setdefault("text_embedder", {})
        text_embedder = text_embedder_parent.setdefault("configs", {})
        if embedding_api_key:
            text_embedder["api_key"] = embedding_api_key
        if embedding_base_url:
            provider = text_embedder_parent.get("model_name")
            if provider == "huggingface":
                text_embedder["huggingface_base_url"] = embedding_base_url
            else:
                text_embedder["openai_base_url"] = embedding_base_url

    summary = config.get("summary_retriever")
    if isinstance(summary, dict):
        summary_configs = summary.setdefault("configs", {})
        summary_configs.setdefault("collection_name", f"{user_id}_summary")
        summary_configs.setdefault("path", str(Path(save_dir) / "summaries"))
        summary_configs.setdefault("on_disk", True)


def ingest_raw_dialogue_lightmem(lightmem: Any, messages: list[dict[str, str]]) -> None:
    from lightmem.memory.utils import MemoryEntry  # type: ignore

    entries = []
    for idx, message in enumerate(messages):
        timestamp = timestamp_for_turn(idx)
        role = message.get("role", "user")
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        entries.append(
            MemoryEntry(
                time_stamp=timestamp,
                float_time_stamp=float(1_700_000_000 + idx),
                weekday="Tue",
                category="dialogue",
                subcategory=role,
                memory_class="raw_dialogue",
                memory=f"{role}: {content}",
                original_memory=content,
                compressed_memory=content,
                speaker_id=role,
                speaker_name=role,
            )
        )
    if entries:
        lightmem.offline_update(entries)


def ingest_full_lightmem(lightmem: Any, messages: list[dict[str, str]]) -> None:
    turn_batches = _native_lightmem_turn_batches(messages)
    if not turn_batches:
        return

    for turn_messages in turn_batches:
        lightmem.add_memory(messages=turn_messages,
                            force_segment=True, force_extract=True)
    lightmem.construct_update_queue_all_entries(
        top_k=int(os.environ.get("LIGHTMEM_FULL_UPDATE_TOP_K", "20")),
        keep_top_n=int(os.environ.get(
            "LIGHTMEM_FULL_UPDATE_KEEP_TOP_N", "10")),
        max_workers=int(os.environ.get(
            "LIGHTMEM_FULL_UPDATE_QUEUE_WORKERS", "8")),
    )
    lightmem.offline_update_all_entries(
        score_threshold=float(os.environ.get(
            "LIGHTMEM_FULL_UPDATE_SCORE_THRESHOLD", "0.8")),
        max_workers=int(os.environ.get(
            "LIGHTMEM_FULL_OFFLINE_UPDATE_WORKERS", "5")),
    )


def _native_lightmem_turn_batches(messages: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    batches: list[list[dict[str, str]]] = []
    pending_user: dict[str, str] | None = None

    def flush_with_assistant(assistant: dict[str, str] | None, idx: int) -> None:
        nonlocal pending_user
        if pending_user is None:
            return
        timestamp = _native_lightmem_timestamp(idx)
        user_message = dict(pending_user)
        user_message["time_stamp"] = timestamp
        if assistant is None:
            assistant_message = {
                "role": "assistant",
                "content": "",
                "time_stamp": timestamp,
            }
        else:
            assistant_message = dict(assistant)
            assistant_message["time_stamp"] = timestamp
        batches.append([user_message, assistant_message])
        pending_user = None

    for idx, message in enumerate(messages):
        role = str(message.get("role") or "user").lower()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        normalized = {"role": role, "content": content}
        if role == "assistant":
            if pending_user is None:
                pending_user = {"role": "user", "content": ""}
            flush_with_assistant(normalized, idx)
            continue
        if pending_user is not None:
            flush_with_assistant(None, idx)
        pending_user = {"role": "user", "content": content}

    flush_with_assistant(None, len(messages))
    return batches


def _native_lightmem_timestamp(idx: int) -> str:
    import time

    return time.strftime("%Y/%m/%d (%a) %H:%M", time.gmtime(1_700_000_000 + idx * 60))
