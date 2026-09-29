from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaselineContext, BaselineEvalConfig
from .common import jsonable_memories, parse_dialogue_to_messages


# ---------------------------------------------------------------------------
# MemoryBank-SiliconFriend baseline adapter.
#
# 复刻 MemoryBank 的记忆机制并对接本仓库的评测基础设施：
#   1. 把 prior dialogue 切成 (query, response) 对，归到一个"日期"下，组成 MemoryBank
#      的 history 结构。
#   2. 用 LLM 生成 MemoryBank 的多级总结：per-date summary / personality，以及
#      overall_history / overall_personality（复用 MemoryBank 仓库里的 prompt 函数）。
#   3. 按 MemoryBank 的 JsonMemoryLoader 方式构建检索文档（每轮对话 + 每日摘要），
#      用 OpenAI 兼容的 embedding 服务向量化，做 top-k 余弦检索并按日期聚合
#      （对齐 LocalMemoryRetrieval.search_memory）。
#   4. 把"检索到的回忆 + 整体回忆总结 + 用户人格/回复策略"组合为注入答案模型的上下文
#      （对齐 build_prompt_with_search_memory_*）。
#
# 与 A-MEM / MemZero 一致：LLM 与 embedding 均直连用户提供的 MEMORY_* 端点
# （MEMORY_BASE_URL / MEMORY_API_KEY 用于 LLM，MEMORY_EMBEDDING_* 用于 embedding），
# 不依赖任何专用网关。每段 prior dialogue 的记忆只构建一次，落盘到
# save_root/MemoryBank/<user_id> 并在进程内缓存，后续同一对话的不同问题复用。
# ---------------------------------------------------------------------------


METHOD = "MemoryBank"

MEMORYBANK_DIR = Path(__file__).resolve().parent / "memorybank" / "vendor"
MEMORYBANK_BANK_DIR = MEMORYBANK_DIR / "memory_bank"

DEFAULT_USER_NAME = "User"
DEFAULT_BOOT_NAME = "AI"
DEFAULT_DATE = "2023-11-14"
DEFAULT_LLM_MODEL = "DeepSeek-V4-Flash"
DEFAULT_EMBEDDING_MODEL = "bge-m3"
DEFAULT_EMBED_BATCH = 64
DEFAULT_REQUEST_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3
# 公平性设置：所有 baseline 的 LLM 摄入调用统一 max_tokens / temperature。
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.2
_COMPLETE_MARKER = ".memory_complete.json"
_CACHE_VERSION = 1

_cache_guard = threading.Lock()
_inproc_cache: dict[str, "_MemoryIndex"] = {}


@dataclass
class _MemoryIndex:
    user_id: str
    save_dir: str
    docs: list[dict[str, Any]]
    vectors: Any  # numpy.ndarray (n_docs, dim), L2-normalized
    overall_history: str
    overall_personality: str


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------
def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _language() -> str:
    lang = (_env("MEMORYBANK_LANGUAGE") or "en").lower()
    return "cn" if lang in {"cn", "zh", "zh-cn", "chinese"} else "en"


def _summary_enabled() -> bool:
    return (_env("MEMORYBANK_DISABLE_SUMMARY") or "").lower() not in {"1", "true", "yes", "on"}


def _request_timeout() -> float:
    raw = _env("MEMORYBANK_REQUEST_TIMEOUT", "MEMORY_REQUEST_TIMEOUT")
    return float(raw) if raw else DEFAULT_REQUEST_TIMEOUT


def _max_retries() -> int:
    raw = _env("MEMORYBANK_MAX_RETRIES", "MEMORY_API_MAX_RETRIES")
    return max(1, int(raw)) if raw else DEFAULT_MAX_RETRIES


def _resolve_llm_model(eval_config: BaselineEvalConfig) -> str:
    return eval_config.llm_model or _env("MEMORY_LLM_MODEL") or DEFAULT_LLM_MODEL


def _resolve_embedding_model(eval_config: BaselineEvalConfig) -> str:
    return eval_config.embedding_model or _env("MEMORY_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL


def _resolve_embedding_dims(eval_config: BaselineEvalConfig) -> int | None:
    if eval_config.embedding_dims is not None:
        return int(eval_config.embedding_dims)
    raw = _env("MEMORY_EMBEDDING_DIMS")
    return int(raw) if raw else None


# ---------------------------------------------------------------------------
# OpenAI 兼容客户端（直连 LLM + embedding 端点，使用 MEMORY_* 凭据）
# ---------------------------------------------------------------------------
def _make_clients(eval_config: BaselineEvalConfig) -> tuple[Any, Any]:
    """返回 (chat_client, embed_client)，均直连 OpenAI 兼容端点。

    使用 ``eval_config.api_key`` / ``base_url``（即 ``MEMORY_API_KEY`` /
    ``MEMORY_BASE_URL``）构建 chat client，用 embedding 侧的 key/url 构建 embed
    client。记忆基线与 mem0 / A-MEM 一样直接走用户提供的
    ``MEMORY_*`` 端点。
    """
    import openai  # noqa: PLC0415

    timeout = _request_timeout()
    max_retries = _max_retries()

    chat_api_key = eval_config.api_key or os.environ.get("OPENAI_API_KEY") or ""
    chat_base_url = eval_config.base_url or os.environ.get("OPENAI_API_BASE") or None
    chat_client = openai.OpenAI(
        api_key=chat_api_key,
        base_url=chat_base_url,
        timeout=timeout,
        max_retries=max_retries,
    )

    embed_api_key = eval_config.embedding_api_key or eval_config.api_key or "sk-local"
    embed_base_url = eval_config.embedding_base_url or eval_config.base_url or None
    embed_client = openai.OpenAI(
        api_key=embed_api_key,
        base_url=embed_base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    return chat_client, embed_client


def _chat_summarize(chat_client: Any, model: str, prompt: str, language: str) -> str:
    """复刻 MemoryBank 的 LLMClientSimple.generate_text_simple 的消息结构。"""
    if language == "cn":
        messages = [
            {"role": "system", "content": "以下是一个人类和一个聪明、懂心理学的AI助手之间的对话记录。"},
            {"role": "user", "content": "你好！请帮我对对话内容归纳总结"},
            {"role": "assistant", "content": "好的，我会尽力帮你的。"},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [
            {"role": "system", "content": "Below is a transcript of a conversation between a human and an AI assistant that is intelligent and knowledgeable in psychology."},
            {"role": "user", "content": "Hello! Please help me summarize the content of the conversation."},
            {"role": "assistant", "content": "Sure, I will do my best to assist you."},
            {"role": "user", "content": prompt},
        ]
    last_exc: Exception | None = None
    for _ in range(_max_retries()):
        try:
            resp = chat_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                top_p=1.0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1.0)
    raise RuntimeError(f"MemoryBank summarization LLM call failed: {last_exc}") from last_exc


def _embed_texts(embed_client: Any, model: str, texts: list[str], dims: int | None) -> Any:
    import numpy as np  # noqa: PLC0415

    # Use a frozen local embedding checkpoint when the answer-model gateway
    # does not expose an embeddings endpoint.
    if Path(model).is_dir():
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        models = getattr(_embed_texts, "_local_models", {})
        encoder = models.get(model)
        if encoder is None:
            encoder = SentenceTransformer(
                model,
                device=os.environ.get("MEMORY_EMBEDDING_DEVICE", "cpu"),
            )
            models[model] = encoder
            setattr(_embed_texts, "_local_models", models)
        return _l2_normalize(
            np.asarray(
                encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )
        )

    vectors: list[list[float]] = []
    extra: dict[str, Any] = {"dimensions": dims} if dims else {}
    for start in range(0, len(texts), DEFAULT_EMBED_BATCH):
        batch = texts[start : start + DEFAULT_EMBED_BATCH]
        last_exc: Exception | None = None
        for _ in range(_max_retries()):
            try:
                resp = embed_client.embeddings.create(model=model, input=batch, **extra)
                vectors.extend([item.embedding for item in resp.data])
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(1.0)
        if last_exc is not None:
            raise RuntimeError(f"MemoryBank embedding call failed: {last_exc}") from last_exc
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return _l2_normalize(arr)


def _l2_normalize(arr: Any) -> Any:
    import numpy as np  # noqa: PLC0415

    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return arr / norms


# ---------------------------------------------------------------------------
# MemoryBank 记忆结构构建
# ---------------------------------------------------------------------------
def _build_dialog_pairs(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """把 chat messages 配对成 MemoryBank 的 {query, response}（对齐 save_local_memory）。"""
    pairs: list[dict[str, str]] = []
    pending_q: str | None = None
    for message in messages:
        role = str(message.get("role") or "user").lower()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            pairs.append({"query": pending_q or "", "response": content})
            pending_q = None
        else:  # user / system 都作为人类发言
            if pending_q is not None:
                pairs.append({"query": pending_q, "response": ""})
            pending_q = content
    if pending_q is not None:
        pairs.append({"query": pending_q, "response": ""})
    return pairs


def _import_memorybank_prompts() -> tuple[Any, Any, Any, Any]:
    """复用 MemoryBank 仓库里的 summary prompt 构造函数。"""
    bank = str(MEMORYBANK_BANK_DIR.resolve())
    if bank not in sys.path:
        sys.path.insert(0, bank)
    from summarize_memory import (  # type: ignore  # noqa: PLC0415
        summarize_content_prompt,
        summarize_overall_personality,
        summarize_overall_prompt,
        summarize_person_prompt,
    )

    return (
        summarize_content_prompt,
        summarize_person_prompt,
        summarize_overall_prompt,
        summarize_overall_personality,
    )


def _build_user_memory(
    pairs: list[dict[str, str]],
    chat_client: Any,
    model: str,
    language: str,
) -> dict[str, Any]:
    """构建 MemoryBank 的 user_memory 结构（history + summary/personality + overall）。"""
    user_name = DEFAULT_USER_NAME
    boot_name = DEFAULT_BOOT_NAME
    date = DEFAULT_DATE

    user_memory: dict[str, Any] = {
        "name": user_name,
        "history": {date: pairs},
        "summary": {},
        "personality": {},
        "overall_history": "",
        "overall_personality": "",
    }
    if not pairs or not _summary_enabled():
        return user_memory

    (
        summarize_content_prompt,
        summarize_person_prompt,
        summarize_overall_prompt,
        summarize_overall_personality,
    ) = _import_memorybank_prompts()

    his_prompt = summarize_content_prompt(pairs, user_name, boot_name, language)
    person_prompt = summarize_person_prompt(pairs, user_name, boot_name, language)
    his_summary = _chat_summarize(chat_client, model, his_prompt, language)
    person_summary = _chat_summarize(chat_client, model, person_prompt, language)
    user_memory["summary"][date] = {"content": his_summary}
    user_memory["personality"][date] = person_summary

    overall_his_prompt = summarize_overall_prompt(list(user_memory["summary"].items()), language=language)
    overall_person_prompt = summarize_overall_personality(
        list(user_memory["personality"].items()), language=language
    )
    user_memory["overall_history"] = _chat_summarize(chat_client, model, overall_his_prompt, language)
    user_memory["overall_personality"] = _chat_summarize(chat_client, model, overall_person_prompt, language)
    return user_memory


def _build_documents(user_memory: dict[str, Any], language: str) -> list[dict[str, Any]]:
    """对齐 MemoryBank 的 JsonMemoryLoader.load：每轮对话一个文档 + 每日摘要文档。"""
    user_kw = "[|用户|]：" if language == "cn" else "[|User|]:"
    ai_kw = "[|AI恋人|]：" if language == "cn" else "[|AI|]:"
    docs: list[dict[str, Any]] = []
    history = user_memory.get("history", {})
    summary = user_memory.get("summary", {})
    for date, content in history.items():
        prefix = f"时间{date}的对话内容：" if language == "cn" else f"Conversation content on {date}:"
        for dialog in content:
            query = str(dialog.get("query") or "").strip()
            response = str(dialog.get("response") or "").strip()
            text = f"{prefix}{user_kw} {query}; {ai_kw} {response}"
            docs.append({"page_content": text, "date": date, "kind": "dialogue"})
        if date in summary and summary[date]:
            content_text = summary[date]["content"] if isinstance(summary[date], dict) else summary[date]
            summary_text = (
                f"时间{date}的对话总结为：{content_text}"
                if language == "cn"
                else f"The summary of the conversation on {date} is: {content_text}"
            )
            docs.append({"page_content": summary_text, "date": date, "kind": "summary"})
    return docs


# ---------------------------------------------------------------------------
# 磁盘缓存
# ---------------------------------------------------------------------------
def _dialogue_digest(
    pairs: list[dict[str, str]],
    *,
    model: str,
    embedding_model: str,
    embedding_dims: int | None,
    language: str,
    summary_enabled: bool,
) -> str:
    h = hashlib.sha1()
    for part in (
        METHOD,
        model,
        embedding_model,
        str(embedding_dims or ""),
        str(LLM_MAX_TOKENS),
        str(LLM_TEMPERATURE),
        language,
        "summary" if summary_enabled else "nosummary",
        f"v{_CACHE_VERSION}",
    ):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    for pair in pairs:
        h.update(str(pair.get("query") or "").encode("utf-8"))
        h.update(b"\0")
        h.update(str(pair.get("response") or "").encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:24]


def _marker_path(save_dir: str) -> Path:
    return Path(save_dir) / _COMPLETE_MARKER


def _docs_path(save_dir: str) -> Path:
    return Path(save_dir) / "documents.json"


def _vectors_path(save_dir: str) -> Path:
    return Path(save_dir) / "embeddings.npy"


def _memory_path(save_dir: str) -> Path:
    return Path(save_dir) / "memory.json"


def _load_from_disk(save_dir: str, digest: str, user_id: str) -> _MemoryIndex | None:
    import numpy as np  # noqa: PLC0415

    marker = _marker_path(save_dir)
    if not marker.is_file():
        return None
    try:
        meta = json.loads(marker.read_text(encoding="utf-8-sig"))
        if meta.get("digest") != digest or meta.get("method") != METHOD:
            return None
        docs = json.loads(_docs_path(save_dir).read_text(encoding="utf-8"))
        user_memory = json.loads(_memory_path(save_dir).read_text(encoding="utf-8"))
        vectors = np.load(_vectors_path(save_dir))
    except Exception:  # noqa: BLE001
        return None
    if len(docs) != len(vectors):
        return None
    return _MemoryIndex(
        user_id=user_id,
        save_dir=save_dir,
        docs=docs,
        vectors=vectors,
        overall_history=str(user_memory.get("overall_history") or ""),
        overall_personality=str(user_memory.get("overall_personality") or ""),
    )


def _save_to_disk(save_dir: str, digest: str, user_memory: dict[str, Any], docs: list[dict[str, Any]], vectors: Any) -> None:
    import numpy as np  # noqa: PLC0415

    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    _memory_path(save_dir).write_text(json.dumps(user_memory, ensure_ascii=False, indent=2), encoding="utf-8")
    _docs_path(save_dir).write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(_vectors_path(save_dir), vectors)
    marker = {
        "method": METHOD,
        "digest": digest,
        "documents_count": len(docs),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": _CACHE_VERSION,
    }
    tmp = _marker_path(save_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_marker_path(save_dir))


def _get_or_build_index(
    pairs: list[dict[str, str]],
    eval_config: BaselineEvalConfig,
) -> _MemoryIndex | None:
    if not pairs:
        return None

    language = _language()
    model = _resolve_llm_model(eval_config)
    embedding_model = _resolve_embedding_model(eval_config)
    embedding_dims = _resolve_embedding_dims(eval_config)
    summary_enabled = _summary_enabled()

    digest = _dialogue_digest(
        pairs,
        model=model,
        embedding_model=embedding_model,
        embedding_dims=embedding_dims,
        language=language,
        summary_enabled=summary_enabled,
    )
    user_id = f"pm_mb_{digest}"
    save_dir = str((eval_config.save_root / METHOD / user_id).resolve())

    with _cache_guard:
        cached = _inproc_cache.get(digest)
        if cached is not None:
            return cached

    disk = _load_from_disk(save_dir, digest, user_id)
    if disk is not None:
        print(f"[memory-cache] disk hit method={METHOD} user_id={user_id} save_dir={save_dir}", flush=True)
        with _cache_guard:
            _inproc_cache[digest] = disk
        return disk

    print(f"[memory-cache] disk miss; building method={METHOD} user_id={user_id} save_dir={save_dir}", flush=True)
    chat_client, embed_client = _make_clients(eval_config)
    user_memory = _build_user_memory(pairs, chat_client, model, language)
    docs = _build_documents(user_memory, language)
    if not docs:
        return None
    vectors = _embed_texts(embed_client, embedding_model, [d["page_content"] for d in docs], embedding_dims)

    index = _MemoryIndex(
        user_id=user_id,
        save_dir=save_dir,
        docs=docs,
        vectors=vectors,
        overall_history=str(user_memory.get("overall_history") or ""),
        overall_personality=str(user_memory.get("overall_personality") or ""),
    )
    try:
        _save_to_disk(save_dir, digest, user_memory, docs, vectors)
    except Exception as exc:  # noqa: BLE001
        print(f"[memory-cache] save failed method={METHOD} save_dir={save_dir}: {exc}", flush=True)
    with _cache_guard:
        _inproc_cache[digest] = index
    return index


# ---------------------------------------------------------------------------
# 检索（对齐 LocalMemoryRetrieval.search_memory：top-k + 按日期聚合）
# ---------------------------------------------------------------------------
def _retrieve(index: _MemoryIndex, query: str, eval_config: BaselineEvalConfig, top_k: int) -> list[dict[str, Any]]:
    import numpy as np  # noqa: PLC0415

    if not index.docs:
        return []
    import openai  # noqa: PLC0415

    embed_client = openai.OpenAI(
        api_key=eval_config.embedding_api_key or eval_config.api_key or "sk-local",
        base_url=eval_config.embedding_base_url or eval_config.base_url or None,
        timeout=_request_timeout(),
        max_retries=_max_retries(),
    )
    query_vec = _embed_texts(embed_client, _resolve_embedding_model(eval_config), [query], _resolve_embedding_dims(eval_config))[0]
    scores = index.vectors @ query_vec
    k = max(1, min(int(top_k), len(index.docs)))
    top_idx = np.argsort(-scores)[:k]

    selected = [
        {
            "index": int(i),
            "date": index.docs[int(i)].get("date") or "",
            "kind": index.docs[int(i)].get("kind") or "dialogue",
            "page_content": index.docs[int(i)].get("page_content") or "",
            "score": float(scores[int(i)]),
        }
        for i in top_idx
    ]
    # 对齐 search_memory：按日期排序后，去掉"对话内容："前缀，同日期合并。
    selected.sort(key=lambda d: d["date"])
    language = _language()
    memories: list[dict[str, Any]] = []
    prev_date = None
    for item in selected:
        content = str(item["page_content"])
        prefix = f"时间{item['date']}的对话内容：" if language == "cn" else f"Conversation content on {item['date']}:"
        content = content.replace(prefix, "").strip()
        if item["date"] != prev_date:
            memories.append(
                {
                    "content": content,
                    "used_content": content,
                    "metadata": {
                        "date": item["date"],
                        "kind": item["kind"],
                        "score": item["score"],
                    },
                }
            )
            prev_date = item["date"]
        else:
            memories[-1]["content"] += f"\n{content}"
            memories[-1]["used_content"] += f"\n{content}"
    return memories


def _format_context(index: _MemoryIndex, memories: list[dict[str, Any]], language: str) -> str:
    """对齐 build_prompt_with_search_memory_*：检索回忆 + 整体总结 + 人格策略。"""
    sections: list[str] = []

    if index.overall_personality.strip():
        head = (
            f"用户的性格以及AI伴侣的回复策略为：{index.overall_personality.strip()}"
            if language == "cn"
            else f"The personality of the user and the response strategy of the AI Companion are: {index.overall_personality.strip()}"
        )
        sections.append(head)

    if index.overall_history.strip():
        head = (
            f"你和用户过去的回忆总结是：{index.overall_history.strip()}"
            if language == "cn"
            else f"The summary of your past memories with the user is: {index.overall_history.strip()}"
        )
        sections.append(head)

    if memories:
        lead = (
            "根据当前用户的问题，与问题最相关的[回忆]是："
            if language == "cn"
            else "Based on the current question, the most relevant [memories] are:"
        )
        parts = [lead]
        for i, mem in enumerate(memories, start=1):
            date = mem.get("metadata", {}).get("date") or ""
            label = f"### Memory {i}" + (f" ({date})" if date else "") + ":"
            parts.append(f"{label}\n{str(mem.get('used_content') or '').strip()}")
        sections.append("\n\n".join(parts))
    else:
        sections.append("[NO RETRIEVED MEMORIES]")

    return "\n\n".join(s for s in sections if s.strip()).strip()


# ---------------------------------------------------------------------------
# 适配器入口
# ---------------------------------------------------------------------------
def build_context(
    prior_dialogue: str,
    user_question: str,
    eval_config: BaselineEvalConfig,
    *,
    sample_key: str | int | None = None,
) -> BaselineContext:
    messages = parse_dialogue_to_messages(prior_dialogue)
    pairs = _build_dialog_pairs(messages)
    index = _get_or_build_index(pairs, eval_config)

    language = _language()
    if index is None:
        return BaselineContext(
            context_text="[NO RETRIEVED MEMORIES]",
            retrieved_memories=[],
            user_id="",
            save_dir="",
            method=METHOD,
            top_k=eval_config.top_k,
        )

    memories = _retrieve(index, user_question, eval_config, eval_config.top_k)
    context_text = _format_context(index, memories, language)
    return BaselineContext(
        context_text=context_text,
        retrieved_memories=jsonable_memories(memories),
        user_id=index.user_id,
        save_dir=index.save_dir,
        method=METHOD,
        top_k=eval_config.top_k,
    )
