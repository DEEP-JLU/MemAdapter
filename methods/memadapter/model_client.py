"""OpenAI-compatible clients used by the MemAdapter runner."""

from __future__ import annotations

import os
import importlib.util
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

try:
    from .efficiency import ModelCallResult, extract_usage, utc_now
except ImportError:
    from efficiency import ModelCallResult, extract_usage, utc_now

MODEL_DEFAULTS = {
    "DeepSeek": {
        "api_key": "DEEPSEEK_API_KEY",
        "base_url": "DEEPSEEK_BASE_URL",
        "model": "DEEPSEEK_MODEL",
        "default_base_url": "",
        "default_model": "deepseek-v4-flash",
    },
    "GPT": {
        "api_key": "EVAL_API_KEY",
        "base_url": "EVAL_BASE_URL",
        "model": "EVAL_MODEL",
        "default_base_url": "",
        "default_model": "gpt-5.6-sol",
    },
    "Qwen": {
        "api_key": "QWEN_API_KEY",
        "base_url": "QWEN_BASE_URL",
        "model": "QWEN_MODEL",
        "default_base_url": "",
        "default_model": "qwen-plus",
    },
}


def _value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def _openai_base_url(value: str) -> str:
    """Accept either a gateway root or its explicit OpenAI-compatible /v1 URL."""

    if not value.strip():
        raise RuntimeError(
            "A model endpoint is required. Set the model family's BASE_URL "
            "environment variable; endpoints are intentionally not stored in this repository."
        )
    parts = urlsplit(value.rstrip("/"))
    if not parts.path:
        parts = parts._replace(path="/v1")
    return urlunsplit(parts)


def _api_mode(*names: str, default: str = "chat_completions") -> str:
    value = _value(*names, default=default).lower().replace("-", "_")
    if value in {"responses", "response"}:
        return "responses"
    return "chat_completions"


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        for part in content or []:
            value = getattr(part, "text", None)
            if value is None and isinstance(part, dict):
                value = part.get("text")
            if value:
                parts.append(str(value))
    return "".join(parts).strip()


_LOCAL_QWEN_LOAD_LOCK = threading.Lock()
_LOCAL_QWEN_CALL_LOCK = threading.Lock()
_LOCAL_QWEN_RUNTIMES: dict[str, dict[str, Any]] = {}


def _local_qwen_runtime(model_path: str) -> dict[str, Any]:
    """Load one local Qwen runtime and cache it for the current process."""

    quantization = _value("QWEN_LOCAL_QUANTIZATION", default="4bit_nf4").lower()
    device_map = _value("QWEN_LOCAL_DEVICE_MAP", default="auto")
    compute_dtype_name = _value("QWEN_LOCAL_COMPUTE_DTYPE", default="bfloat16").lower()
    cache_key = "|".join((str(Path(model_path).resolve()), quantization, device_map, compute_dtype_name))
    with _LOCAL_QWEN_LOAD_LOCK:
        cached = _LOCAL_QWEN_RUNTIMES.get(cache_key)
        if cached is not None:
            return cached

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Local Qwen requires torch and transformers in the active environment."
            ) from exc

        use_4bit = quantization in {"4bit", "4bit_nf4", "nf4"}
        if use_4bit:
            if importlib.util.find_spec("bitsandbytes") is None:
                raise RuntimeError(
                    "QWEN_LOCAL_QUANTIZATION=4bit_nf4 requires bitsandbytes."
                )
            if importlib.util.find_spec("accelerate") is None:
                raise RuntimeError(
                    "QWEN_LOCAL_DEVICE_MAP=auto requires accelerate."
                )
            from transformers import BitsAndBytesConfig

        if compute_dtype_name in {"float16", "fp16", "half"}:
            compute_dtype = torch.float16
        elif compute_dtype_name in {"float32", "fp32"}:
            compute_dtype = torch.float32
        else:
            compute_dtype = torch.bfloat16

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        }
        if use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["torch_dtype"] = compute_dtype
        else:
            model_kwargs["torch_dtype"] = "auto"

        offload_folder = _value("QWEN_LOCAL_OFFLOAD_FOLDER")
        if offload_folder:
            model_kwargs["offload_folder"] = offload_folder

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        model.eval()
        model_device = getattr(model, "device", None)
        if model_device is None:
            model_device = next(model.parameters()).device
        runtime = {
            "backend": "transformers",
            "model": _value("QWEN_LOCAL_MODEL_NAME", default=Path(model_path).name),
            "model_path": str(Path(model_path).resolve()),
            "quantization": "4-bit NF4" if use_4bit else "none",
            "device_map": device_map,
            "compute_dtype": str(compute_dtype).replace("torch.", ""),
            "device": str(model_device),
            "tokenizer": tokenizer,
            "model_object": model,
        }
        _LOCAL_QWEN_RUNTIMES[cache_key] = runtime
        print(
            "[local-qwen] loaded "
            f"model={runtime['model']} quantization={runtime['quantization']} "
            f"device_map={device_map} device={runtime['device']}",
            flush=True,
        )
        return runtime


class ModelClient:
    """Small adapter exposing one stable completion method to MemAdapter."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.runtime_config: dict[str, Any] = {"backend": "openai-compatible"}
        local_qwen_path = (
            _value("QWEN_LOCAL_MODEL_PATH") if model_name == "Qwen" else ""
        )
        if local_qwen_path:
            self.api_key = ""
            self.base_url = "local://transformers"
            self.model = _value(
                "QWEN_LOCAL_MODEL_NAME",
                default=Path(local_qwen_path).name,
            )
            self.temperature = float(_value("MEMADAPTER_TEMPERATURE", default="0"))
            self.max_tokens = int(_value("MEMADAPTER_MAX_TOKENS", default="4096"))
            runtime = _local_qwen_runtime(local_qwen_path)
            self._local_qwen = runtime
            self.runtime_config = {
                key: value
                for key, value in runtime.items()
                if key not in {"tokenizer", "model_object"}
            }
            return

        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The openai package is required for a formal run. "
                "Install benchmark/requirements.txt first."
            ) from exc
        try:
            defaults = MODEL_DEFAULTS[model_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported model: {model_name}") from exc
        self.api_key = _value(defaults["api_key"], "OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                f"Missing API key. Set {defaults['api_key']} for model {model_name}."
            )
        self.base_url = _openai_base_url(
            _value(
                defaults["base_url"],
                "OPENAI_BASE_URL",
                default=defaults["default_base_url"],
            )
        )
        self.model = _value(defaults["model"], "OPENAI_MODEL", default=defaults["default_model"])
        self.api_mode = _api_mode("MEMADAPTER_API_MODE", "OPENAI_API_MODE")
        self.temperature = float(_value("MEMADAPTER_TEMPERATURE", default="0"))
        self.max_tokens = int(_value("MEMADAPTER_MAX_TOKENS", default="8192"))
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=float(_value("MEMADAPTER_REQUEST_TIMEOUT", default="180")),
            max_retries=0,
        )
        self.runtime_config = {
            "backend": "openai-compatible",
            "api_mode": self.api_mode,
        }

    def _complete_local_qwen(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
    ) -> ModelCallResult:
        """Run a deterministic local Qwen chat-template completion."""

        import torch

        runtime = self._local_qwen
        tokenizer = runtime["tokenizer"]
        model = runtime["model_object"]
        request_started_at = utc_now()
        started = time.perf_counter()
        input_tokens: int | None = None
        effective_max_tokens = int(max_tokens or self.max_tokens)
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "return_dict": True,
            }
            try:
                encoded = tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=False,
                    **template_kwargs,
                )
            except (TypeError, ValueError):
                encoded = tokenizer.apply_chat_template(messages, **template_kwargs)

            if hasattr(encoded, "items"):
                model_inputs = dict(encoded)
            else:
                model_inputs = {"input_ids": encoded}
            input_ids = model_inputs["input_ids"]
            input_tokens = int(input_ids.shape[-1])
            model_device = getattr(model, "device", None)
            if model_device is None:
                model_device = next(model.parameters()).device
            model_inputs = {
                key: value.to(model_device) if torch.is_tensor(value) else value
                for key, value in model_inputs.items()
            }
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": effective_max_tokens,
                "do_sample": False,
                "use_cache": True,
            }
            if pad_token_id is not None:
                generation_kwargs["pad_token_id"] = pad_token_id
            if tokenizer.eos_token_id is not None:
                generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
            with _LOCAL_QWEN_CALL_LOCK, torch.inference_mode():
                generated = model.generate(**model_inputs, **generation_kwargs)
            new_token_ids = generated[0, input_tokens:]
            text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
            response_received_at = utc_now()
            latency_ms = (time.perf_counter() - started) * 1000
            if not text:
                raise RuntimeError("Local Qwen returned an empty response")
            output_tokens = int(new_token_ids.shape[-1])
            return ModelCallResult(
                text=text,
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=effective_max_tokens,
                request_started_at=request_started_at,
                response_received_at=response_received_at,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cached_input_tokens=None,
                reasoning_tokens=None,
                usage_available=True,
                finish_reason=(
                    "length" if output_tokens >= effective_max_tokens else "stop"
                ),
            )
        except BaseException as exc:
            if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                exc._efficiency_request_started_at = request_started_at
                exc._efficiency_response_received_at = utc_now()
                exc._efficiency_latency_ms = (time.perf_counter() - started) * 1000
                exc._efficiency_input_tokens = input_tokens
                exc._efficiency_output_tokens = None
                exc._efficiency_total_tokens = None
                exc._efficiency_usage_available = input_tokens is not None
                exc._efficiency_finish_reason = None
            raise

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        return self.complete_result(
            system_prompt,
            user_prompt,
            json_mode=json_mode,
            max_tokens=max_tokens,
        ).text

    def complete_result(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ModelCallResult:
        if self.model_name == "Qwen" and hasattr(self, "_local_qwen"):
            return self._complete_local_qwen(
                system_prompt,
                user_prompt,
                max_tokens=max_tokens,
            )
        if getattr(self, "api_mode", "chat_completions") == "responses":
            return self._complete_responses(
                system_prompt,
                user_prompt,
                json_mode=json_mode,
                max_tokens=max_tokens,
            )

        request: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        top_p = _value("MEMADAPTER_TOP_P")
        if top_p:
            request["top_p"] = float(top_p)
        # DashScope Qwen3 thinking mode returns a literal JSON string such as
        # `"nil"` when the OpenAI `response_format=json_object` flag is set.
        # The stage prompts already require JSON, so omit that incompatible
        # transport-level constraint only for Qwen and parse its normal reply.
        if (
            json_mode
            and self.model_name != "Qwen"
            and not _value("MEMADAPTER_DISABLE_JSON_MODE")
        ):
            request["response_format"] = {"type": "json_object"}
        if self.model_name == "GPT" and not _value("MEMADAPTER_ENABLE_REASONING"):
            request["reasoning_effort"] = "none"
        elif self.model_name == "DeepSeek" and not _value("MEMADAPTER_ENABLE_REASONING"):
            request["extra_body"] = {"thinking": {"type": "disabled"}}
            request["reasoning_effort"] = "none"
        elif self.model_name == "Qwen":
            # DashScope's OpenAI-compatible endpoint controls Qwen3 reasoning
            # through this provider-specific field.  Keep it explicit in the
            # recorded runtime configuration rather than relying on a model
            # default that can vary by deployment.
            enabled = _value("QWEN_ENABLE_THINKING", default="true").lower()
            qwen_extra: dict[str, Any] = {"enable_thinking": enabled in {"1", "true", "yes"}}
            top_k = _value("MEMADAPTER_TOP_K")
            if top_k:
                qwen_extra["top_k"] = int(top_k)
            request["extra_body"] = qwen_extra

        request_started_at = utc_now()
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**request)
        except BaseException as exc:
            if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                exc._efficiency_request_started_at = request_started_at
                exc._efficiency_response_received_at = utc_now()
                exc._efficiency_latency_ms = (time.perf_counter() - started) * 1000
            raise
        response_received_at = utc_now()
        latency_ms = (time.perf_counter() - started) * 1000
        choices = getattr(response, "choices", None) or []
        if not choices:
            error = RuntimeError("Model response contains no choices")
            usage = extract_usage(response)
            error._efficiency_input_tokens = usage["input_tokens"]
            error._efficiency_output_tokens = usage["output_tokens"]
            error._efficiency_total_tokens = usage["total_tokens"]
            error._efficiency_cached_input_tokens = usage["cached_input_tokens"]
            error._efficiency_reasoning_tokens = usage["reasoning_tokens"]
            error._efficiency_usage_available = usage["usage_available"]
            error._efficiency_response_received_at = response_received_at
            error._efficiency_latency_ms = latency_ms
            raise error
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message else None
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        text = str(content or "").strip()
        if text:
            usage = extract_usage(response)
            return ModelCallResult(
                text=text,
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                request_started_at=request_started_at,
                response_received_at=response_received_at,
                latency_ms=latency_ms,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"],
                cached_input_tokens=usage["cached_input_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
                usage_available=usage["usage_available"],
                finish_reason=getattr(choices[0], "finish_reason", None),
            )
        reasoning = str(getattr(message, "reasoning_content", "") or "").strip()
        if reasoning.startswith("{") or reasoning.startswith("```json"):
            usage = extract_usage(response)
            return ModelCallResult(
                text=reasoning,
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                request_started_at=request_started_at,
                response_received_at=response_received_at,
                latency_ms=latency_ms,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"],
                cached_input_tokens=usage["cached_input_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
                usage_available=usage["usage_available"],
                finish_reason=getattr(choices[0], "finish_reason", None),
            )
        error = RuntimeError(
            f"Empty model response: finish_reason={getattr(choices[0], 'finish_reason', None)!r}"
        )
        usage = extract_usage(response)
        error._efficiency_input_tokens = usage["input_tokens"]
        error._efficiency_output_tokens = usage["output_tokens"]
        error._efficiency_total_tokens = usage["total_tokens"]
        error._efficiency_cached_input_tokens = usage["cached_input_tokens"]
        error._efficiency_reasoning_tokens = usage["reasoning_tokens"]
        error._efficiency_usage_available = usage["usage_available"]
        error._efficiency_response_received_at = response_received_at
        error._efficiency_latency_ms = latency_ms
        error._efficiency_finish_reason = getattr(choices[0], "finish_reason", None)
        raise error

    def _complete_responses(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ModelCallResult:
        """Run an OpenAI Responses API completion with chat-compatible metadata."""

        request: dict[str, Any] = {
            "model": self.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "temperature": self.temperature,
            "max_output_tokens": max_tokens or self.max_tokens,
        }
        if json_mode and not _value("MEMADAPTER_DISABLE_JSON_MODE"):
            request["text"] = {"format": {"type": "json_object"}}
        if not _value("MEMADAPTER_ENABLE_REASONING"):
            request["reasoning"] = {"effort": "none"}

        request_started_at = utc_now()
        started = time.perf_counter()
        try:
            response = self.client.responses.create(**request)
        except BaseException as exc:
            if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                exc._efficiency_request_started_at = request_started_at
                exc._efficiency_response_received_at = utc_now()
                exc._efficiency_latency_ms = (time.perf_counter() - started) * 1000
            raise
        response_received_at = utc_now()
        latency_ms = (time.perf_counter() - started) * 1000
        text = _response_text(response)
        usage = extract_usage(response)
        if text:
            incomplete = getattr(response, "incomplete_details", None)
            finish_reason = getattr(incomplete, "reason", None) if incomplete else None
            return ModelCallResult(
                text=text,
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                request_started_at=request_started_at,
                response_received_at=response_received_at,
                latency_ms=latency_ms,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"],
                cached_input_tokens=usage["cached_input_tokens"],
                reasoning_tokens=usage["reasoning_tokens"],
                usage_available=usage["usage_available"],
                finish_reason=finish_reason or getattr(response, "status", None),
            )
        error = RuntimeError(
            "Empty Responses API response: "
            f"status={getattr(response, 'status', None)!r}"
        )
        error._efficiency_input_tokens = usage["input_tokens"]
        error._efficiency_output_tokens = usage["output_tokens"]
        error._efficiency_total_tokens = usage["total_tokens"]
        error._efficiency_cached_input_tokens = usage["cached_input_tokens"]
        error._efficiency_reasoning_tokens = usage["reasoning_tokens"]
        error._efficiency_usage_available = usage["usage_available"]
        error._efficiency_response_received_at = response_received_at
        error._efficiency_latency_ms = latency_ms
        error._efficiency_finish_reason = getattr(response, "status", None)
        raise error


__all__ = ["MODEL_DEFAULTS", "ModelClient"]


