from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from memory import Memory


PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "max_completion_tokens": 16384,
    },
    "custom": {
        "api_key_env": "CUSTOM_API_KEY",
        "base_url": None,
        "max_completion_tokens": None,
    },
}

_PREFIXES = [
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
]

@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[dict[str, Any]]
    input_tokens: int
    output_tokens: int
    latency_ms: int
    ttft_ms: int | None
    decode_time_ms: int | None
    measurement: str
    finish_reason: str | None = None
    cached_tokens: int | None = None


def detect_provider(model: str) -> str:
    if "/" in model:
        return model.split("/", 1)[0]
    for prefix, provider_name in _PREFIXES:
        if model.lower().startswith(prefix):
            return provider_name
    return "openai"


def bare_model(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def get_api_key(provider_name: str, config: dict[str, Any]) -> str:
    config_key = str(config.get(f"{provider_name}_api_key", "")).strip()
    if config_key:
        return config_key

    env_var = PROVIDERS.get(provider_name, {}).get("api_key_env")
    if env_var:
        return os.environ.get(env_var, "")
    return ""


def get_base_url(provider_name: str, config: dict[str, Any]) -> str | None:
    if provider_name == "openai":
        override = str(config.get("openai_base_url", "")).strip()
        return override or PROVIDERS["openai"]["base_url"]

    if provider_name == "custom":
        return str(config.get("custom_base_url", "")).strip() or None

    return PROVIDERS.get(provider_name, {}).get("base_url")


def tools_to_openai(tool_schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in tool_schemas
    ]


def messages_to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]

        if role in ("system", "user"):
            result.append({
                "role": role,
                "content": str(message.get("content", "")),
            })

        elif role == "assistant":
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content") or None,
            }
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                assistant_message["tool_calls"] = []
                for tool_call in tool_calls:
                    assistant_message["tool_calls"].append({
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["input"], ensure_ascii=False),
                        },
                    })
            result.append(assistant_message)

        elif role == "tool":
            result.append({
                "role": "tool",
                "tool_call_id": message["tool_call_id"],
                "content": str(message.get("content", "")),
            })

    return result


def _build_openai_messages(memory: Memory) -> list[dict[str, Any]]:
    base: list[dict[str, Any]] = (
        [{"role": "system", "content": memory.system_prompt}] + memory.messages
        if memory.system_prompt
        else memory.messages
    )
    return messages_to_openai(base)


def _openai_client_from_config(config: dict[str, Any]) -> tuple[OpenAI, str]:
    model = str(config.get("model", "gpt-4o-mini"))
    provider_name = str(config.get("backend") or detect_provider(model))

    if provider_name not in ("openai", "custom"):
        raise NotImplementedError(f"Backend '{provider_name}' is not implemented yet.")

    api_key = get_api_key(provider_name, config)
    if not api_key:
        missing_var = PROVIDERS[provider_name]["api_key_env"]
        raise RuntimeError(f"{missing_var} is not set. Add it to your environment or config.")

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = get_base_url(provider_name, config)
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs), provider_name


def run_assistant_turn(
    memory: Memory,
    tool_schemas: list[dict[str, Any]],
    config: dict[str, Any],
) -> AssistantTurn:
    client, provider_name = _openai_client_from_config(config)
    model = str(config.get("model", "gpt-4o-mini"))
    messages = _build_openai_messages(memory)

    kwargs: dict[str, Any] = {
        "model": bare_model(model),
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    if tool_schemas:
        kwargs["tools"] = tools_to_openai(tool_schemas)
        kwargs["tool_choice"] = "auto"

    max_tokens = config.get("max_tokens")
    if max_tokens:
        cap = PROVIDERS.get(provider_name, {}).get("max_completion_tokens")
        value = min(int(max_tokens), cap) if cap else int(max_tokens)
        if provider_name == "openai":
            kwargs["max_completion_tokens"] = value
        else:
            kwargs["max_tokens"] = value

    if "temperature" in config:
        kwargs["temperature"] = float(config.get("temperature", 0.0))

    text = ""
    tool_buf: dict[int, dict[str, Any]] = {}
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens: int | None = None
    finish_reason: str | None = None

    request_start = time.perf_counter()
    first_token_at: float | None = None

    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        if hasattr(chunk, "usage") and chunk.usage:
            prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0) or prompt_tokens
            completion_tokens = getattr(chunk.usage, "completion_tokens", 0) or completion_tokens
            details = getattr(chunk.usage, "prompt_tokens_details", None)
            if details is not None:
                # OpenAI exposes prefix-cache hit count here. Older API
                # versions and some non-OpenAI providers omit this — leave
                # cached_tokens as None when unmeasured.
                cached = getattr(details, "cached_tokens", None)
                if cached is not None:
                    cached_tokens = int(cached)

        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        delta = choice.delta
        if choice.finish_reason:
            finish_reason = choice.finish_reason

        if delta.content:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            text += delta.content

        if delta.tool_calls:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            for tool_call in delta.tool_calls:
                index = tool_call.index
                if index not in tool_buf:
                    tool_buf[index] = {"id": "", "name": "", "args": ""}
                if tool_call.id:
                    tool_buf[index]["id"] = tool_call.id
                if tool_call.function:
                    if tool_call.function.name:
                        tool_buf[index]["name"] += tool_call.function.name
                    if tool_call.function.arguments:
                        tool_buf[index]["args"] += tool_call.function.arguments

    request_end = time.perf_counter()

    reconstructed_tool_calls: list[dict[str, Any]] = []
    for index in sorted(tool_buf):
        buffered = tool_buf[index]
        try:
            tool_input = json.loads(buffered["args"]) if buffered["args"] else {}
        except json.JSONDecodeError:
            tool_input = {"_raw": buffered["args"]}

        reconstructed_tool_calls.append({
            "id": buffered["id"] or f"call_{index}",
            "name": buffered["name"],
            "input": tool_input,
        })

    ttft_ms = int((first_token_at - request_start) * 1000) if first_token_at is not None else None
    decode_time_ms = int((request_end - first_token_at) * 1000) if first_token_at is not None else None

    return AssistantTurn(
        text=text,
        tool_calls=reconstructed_tool_calls,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        latency_ms=int((request_end - request_start) * 1000),
        ttft_ms=ttft_ms,
        decode_time_ms=decode_time_ms,
        measurement="client_streaming",
        finish_reason=finish_reason,
        cached_tokens=cached_tokens,
    )
