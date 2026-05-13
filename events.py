"""Trace event vocabulary — the single source of truth for the trace contract.

Every event the engine writes to a trace file is constructed by one of the
functions below. Each constructor's signature is the schema for that event
type: keyword-only arguments name the fields that belong to that event,
and adding or renaming a field is a single edit at one call site.

Common fields on every event (set by ``_base``):
  - type:        event-type tag, e.g. "episode_start", "tool_call"
  - timestamp:   UTC ISO 8601, set at construction time
  - step_id:     loop iteration index; None for events outside the step loop
  - episode_id, run_id, depth, root_episode_id,
    parent_episode_id, parent_run_id, parent_step_id
                 — copied from the Episode via ``Episode.to_event_fields``

Event vocabulary:
  episode_start, episode_end       — episode lifecycle
  step_start, step_end             — per-step lifecycle
  llm_call                         — one assistant turn (LLM call)
  tool_call                        — one tool execution
  subagent_start, subagent_end     — subagent dispatch
  context_update                   — post-step memory snapshot
  config_warning                   — flagged config combination at run start
  env_setup                        — per-instance venv build outcome (SWE-bench)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from engine import Episode


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base(event_type: str, episode: "Episode", step_id: int | None) -> dict[str, Any]:
    return {
        "type": event_type,
        "timestamp": utc_timestamp(),
        "step_id": step_id,
        **episode.to_event_fields(),
    }


def episode_start(
    episode: "Episode",
    *,
    task_input: str,
    model: str,
    backend: str,
    system_prompt_name: str,
    system_prompt_chars: int,
    cfg: dict[str, Any],
    workload_info: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        **_base("episode_start", episode, step_id=None),
        "task_input": task_input,
        "model": model,
        "backend": backend,
        "system_prompt_name": system_prompt_name,
        "system_prompt_chars": system_prompt_chars,
        "cfg": cfg,
        "workload_info": workload_info,
    }


def episode_end(
    episode: "Episode",
    *,
    status: str,
    stop_reason: str,
    step_count: int,
    history_length: int,
) -> dict[str, Any]:
    return {
        **_base("episode_end", episode, step_id=None),
        "status": status,
        "stop_reason": stop_reason,
        "step_count": step_count,
        "history_length": history_length,
    }


def step_start(
    episode: "Episode",
    *,
    step_id: int,
    history_length: int,
) -> dict[str, Any]:
    return {
        **_base("step_start", episode, step_id=step_id),
        "history_length": history_length,
    }


def step_end(
    episode: "Episode",
    *,
    step_id: int,
    action_type: str,
    tool_call_count: int,
    executed_tool_call_count: int,
    stop_reason: str | None,
) -> dict[str, Any]:
    return {
        **_base("step_end", episode, step_id=step_id),
        "action_type": action_type,
        "tool_call_count": tool_call_count,
        "executed_tool_call_count": executed_tool_call_count,
        "stop_reason": stop_reason,
    }


def llm_call(
    episode: "Episode",
    *,
    step_id: int,
    backend: str,
    model: str,
    latency_ms: int,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int | None,
    ttft_ms: int | None,
    decode_time_ms: int | None,
    measurement: str,
    finish_reason: str | None,
    assistant_text_preview: str,
    tool_call_count: int,
) -> dict[str, Any]:
    return {
        **_base("llm_call", episode, step_id=step_id),
        "backend": backend,
        "model": model,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "ttft_ms": ttft_ms,
        "decode_time_ms": decode_time_ms,
        "measurement": measurement,
        "finish_reason": finish_reason,
        "assistant_text_preview": assistant_text_preview,
        "tool_call_count": tool_call_count,
    }


def tool_call(
    episode: "Episode",
    *,
    step_id: int,
    tool_call_id: str,
    tool_name: str,
    params: dict[str, Any],
    started_at_ms: int,
    ended_at_ms: int,
    latency_ms: int,
    exit_status: str,
    result_preview: str,
) -> dict[str, Any]:
    return {
        **_base("tool_call", episode, step_id=step_id),
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "params": params,
        "started_at_ms": started_at_ms,
        "ended_at_ms": ended_at_ms,
        "latency_ms": latency_ms,
        "exit_status": exit_status,
        "result_preview": result_preview,
    }


def context_update(
    episode: "Episode",
    *,
    step_id: int,
    history_length: int,
    tool_call_count: int,
    executed_tool_call_count: int,
) -> dict[str, Any]:
    return {
        **_base("context_update", episode, step_id=step_id),
        "history_length": history_length,
        "tool_call_count": tool_call_count,
        "executed_tool_call_count": executed_tool_call_count,
    }


def env_setup(
    episode: "Episode",
    *,
    status: str,
    venv_path: str | None,
    python_version: str | None,
    install_commands: list[str],
    duration_ms: int,
    error: str | None,
) -> dict[str, Any]:
    return {
        **_base("env_setup", episode, step_id=None),
        "status": status,
        "venv_path": venv_path,
        "python_version": python_version,
        "install_commands": install_commands,
        "duration_ms": duration_ms,
        "error": error,
    }


def config_warning(
    episode: "Episode",
    *,
    field: str,
    value: Any,
    note: str,
) -> dict[str, Any]:
    return {
        **_base("config_warning", episode, step_id=None),
        "field": field,
        "value": value,
        "note": note,
    }


def subagent_start(
    parent_episode: "Episode",
    *,
    step_id: int,
    tool_call_id: str,
    child_episode_id: str,
    child_run_id: str,
    child_depth: int,
    task: str,
    description: str | None,
) -> dict[str, Any]:
    return {
        **_base("subagent_start", parent_episode, step_id=step_id),
        "tool_call_id": tool_call_id,
        "child_episode_id": child_episode_id,
        "child_run_id": child_run_id,
        "child_depth": child_depth,
        "task": task,
        "description": description,
    }


def subagent_end(
    parent_episode: "Episode",
    *,
    step_id: int,
    tool_call_id: str,
    child_episode_id: str,
    child_run_id: str,
    child_trace_path: str,
    child_status: str,
    child_stop_reason: str,
    child_step_count: int,
    started_at_ms: int,
    ended_at_ms: int,
    duration_ms: int,
    error: str | None = None,
) -> dict[str, Any]:
    payload = {
        **_base("subagent_end", parent_episode, step_id=step_id),
        "tool_call_id": tool_call_id,
        "child_episode_id": child_episode_id,
        "child_run_id": child_run_id,
        "child_trace_path": child_trace_path,
        "child_status": child_status,
        "child_stop_reason": child_stop_reason,
        "child_step_count": child_step_count,
        "started_at_ms": started_at_ms,
        "ended_at_ms": ended_at_ms,
        "duration_ms": duration_ms,
    }
    if error is not None:
        payload["error"] = error
    return payload
