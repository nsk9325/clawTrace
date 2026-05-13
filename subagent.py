"""Subagent dispatch — recursive episodes spawned mid-turn.

Subagents are not tools. The LLM sees ``SUBAGENT_SCHEMA`` in its tool list
(when ``enable_subagents`` is set) and calls it like a tool, but the engine
routes the call to ``spawn(...)`` here instead of the tool registry. This
keeps the tool layer (``tools.py``) free of the engine-recursion concern
and lets subagents own their concurrency policy via ``subagent_parallelism``.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import events


SUBAGENT_TOOL_NAME = "spawn_subagent"

SUBAGENT_SCHEMA: dict[str, Any] = {
    "name": SUBAGENT_TOOL_NAME,
    "description": (
        "Spawn a subagent to work on a sub-task. Returns the subagent's final text "
        "when it completes. Use for independent sub-tasks that can stand alone."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "The task for the subagent"},
            "description": {"type": "string", "description": "Optional short description"},
            "model": {"type": "string", "description": "Override model"},
            "max_steps": {"type": "integer", "description": "Override max steps"},
        },
        "required": ["task"],
    },
}


def spawn(
    params: dict[str, Any],
    config: dict[str, Any],
    context: Any,
    origin: float,
) -> str:
    """Dispatch a subagent call. ``origin`` is the step's perf_counter origin
    so subagent_end can carry step-relative timing comparable to tool_call
    events (used by the analyzer for parallel-aware wall computation)."""
    from engine import Episode, run_episode

    if context is None:
        return "Error: spawn_subagent requires a runtime context."

    if not config.get("enable_subagents", False):
        return "Error: subagents are disabled (set enable_subagents=true)."

    task = str(params.get("task", "")).strip()
    if not task:
        return "Error: spawn_subagent requires a 'task' parameter."

    parent_episode: Episode = context.episode
    budget = context.budget
    reservation = budget.reserve(parent_episode.episode_id, parent_episode.depth)
    if not reservation.ok:
        return f"Error: {reservation.reason}"

    child_config = dict(config)
    for override_key in ("model", "max_steps"):
        if params.get(override_key) is not None:
            child_config[override_key] = params[override_key]

    child_episode = Episode.new_child(parent_episode, parent_step_id=context.step_id)
    child_episode_offset_ms = int((child_episode.t0 - parent_episode.t0) * 1000)
    child_trace_path = (
        Path(child_config.get("output_dir", "traces"))
        / child_episode.root_episode_id
        / f"{child_episode.episode_id}.jsonl"
    )

    context.writer.write(
        events.subagent_start(
            parent_episode,
            step_id=context.step_id,
            tool_call_id=context.tool_call_id,
            child_episode_id=child_episode.episode_id,
            child_run_id=child_episode.run_id,
            child_depth=child_episode.depth,
            task=task,
            description=params.get("description"),
        )
    )

    started = time.perf_counter()
    result: dict[str, Any] | None = None
    error: str | None = None
    try:
        result = run_episode(
            config=child_config,
            trace_path=child_trace_path,
            task_input=task,
            episode=child_episode,
            budget=budget,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        budget.release()

    ended = time.perf_counter()
    started_at_ms = int((started - origin) * 1000)
    ended_at_ms = int((ended - origin) * 1000)
    duration_ms = ended_at_ms - started_at_ms

    if result is not None:
        context.writer.write(
            events.subagent_end(
                parent_episode,
                step_id=context.step_id,
                tool_call_id=context.tool_call_id,
                child_episode_id=result["episode_id"],
                child_run_id=result["run_id"],
                child_trace_path=result["trace_path"],
                child_status=result["status"],
                child_stop_reason=result["stop_reason"],
                child_step_count=result["step_count"],
                started_at_ms=started_at_ms,
                ended_at_ms=ended_at_ms,
                duration_ms=duration_ms,
                child_episode_offset_ms=child_episode_offset_ms,
            )
        )
        return result["final_text"] or "(subagent produced no final text)"

    context.writer.write(
        events.subagent_end(
            parent_episode,
            step_id=context.step_id,
            tool_call_id=context.tool_call_id,
            child_episode_id=child_episode.episode_id,
            child_run_id=child_episode.run_id,
            child_trace_path=str(child_trace_path),
            child_status="crashed",
            child_stop_reason="exception",
            child_step_count=0,
            started_at_ms=started_at_ms,
            ended_at_ms=ended_at_ms,
            duration_ms=duration_ms,
            child_episode_offset_ms=child_episode_offset_ms,
            error=error,
        )
    )
    return f"Error: subagent crashed — {error}"
