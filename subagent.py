from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools import ToolDef, register_tool
from trace import make_event


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    system_prompt: str | None = None
    model: str | None = None
    max_steps: int | None = None


_SPAWN_SCHEMA = {
    "name": "spawn_subagent",
    "description": (
        "Spawn a subagent to work on a sub-task. Returns the subagent's final text "
        "when it completes. Use for independent sub-tasks that can stand alone."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "The task for the subagent"},
            "description": {"type": "string", "description": "Optional short description"},
            "system_prompt": {"type": "string", "description": "Override system prompt"},
            "model": {"type": "string", "description": "Override model"},
            "max_steps": {"type": "integer", "description": "Override max steps"},
        },
        "required": ["task"],
    },
}


def _spawn_subagent(params: dict[str, Any], config: dict[str, Any], context: Any) -> str:
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
    child_trace_path = Path(child_config.get("output_dir", "traces")) / f"{child_episode.episode_id}.jsonl"

    context.writer.write(
        make_event(
            "subagent_start",
            step_id=context.step_id,
            **parent_episode.to_event_fields(),
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

    duration_ms = int((time.perf_counter() - started) * 1000)

    if result is not None:
        context.writer.write(
            make_event(
                "subagent_end",
                step_id=context.step_id,
                **parent_episode.to_event_fields(),
                tool_call_id=context.tool_call_id,
                child_episode_id=result["episode_id"],
                child_run_id=result["run_id"],
                child_trace_path=result["trace_path"],
                child_status=result["status"],
                child_stop_reason=result["stop_reason"],
                child_step_count=result["step_count"],
                duration_ms=duration_ms,
            )
        )
        return result["final_text"] or "(subagent produced no final text)"

    context.writer.write(
        make_event(
            "subagent_end",
            step_id=context.step_id,
            **parent_episode.to_event_fields(),
            tool_call_id=context.tool_call_id,
            child_episode_id=child_episode.episode_id,
            child_run_id=child_episode.run_id,
            child_trace_path=str(child_trace_path),
            child_status="crashed",
            child_stop_reason="exception",
            child_step_count=0,
            duration_ms=duration_ms,
            error=error,
        )
    )
    return f"Error: subagent crashed — {error}"


def _register_subagent_tool() -> None:
    register_tool(
        ToolDef(
            name="spawn_subagent",
            schema=_SPAWN_SCHEMA,
            func=_spawn_subagent,
            read_only=False,
            concurrent_safe=False,
        )
    )


_register_subagent_tool()
