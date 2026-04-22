from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from config import load_config
from llm import AssistantTurn, run_assistant_turn
from tools import execute_tool, get_tool_schemas
from trace import TraceWriter, make_event


def _text_preview(text: str, max_length: int = 200) -> str:
    preview = text.strip()
    if len(preview) <= max_length:
        return preview
    return preview[:max_length] + "..."


def _episode_task_input(task_input: str) -> str:
    cleaned = task_input.strip()
    if cleaned:
        return cleaned
    return "Inspect the current working directory and help with the task."


def run_episode(
    config: dict[str, Any] | None = None,
    trace_path: str | Path | None = None,
    task_input: str = "",
) -> dict[str, Any]:
    cfg = dict(load_config())
    if config:
        cfg.update(config)

    task = _episode_task_input(task_input)
    cfg["task_input"] = task

    episode_id = f"episode_{uuid.uuid4().hex[:8]}"
    run_id = f"run_{uuid.uuid4().hex[:8]}"

    if trace_path is None:
        trace_dir = Path(cfg["output_dir"])
        trace_path = trace_dir / f"{episode_id}.jsonl"
    else:
        trace_path = Path(trace_path)

    history: list[dict[str, Any]] = [{"role": "user", "content": task}]
    completed_steps = 0
    final_status = "completed"
    tool_schemas = get_tool_schemas()

    with TraceWriter(trace_path) as writer:
        for step_id in range(int(cfg["max_steps"])):
            writer.write(
                make_event(
                    "step_start",
                    episode_id=episode_id,
                    run_id=run_id,
                    step_id=step_id,
                    task_input=task,
                    history_length=len(history),
                )
            )

            assistant_turn: AssistantTurn = run_assistant_turn(history, tool_schemas, cfg)

            writer.write(
                make_event(
                    "llm_call",
                    episode_id=episode_id,
                    run_id=run_id,
                    step_id=step_id,
                    backend=str(cfg.get("backend", "openai")),
                    model=str(cfg.get("model", "gpt-4o-mini")),
                    latency_ms=assistant_turn.latency_ms,
                    prompt_tokens=assistant_turn.input_tokens,
                    completion_tokens=assistant_turn.output_tokens,
                    ttft_ms=assistant_turn.ttft_ms,
                    prefill_time_ms=assistant_turn.prefill_time_ms,
                    decode_time_ms=assistant_turn.decode_time_ms,
                    measurement=assistant_turn.measurement,
                    assistant_text_preview=_text_preview(assistant_turn.text),
                    tool_call_count=len(assistant_turn.tool_calls),
                )
            )

            history.append({
                "role": "assistant",
                "content": assistant_turn.text,
                "tool_calls": assistant_turn.tool_calls,
            })

            for tool_call in assistant_turn.tool_calls:
                tool_start = time.perf_counter()
                result = execute_tool(tool_call["name"], tool_call["input"], cfg)
                latency_ms = int((time.perf_counter() - tool_start) * 1000)

                writer.write(
                    make_event(
                        "tool_call",
                        episode_id=episode_id,
                        run_id=run_id,
                        step_id=step_id,
                        tool_call_id=tool_call["id"],
                        tool_name=tool_call["name"],
                        params=tool_call["input"],
                        latency_ms=latency_ms,
                        exit_status="error" if result.startswith("Error:") else "ok",
                        result_preview=_text_preview(result),
                    )
                )

                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_call["name"],
                    "content": result,
                })

            writer.write(
                make_event(
                    "context_update",
                    episode_id=episode_id,
                    run_id=run_id,
                    step_id=step_id,
                    history_length=len(history),
                    tool_call_count=len(assistant_turn.tool_calls),
                )
            )

            writer.write(
                make_event(
                    "step_end",
                    episode_id=episode_id,
                    run_id=run_id,
                    step_id=step_id,
                    action_type="assistant_turn",
                    tool_call_count=len(assistant_turn.tool_calls),
                )
            )

            completed_steps = step_id + 1
            if not assistant_turn.tool_calls:
                break

        writer.write(
            make_event(
                "episode_end",
                episode_id=episode_id,
                run_id=run_id,
                step_id=None,
                status=final_status,
                step_count=completed_steps,
                history_length=len(history),
            )
        )

    return {
        "episode_id": episode_id,
        "run_id": run_id,
        "trace_path": str(trace_path),
        "status": final_status,
        "step_count": completed_steps,
        "task_input": task,
        "history": history,
    }
