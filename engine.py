from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import RunConfig, load_config
from llm import AssistantTurn, run_assistant_turn
from tools import execute_tool, get_tool, get_tool_schemas
from trace import TraceWriter, make_event


@dataclass(frozen=True)
class Episode:
    episode_id: str
    run_id: str
    depth: int = 0
    root_episode_id: str = ""
    parent_episode_id: str | None = None
    parent_run_id: str | None = None
    parent_step_id: int | None = None

    @classmethod
    def new_root(cls) -> "Episode":
        episode_id = f"episode_{uuid.uuid4().hex[:8]}"
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        return cls(
            episode_id=episode_id,
            run_id=run_id,
            depth=0,
            root_episode_id=episode_id,
        )


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    name: str
    output: str
    started_at_ms: int
    ended_at_ms: int
    latency_ms: int
    exit_status: str


class ToolExecutor:
    def __init__(self, cfg: RunConfig, writer: TraceWriter, episode: Episode):
        self.cfg = cfg
        self.writer = writer
        self.episode = episode
        self._cfg_dict = cfg.to_dict()

    def execute(self, tool_calls: list[dict[str, Any]], step_id: int) -> list[ToolResult]:
        origin = time.perf_counter()
        if self._can_parallelize(tool_calls):
            return self._execute_parallel(tool_calls, step_id, origin)
        return self._execute_serial(tool_calls, step_id, origin)

    def _can_parallelize(self, tool_calls: list[dict[str, Any]]) -> bool:
        if not self.cfg.allow_parallel_tools or len(tool_calls) < 2:
            return False
        for tool_call in tool_calls:
            tool_def = get_tool(tool_call["name"])
            if tool_def is None or not tool_def.concurrent_safe:
                return False
        return True

    def _execute_serial(
        self,
        tool_calls: list[dict[str, Any]],
        step_id: int,
        origin: float,
    ) -> list[ToolResult]:
        return [self._run_one(tool_call, step_id, origin) for tool_call in tool_calls]

    def _execute_parallel(
        self,
        tool_calls: list[dict[str, Any]],
        step_id: int,
        origin: float,
    ) -> list[ToolResult]:
        workers = min(len(tool_calls), max(1, self.cfg.max_parallel_tools))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self._run_one, tool_call, step_id, origin)
                for tool_call in tool_calls
            ]
            return [future.result() for future in futures]

    def _run_one(
        self,
        tool_call: dict[str, Any],
        step_id: int,
        origin: float,
    ) -> ToolResult:
        started = time.perf_counter()
        output = execute_tool(tool_call["name"], tool_call["input"], self._cfg_dict)
        ended = time.perf_counter()

        started_at_ms = int((started - origin) * 1000)
        ended_at_ms = int((ended - origin) * 1000)
        latency_ms = ended_at_ms - started_at_ms
        exit_status = "error" if output.startswith("Error:") else "ok"

        self.writer.write(
            make_event(
                "tool_call",
                episode_id=self.episode.episode_id,
                run_id=self.episode.run_id,
                step_id=step_id,
                tool_call_id=tool_call["id"],
                tool_name=tool_call["name"],
                params=tool_call["input"],
                started_at_ms=started_at_ms,
                ended_at_ms=ended_at_ms,
                latency_ms=latency_ms,
                exit_status=exit_status,
                result_preview=_text_preview(output),
            )
        )

        return ToolResult(
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
            output=output,
            started_at_ms=started_at_ms,
            ended_at_ms=ended_at_ms,
            latency_ms=latency_ms,
            exit_status=exit_status,
        )


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


def _resolve_trace_path(trace_path: str | Path | None, cfg: RunConfig, episode: Episode) -> Path:
    if trace_path is not None:
        return Path(trace_path)
    return Path(cfg.output_dir) / f"{episode.episode_id}.jsonl"


def run_episode(
    config: dict[str, Any] | None = None,
    trace_path: str | Path | None = None,
    task_input: str = "",
) -> dict[str, Any]:
    merged = dict(load_config())
    if config:
        merged.update(config)
    cfg = RunConfig.from_dict(merged)
    cfg_dict = cfg.to_dict()

    task = _episode_task_input(task_input)
    episode = Episode.new_root()
    trace_path = _resolve_trace_path(trace_path, cfg, episode)

    history: list[dict[str, Any]] = [{"role": "user", "content": task}]
    completed_steps = 0
    final_status = "completed"
    final_stop_reason = "max_steps_reached"
    tool_schemas = get_tool_schemas()

    with TraceWriter(trace_path) as writer:
        executor = ToolExecutor(cfg, writer, episode)

        for step_id in range(cfg.max_steps):
            writer.write(
                make_event(
                    "step_start",
                    episode_id=episode.episode_id,
                    run_id=episode.run_id,
                    step_id=step_id,
                    task_input=task,
                    history_length=len(history),
                )
            )

            assistant_turn: AssistantTurn = run_assistant_turn(history, tool_schemas, cfg_dict)

            writer.write(
                make_event(
                    "llm_call",
                    episode_id=episode.episode_id,
                    run_id=episode.run_id,
                    step_id=step_id,
                    backend=cfg.backend,
                    model=cfg.model,
                    latency_ms=assistant_turn.latency_ms,
                    prompt_tokens=assistant_turn.input_tokens,
                    completion_tokens=assistant_turn.output_tokens,
                    ttft_ms=assistant_turn.ttft_ms,
                    prefill_time_ms=assistant_turn.prefill_time_ms,
                    decode_time_ms=assistant_turn.decode_time_ms,
                    measurement=assistant_turn.measurement,
                    finish_reason=assistant_turn.finish_reason,
                    assistant_text_preview=_text_preview(assistant_turn.text),
                    tool_call_count=len(assistant_turn.tool_calls),
                )
            )

            history.append({
                "role": "assistant",
                "content": assistant_turn.text,
                "tool_calls": assistant_turn.tool_calls,
            })

            tool_results = executor.execute(assistant_turn.tool_calls, step_id)
            for result in tool_results:
                history.append({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "name": result.name,
                    "content": result.output,
                })

            stop_reason = "no_tool_calls" if not assistant_turn.tool_calls else None

            writer.write(
                make_event(
                    "context_update",
                    episode_id=episode.episode_id,
                    run_id=episode.run_id,
                    step_id=step_id,
                    history_length=len(history),
                    tool_call_count=len(assistant_turn.tool_calls),
                    executed_tool_call_count=len(tool_results),
                )
            )

            writer.write(
                make_event(
                    "step_end",
                    episode_id=episode.episode_id,
                    run_id=episode.run_id,
                    step_id=step_id,
                    action_type="assistant_turn",
                    tool_call_count=len(assistant_turn.tool_calls),
                    executed_tool_call_count=len(tool_results),
                    stop_reason=stop_reason,
                )
            )

            completed_steps = step_id + 1
            if stop_reason is not None:
                final_stop_reason = stop_reason
                break

        writer.write(
            make_event(
                "episode_end",
                episode_id=episode.episode_id,
                run_id=episode.run_id,
                step_id=None,
                status=final_status,
                stop_reason=final_stop_reason,
                step_count=completed_steps,
                history_length=len(history),
            )
        )

    return {
        "episode_id": episode.episode_id,
        "run_id": episode.run_id,
        "trace_path": str(trace_path),
        "status": final_status,
        "step_count": completed_steps,
        "task_input": task,
        "history": history,
    }
