from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import events
import subagent
from config import RunConfig, load_config
from llm import AssistantTurn, run_assistant_turn
from memory import Memory
from prompts import REGISTRY as PROMPT_REGISTRY, build_system_prompt
from tools import execute_tool, get_tool, get_tool_schemas
from trace import TraceWriter


SUPPORTED_SUBAGENT_PARALLELISM = {"serial", "shared"}
FUTURE_SUBAGENT_PARALLELISM = {"shared+optimistic", "worktree"}


@dataclass(frozen=True)
class Episode:
    episode_id: str
    run_id: str
    depth: int = 0
    root_episode_id: str = ""
    parent_episode_id: str | None = None
    parent_run_id: str | None = None
    parent_step_id: int | None = None
    # perf_counter origin captured at construction. Every event's t_ms is
    # measured against this — single clock source, so latency_ms summed over
    # a step is structurally bounded by step_end.t_ms - step_start.t_ms.
    t0: float = field(default_factory=time.perf_counter, compare=False)

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

    @classmethod
    def new_child(cls, parent: "Episode", parent_step_id: int) -> "Episode":
        return cls(
            episode_id=f"episode_{uuid.uuid4().hex[:8]}",
            run_id=f"run_{uuid.uuid4().hex[:8]}",
            depth=parent.depth + 1,
            root_episode_id=parent.root_episode_id,
            parent_episode_id=parent.episode_id,
            parent_run_id=parent.run_id,
            parent_step_id=parent_step_id,
        )

    def to_event_fields(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "depth": self.depth,
            "root_episode_id": self.root_episode_id,
            "parent_episode_id": self.parent_episode_id,
            "parent_run_id": self.parent_run_id,
            "parent_step_id": self.parent_step_id,
        }


@dataclass(frozen=True)
class Reservation:
    ok: bool
    reason: str = ""


class SubagentBudget:
    def __init__(self, cfg: RunConfig):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._total = 0
        self._per_parent: dict[str, int] = {}
        self._concurrent = 0

    def reserve(self, parent_episode_id: str, parent_depth: int) -> Reservation:
        with self._lock:
            if self._cfg.max_subagent_depth > 0 and parent_depth >= self._cfg.max_subagent_depth:
                return Reservation(False, f"max_subagent_depth ({self._cfg.max_subagent_depth}) reached")
            if self._cfg.max_subagents_total > 0 and self._total >= self._cfg.max_subagents_total:
                return Reservation(False, f"max_subagents_total ({self._cfg.max_subagents_total}) reached")
            per_parent_count = self._per_parent.get(parent_episode_id, 0)
            if self._cfg.max_subagents_per_parent > 0 and per_parent_count >= self._cfg.max_subagents_per_parent:
                return Reservation(False, f"max_subagents_per_parent ({self._cfg.max_subagents_per_parent}) reached")
            if self._cfg.max_concurrent_subagents > 0 and self._concurrent >= self._cfg.max_concurrent_subagents:
                return Reservation(False, f"max_concurrent_subagents ({self._cfg.max_concurrent_subagents}) reached")

            self._total += 1
            self._per_parent[parent_episode_id] = per_parent_count + 1
            self._concurrent += 1
            return Reservation(True)

    def release(self) -> None:
        with self._lock:
            if self._concurrent > 0:
                self._concurrent -= 1


@dataclass(frozen=True)
class RuntimeContext:
    episode: Episode
    writer: TraceWriter
    budget: SubagentBudget
    step_id: int
    tool_call_id: str
    cfg: RunConfig


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
    def __init__(
        self,
        cfg: RunConfig,
        writer: TraceWriter,
        episode: Episode,
        budget: SubagentBudget,
    ):
        self.cfg = cfg
        self.writer = writer
        self.episode = episode
        self.budget = budget
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
            if tool_call["name"] == subagent.SUBAGENT_TOOL_NAME:
                if self.cfg.subagent_parallelism == "serial":
                    return False
                continue
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
        context = RuntimeContext(
            episode=self.episode,
            writer=self.writer,
            budget=self.budget,
            step_id=step_id,
            tool_call_id=tool_call["id"],
            cfg=self.cfg,
        )
        is_subagent_call = tool_call["name"] == subagent.SUBAGENT_TOOL_NAME

        started = time.perf_counter()
        if is_subagent_call:
            output = subagent.spawn(tool_call["input"], self._cfg_dict, context, origin=origin)
        else:
            output = execute_tool(tool_call["name"], tool_call["input"], self._cfg_dict, context)
        ended = time.perf_counter()

        started_at_ms = int((started - origin) * 1000)
        ended_at_ms = int((ended - origin) * 1000)
        latency_ms = ended_at_ms - started_at_ms
        exit_status = "error" if output.startswith("Error:") else "ok"

        # Subagent dispatch is recorded by subagent_start/subagent_end (emitted
        # inside subagent.spawn). No tool_call event for it — subagents aren't
        # tools.
        if not is_subagent_call:
            self.writer.write(
                events.tool_call(
                    self.episode,
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
    # Per-root-episode directory: parent and all children land as siblings.
    # Folder name uses root_episode_id; filename uses each episode's own id.
    return Path(cfg.output_dir) / episode.root_episode_id / f"{episode.episode_id}.jsonl"


def _validate_subagent_parallelism(cfg: RunConfig) -> None:
    mode = cfg.subagent_parallelism
    if mode in FUTURE_SUBAGENT_PARALLELISM:
        raise NotImplementedError(
            f"subagent_parallelism='{mode}' is a future mode, not yet implemented"
        )
    if mode not in SUPPORTED_SUBAGENT_PARALLELISM:
        raise ValueError(f"Unknown subagent_parallelism: '{mode}'")


def _resolve_tool_schemas(cfg: RunConfig) -> list[dict[str, Any]]:
    schemas = list(get_tool_schemas())
    if cfg.enable_subagents:
        schemas.append(subagent.SUBAGENT_SCHEMA)
    return schemas


def run_episode(
    config: dict[str, Any] | None = None,
    trace_path: str | Path | None = None,
    task_input: str = "",
    episode: Episode | None = None,
    budget: SubagentBudget | None = None,
    workload_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(load_config())
    if config:
        merged.update(config)
    cfg = RunConfig.from_dict(merged)
    _validate_subagent_parallelism(cfg)
    cfg_dict = cfg.to_dict()

    task = _episode_task_input(task_input)
    if episode is None:
        episode = Episode.new_root()
    if budget is None:
        budget = SubagentBudget(cfg)

    trace_path = _resolve_trace_path(trace_path, cfg, episode)

    is_subagent = episode.depth > 0
    system_prompt = build_system_prompt(cfg, is_subagent=is_subagent)
    memory = Memory.with_initial_user(task, system_prompt=system_prompt)
    prompt_name = (
        "minimal" if is_subagent
        else (cfg.system_prompt if cfg.system_prompt in PROMPT_REGISTRY else "<custom>")
    )
    completed_steps = 0
    final_stop_reason = "max_steps_reached"
    tool_schemas = _resolve_tool_schemas(cfg)

    with TraceWriter(trace_path) as writer:
        writer.write(
            events.episode_start(
                episode,
                task_input=task,
                model=cfg.model,
                backend=cfg.backend,
                system_prompt_name=prompt_name,
                system_prompt_chars=len(system_prompt),
                cfg=cfg_dict,
                workload_info=workload_info,
            )
        )

        if episode.depth == 0 and cfg.subagent_parallelism != "serial":
            writer.write(
                events.config_warning(
                    episode,
                    field="subagent_parallelism",
                    value=cfg.subagent_parallelism,
                    note="parallel subagents share cwd; writes may race",
                )
            )

        executor = ToolExecutor(cfg, writer, episode, budget)

        for step_id in range(cfg.max_steps):
            writer.write(events.step_start(episode, step_id=step_id, history_length=len(memory)))

            assistant_turn: AssistantTurn = run_assistant_turn(memory, tool_schemas, cfg_dict)

            writer.write(
                events.llm_call(
                    episode,
                    step_id=step_id,
                    backend=cfg.backend,
                    model=cfg.model,
                    latency_ms=assistant_turn.latency_ms,
                    prompt_tokens=assistant_turn.input_tokens,
                    completion_tokens=assistant_turn.output_tokens,
                    cached_tokens=assistant_turn.cached_tokens,
                    ttft_ms=assistant_turn.ttft_ms,
                    decode_time_ms=assistant_turn.decode_time_ms,
                    measurement=assistant_turn.measurement,
                    finish_reason=assistant_turn.finish_reason,
                    assistant_text_preview=_text_preview(assistant_turn.text),
                    tool_call_count=len(assistant_turn.tool_calls),
                )
            )

            memory.append_assistant(assistant_turn.text, assistant_turn.tool_calls)

            tool_results = executor.execute(assistant_turn.tool_calls, step_id)
            for result in tool_results:
                memory.append_tool(result.tool_call_id, result.name, result.output)

            stop_reason = "no_tool_calls" if not assistant_turn.tool_calls else None

            writer.write(
                events.context_update(
                    episode,
                    step_id=step_id,
                    history_length=len(memory),
                    tool_call_count=len(assistant_turn.tool_calls),
                    executed_tool_call_count=len(tool_results),
                )
            )

            writer.write(
                events.step_end(
                    episode,
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

        final_status = "completed" if final_stop_reason == "no_tool_calls" else "incomplete"

        writer.write(
            events.episode_end(
                episode,
                status=final_status,
                stop_reason=final_stop_reason,
                step_count=completed_steps,
                history_length=len(memory),
            )
        )

    return {
        "episode_id": episode.episode_id,
        "run_id": episode.run_id,
        "trace_path": str(trace_path),
        "status": final_status,
        "stop_reason": final_stop_reason,
        "step_count": completed_steps,
        "task_input": task,
        "history": memory.messages,
        "final_text": memory.final_assistant_text(),
    }