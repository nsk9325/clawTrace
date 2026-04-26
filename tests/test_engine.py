from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path


def _load_module(module_name: str, file_name: str):
    module_path = Path(__file__).resolve().parent.parent / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("config", "config.py")
llm = _load_module("llm", "llm.py")
_load_module("tools", "tools.py")
_load_module("trace", "trace.py")
engine = _load_module("engine", "engine.py")


def _build_executor(tmp_path, cfg_overrides=None):
    import config as config_mod
    from trace import TraceWriter

    data = {**config_mod.DEFAULT_CONFIG, **(cfg_overrides or {})}
    cfg = config_mod.RunConfig.from_dict(data)
    episode = engine.Episode.new_root()
    budget = engine.SubagentBudget(cfg)
    writer = TraceWriter(tmp_path / "exec.jsonl")
    return engine.ToolExecutor(cfg, writer, episode, budget), writer


def test_can_parallelize_requires_allow_flag(tmp_path):
    executor, writer = _build_executor(tmp_path, {"allow_parallel_tools": False})
    try:
        batch = [
            {"id": "a", "name": "read_file", "input": {"file_path": "x"}},
            {"id": "b", "name": "glob", "input": {"pattern": "*"}},
        ]
        assert executor._can_parallelize(batch) is False
    finally:
        writer.close()


def test_can_parallelize_requires_all_concurrent_safe(tmp_path):
    executor, writer = _build_executor(tmp_path, {"allow_parallel_tools": True})
    try:
        batch = [
            {"id": "a", "name": "read_file", "input": {"file_path": "x"}},
            {"id": "b", "name": "bash", "input": {"command": "echo hi"}},
        ]
        assert executor._can_parallelize(batch) is False
    finally:
        writer.close()


def test_can_parallelize_permits_all_safe_batch(tmp_path):
    executor, writer = _build_executor(tmp_path, {"allow_parallel_tools": True})
    try:
        batch = [
            {"id": "a", "name": "read_file", "input": {"file_path": "x"}},
            {"id": "b", "name": "glob", "input": {"pattern": "*"}},
        ]
        assert executor._can_parallelize(batch) is True
    finally:
        writer.close()


def test_parallel_tools_overlap_in_time(tmp_path):
    import time as time_mod

    trace_path = tmp_path / "episode_parallel.jsonl"
    turns = [
        llm.AssistantTurn(
            text="",
            tool_calls=[
                {"id": "call_1", "name": "read_file", "input": {"file_path": "a"}},
                {"id": "call_2", "name": "read_file", "input": {"file_path": "b"}},
            ],
            input_tokens=1, output_tokens=1, latency_ms=1,
            ttft_ms=0, prefill_time_ms=0, decode_time_ms=1,
            measurement="client_streaming",
        ),
        llm.AssistantTurn(
            text="done", tool_calls=[],
            input_tokens=1, output_tokens=1, latency_ms=1,
            ttft_ms=0, prefill_time_ms=0, decode_time_ms=1,
            measurement="client_streaming",
        ),
    ]

    original_execute = engine.execute_tool
    def slow_execute(name, params, cfg, context=None):
        time_mod.sleep(0.05)
        return "(empty file)"
    engine.execute_tool = slow_execute
    try:
        _with_fake_turns(
            turns,
            lambda: engine.run_episode(
                config={"allow_parallel_tools": True, "max_parallel_tools": 4},
                trace_path=trace_path,
                task_input="go",
            ),
        )
    finally:
        engine.execute_tool = original_execute

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    tool_events = sorted((r for r in rows if r["type"] == "tool_call"), key=lambda r: r["tool_call_id"])
    assert len(tool_events) == 2
    a, b = tool_events
    assert b["started_at_ms"] < a["ended_at_ms"]


def _with_fake_turns(turns, fn):
    original = engine.run_assistant_turn

    def fake_run_assistant_turn(memory, tool_schemas, config):
        _ = memory
        _ = tool_schemas
        _ = config
        return turns.pop(0)

    engine.run_assistant_turn = fake_run_assistant_turn
    try:
        return fn()
    finally:
        engine.run_assistant_turn = original


def test_run_episode_creates_trace_file(tmp_path):
    trace_path = tmp_path / "episode_create.jsonl"
    turns = [
        llm.AssistantTurn(
            text="I should inspect the directory first.",
            tool_calls=[{"id": "call_1", "name": "bash", "input": {"command": "printf hello"}}],
            input_tokens=10,
            output_tokens=5,
            latency_ms=12,
            ttft_ms=3,
            prefill_time_ms=3,
            decode_time_ms=9,
            measurement="client_streaming",
        ),
        llm.AssistantTurn(
            text="I am done.",
            tool_calls=[],
            input_tokens=12,
            output_tokens=4,
            latency_ms=10,
            ttft_ms=2,
            prefill_time_ms=2,
            decode_time_ms=8,
            measurement="client_streaming",
        ),
    ]

    def run():
        return engine.run_episode(
            config={"max_steps": 3},
            trace_path=trace_path,
            task_input="Say hello",
        )

    result = _with_fake_turns(turns, run)

    assert result["status"] == "completed"
    assert result["step_count"] == 2
    assert trace_path.exists()


def test_run_episode_writes_multi_step_events(tmp_path):
    trace_path = tmp_path / "episode_events.jsonl"
    turns = [
        llm.AssistantTurn(
            text="I should inspect the directory first.",
            tool_calls=[{"id": "call_1", "name": "bash", "input": {"command": "printf hello"}}],
            input_tokens=10,
            output_tokens=5,
            latency_ms=12,
            ttft_ms=3,
            prefill_time_ms=3,
            decode_time_ms=9,
            measurement="client_streaming",
        ),
        llm.AssistantTurn(
            text="I am done.",
            tool_calls=[],
            input_tokens=12,
            output_tokens=4,
            latency_ms=10,
            ttft_ms=2,
            prefill_time_ms=2,
            decode_time_ms=8,
            measurement="client_streaming",
        ),
    ]

    _with_fake_turns(
        turns,
        lambda: engine.run_episode(trace_path=trace_path, task_input="Say hello"),
    )

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    event_types = [row["type"] for row in rows]

    assert event_types == [
        "episode_start",
        "step_start",
        "llm_call",
        "tool_call",
        "context_update",
        "step_end",
        "step_start",
        "llm_call",
        "context_update",
        "step_end",
        "episode_end",
    ]
    assert rows[2]["tool_call_count"] == 1
    assert rows[3]["tool_name"] == "bash"
    assert rows[3]["params"] == {"command": "printf hello"}
    assert rows[7]["tool_call_count"] == 0


def test_run_episode_tracks_history_and_task_input(tmp_path):
    trace_path = tmp_path / "episode_task.jsonl"
    turns = [
        llm.AssistantTurn(
            text="I should inspect the directory first.",
            tool_calls=[],
            input_tokens=10,
            output_tokens=5,
            latency_ms=12,
            ttft_ms=3,
            prefill_time_ms=3,
            decode_time_ms=9,
            measurement="client_streaming",
        ),
    ]

    result = _with_fake_turns(
        turns,
        lambda: engine.run_episode(trace_path=trace_path, task_input="Read the README file"),
    )

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]

    assert rows[0]["task_input"] == "Read the README file"
    assert result["history"][0] == {"role": "user", "content": "Read the README file"}
    assert result["history"][1]["role"] == "assistant"


def test_tool_call_event_has_timing_offsets(tmp_path):
    trace_path = tmp_path / "episode_timing.jsonl"
    turns = [
        llm.AssistantTurn(
            text="",
            tool_calls=[{"id": "call_1", "name": "bash", "input": {"command": "printf hi"}}],
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            ttft_ms=0,
            prefill_time_ms=0,
            decode_time_ms=1,
            measurement="client_streaming",
        ),
        llm.AssistantTurn(
            text="done",
            tool_calls=[],
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            ttft_ms=0,
            prefill_time_ms=0,
            decode_time_ms=1,
            measurement="client_streaming",
        ),
    ]

    _with_fake_turns(
        turns,
        lambda: engine.run_episode(trace_path=trace_path, task_input="hi"),
    )

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    tool_events = [r for r in rows if r["type"] == "tool_call"]
    assert len(tool_events) == 1
    tc = tool_events[0]
    assert "started_at_ms" in tc
    assert "ended_at_ms" in tc
    assert tc["started_at_ms"] >= 0
    assert tc["ended_at_ms"] >= tc["started_at_ms"]
    assert tc["latency_ms"] == tc["ended_at_ms"] - tc["started_at_ms"]


def test_run_episode_ignores_finish_reason_for_loop_control(tmp_path):
    trace_path = tmp_path / "episode_finish_reason.jsonl"
    turns = [
        llm.AssistantTurn(
            text="I should inspect the directory first.",
            tool_calls=[{"id": "call_1", "name": "bash", "input": {"command": "printf hello"}}],
            input_tokens=10,
            output_tokens=5,
            latency_ms=12,
            ttft_ms=3,
            prefill_time_ms=3,
            decode_time_ms=9,
            measurement="client_streaming",
            finish_reason="stop",
        ),
        llm.AssistantTurn(
            text="I am done.",
            tool_calls=[],
            input_tokens=12,
            output_tokens=4,
            latency_ms=10,
            ttft_ms=2,
            prefill_time_ms=2,
            decode_time_ms=8,
            measurement="client_streaming",
        ),
    ]

    result = _with_fake_turns(
        turns,
        lambda: engine.run_episode(trace_path=trace_path, task_input="Say hello"),
    )

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    event_types = [row["type"] for row in rows]

    assert result["step_count"] == 2
    assert event_types == [
        "episode_start",
        "step_start",
        "llm_call",
        "tool_call",
        "context_update",
        "step_end",
        "step_start",
        "llm_call",
        "context_update",
        "step_end",
        "episode_end",
    ]
    assert rows[2]["finish_reason"] == "stop"
    assert rows[3]["tool_name"] == "bash"
    assert rows[4]["executed_tool_call_count"] == 1
    assert rows[5]["stop_reason"] is None
    assert rows[9]["stop_reason"] == "no_tool_calls"
    assert rows[10]["stop_reason"] == "no_tool_calls"


def test_subagent_budget_depth_rejects_when_parent_depth_hits_cap():
    import config as config_mod
    cfg = config_mod.RunConfig.from_dict({
        **config_mod.DEFAULT_CONFIG,
        "max_subagent_depth": 2,
        "max_concurrent_subagents": 0,
    })
    budget = engine.SubagentBudget(cfg)
    assert budget.reserve("parent_a", parent_depth=0).ok
    assert budget.reserve("parent_a", parent_depth=1).ok
    rejected = budget.reserve("parent_a", parent_depth=2)
    assert not rejected.ok
    assert "depth" in rejected.reason


def test_subagent_budget_concurrent_release_restores_slot():
    import config as config_mod
    cfg = config_mod.RunConfig.from_dict({
        **config_mod.DEFAULT_CONFIG,
        "max_concurrent_subagents": 1,
    })
    budget = engine.SubagentBudget(cfg)
    first = budget.reserve("parent_a", parent_depth=0)
    assert first.ok
    second = budget.reserve("parent_a", parent_depth=0)
    assert not second.ok
    budget.release()
    third = budget.reserve("parent_a", parent_depth=0)
    assert third.ok


def test_subagent_budget_per_parent_cap():
    import config as config_mod
    cfg = config_mod.RunConfig.from_dict({
        **config_mod.DEFAULT_CONFIG,
        "max_subagents_per_parent": 2,
        "max_concurrent_subagents": 0,
    })
    budget = engine.SubagentBudget(cfg)
    assert budget.reserve("parent_a", parent_depth=0).ok
    assert budget.reserve("parent_a", parent_depth=0).ok
    over = budget.reserve("parent_a", parent_depth=0)
    assert not over.ok
    assert "per_parent" in over.reason
    # different parent should still have its own quota
    assert budget.reserve("parent_b", parent_depth=0).ok


def test_spawn_subagent_end_to_end(tmp_path):
    import subagent as _subagent_mod  # noqa: F401 — registers spawn_subagent

    trace_path = tmp_path / "parent.jsonl"
    turns = [
        llm.AssistantTurn(
            text="",
            tool_calls=[{
                "id": "call_1", "name": "spawn_subagent",
                "input": {"task": "do the subtask"},
            }],
            input_tokens=1, output_tokens=1, latency_ms=1,
            ttft_ms=0, prefill_time_ms=0, decode_time_ms=1,
            measurement="client_streaming",
        ),
        llm.AssistantTurn(
            text="child finished the subtask", tool_calls=[],
            input_tokens=1, output_tokens=1, latency_ms=1,
            ttft_ms=0, prefill_time_ms=0, decode_time_ms=1,
            measurement="client_streaming",
        ),
        llm.AssistantTurn(
            text="parent done", tool_calls=[],
            input_tokens=1, output_tokens=1, latency_ms=1,
            ttft_ms=0, prefill_time_ms=0, decode_time_ms=1,
            measurement="client_streaming",
        ),
    ]

    result = _with_fake_turns(
        turns,
        lambda: engine.run_episode(
            config={
                "enable_subagents": True,
                "max_subagent_depth": 2,
                "max_concurrent_subagents": 4,
                "output_dir": str(tmp_path),
            },
            trace_path=trace_path,
            task_input="parent task",
        ),
    )

    parent_rows = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines()]
    event_types = [r["type"] for r in parent_rows]
    assert "subagent_start" in event_types
    assert "subagent_end" in event_types

    end_event = next(r for r in parent_rows if r["type"] == "subagent_end")
    child_trace_file = Path(end_event["child_trace_path"])
    assert child_trace_file.exists()

    child_rows = [json.loads(l) for l in child_trace_file.read_text(encoding="utf-8").splitlines()]
    first_child_event = child_rows[0]
    assert first_child_event["depth"] == 1
    assert first_child_event["parent_episode_id"] == result["episode_id"]
    assert first_child_event["root_episode_id"] == result["episode_id"]


def test_spawn_subagent_disabled_returns_error_when_called(tmp_path):
    import subagent as _subagent_mod  # noqa: F401
    import config as config_mod

    cfg = config_mod.RunConfig.from_dict(config_mod.DEFAULT_CONFIG)
    executor, writer = _build_executor(tmp_path)
    try:
        result = executor._run_one(
            {"id": "c1", "name": "spawn_subagent", "input": {"task": "x"}},
            step_id=0,
            origin=time.perf_counter(),
        )
        assert result.exit_status == "error"
        assert "disabled" in result.output
    finally:
        writer.close()


def test_runs_per_task_produces_distinct_episodes(tmp_path):
    runner = _load_module("runner", "runner.py")

    turns = [
        llm.AssistantTurn(
            text=f"done {i}", tool_calls=[],
            input_tokens=1, output_tokens=1, latency_ms=1,
            ttft_ms=0, prefill_time_ms=0, decode_time_ms=1,
            measurement="client_streaming",
        )
        for i in range(3)
    ]

    results = _with_fake_turns(
        turns,
        lambda: runner._run_all(
            "task",
            {"runs_per_task": 3, "output_dir": str(tmp_path)},
        ),
    )

    assert len(results) == 3
    episode_ids = [r[0]["episode_id"] for r in results]
    assert len(set(episode_ids)) == 3
    trace_files = sorted(tmp_path.glob("episode_*.jsonl"))
    assert len(trace_files) == 3


def test_max_steps_reached_reports_incomplete_status(tmp_path):
    trace_path = tmp_path / "episode_incomplete.jsonl"
    turns = [
        llm.AssistantTurn(
            text="",
            tool_calls=[{"id": f"c{i}", "name": "read_file", "input": {"file_path": "x"}}],
            input_tokens=1, output_tokens=1, latency_ms=1,
            ttft_ms=0, prefill_time_ms=0, decode_time_ms=1,
            measurement="client_streaming",
        )
        for i in range(5)
    ]

    result = _with_fake_turns(
        turns,
        lambda: engine.run_episode(
            config={"max_steps": 2},
            trace_path=trace_path,
            task_input="x",
        ),
    )

    assert result["status"] == "incomplete"
    assert result["stop_reason"] == "max_steps_reached"
    rows = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines()]
    episode_end = next(r for r in rows if r["type"] == "episode_end")
    assert episode_end["status"] == "incomplete"


def test_subagent_crash_emits_subagent_end_with_error(tmp_path, monkeypatch):
    import subagent as subagent_mod  # noqa: F401 — registers spawn_subagent

    trace_path = tmp_path / "parent_crash.jsonl"
    turns = [
        llm.AssistantTurn(
            text="",
            tool_calls=[{
                "id": "call_1", "name": "spawn_subagent",
                "input": {"task": "do it"},
            }],
            input_tokens=1, output_tokens=1, latency_ms=1,
            ttft_ms=0, prefill_time_ms=0, decode_time_ms=1,
            measurement="client_streaming",
        ),
        llm.AssistantTurn(
            text="parent done after crash", tool_calls=[],
            input_tokens=1, output_tokens=1, latency_ms=1,
            ttft_ms=0, prefill_time_ms=0, decode_time_ms=1,
            measurement="client_streaming",
        ),
    ]

    original = engine.run_episode

    def crashing_run_episode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(engine, "run_episode", crashing_run_episode)

    _with_fake_turns(
        turns,
        lambda: original(
            config={
                "enable_subagents": True,
                "max_subagent_depth": 2,
                "max_concurrent_subagents": 4,
                "output_dir": str(tmp_path),
            },
            trace_path=trace_path,
            task_input="parent task",
        ),
    )

    rows = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines()]
    start_events = [r for r in rows if r["type"] == "subagent_start"]
    end_events = [r for r in rows if r["type"] == "subagent_end"]
    assert len(start_events) == len(end_events) == 1
    assert end_events[0]["child_status"] == "crashed"
    assert "boom" in end_events[0]["error"]


def test_episode_start_is_first_and_carries_cfg(tmp_path):
    trace_path = tmp_path / "episode_start.jsonl"
    turns = [
        llm.AssistantTurn(
            text="done", tool_calls=[],
            input_tokens=1, output_tokens=1, latency_ms=1,
            ttft_ms=0, prefill_time_ms=0, decode_time_ms=1,
            measurement="client_streaming",
        ),
    ]

    _with_fake_turns(
        turns,
        lambda: engine.run_episode(
            config={"system_prompt": "minimal"},
            trace_path=trace_path,
            task_input="hello",
        ),
    )

    rows = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["type"] == "episode_start"
    assert rows[0]["task_input"] == "hello"
    assert rows[0]["system_prompt_name"] == "minimal"
    assert rows[0]["system_prompt_chars"] > 0
    assert rows[0]["cfg"]["system_prompt"] == "minimal"
    assert rows[0]["workload_info"] is None


def test_episode_start_marks_custom_system_prompt(tmp_path):
    trace_path = tmp_path / "episode_custom.jsonl"
    turns = [
        llm.AssistantTurn(
            text="done", tool_calls=[],
            input_tokens=1, output_tokens=1, latency_ms=1,
            ttft_ms=0, prefill_time_ms=0, decode_time_ms=1,
            measurement="client_streaming",
        ),
    ]

    _with_fake_turns(
        turns,
        lambda: engine.run_episode(
            config={"system_prompt": "totally custom prompt text"},
            trace_path=trace_path,
            task_input="x",
        ),
    )

    rows = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["system_prompt_name"] == "<custom>"


def test_episode_start_workload_info_passes_through(tmp_path):
    trace_path = tmp_path / "episode_workload.jsonl"
    turns = [
        llm.AssistantTurn(
            text="done", tool_calls=[],
            input_tokens=1, output_tokens=1, latency_ms=1,
            ttft_ms=0, prefill_time_ms=0, decode_time_ms=1,
            measurement="client_streaming",
        ),
    ]

    workload = {"instance_id": "django__django-15555", "repo": "django/django"}
    _with_fake_turns(
        turns,
        lambda: engine.run_episode(
            trace_path=trace_path,
            task_input="x",
            workload_info=workload,
        ),
    )

    rows = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["workload_info"] == workload


def test_step_start_no_longer_duplicates_task_input(tmp_path):
    trace_path = tmp_path / "episode_no_dup.jsonl"
    turns = [
        llm.AssistantTurn(
            text="done", tool_calls=[],
            input_tokens=1, output_tokens=1, latency_ms=1,
            ttft_ms=0, prefill_time_ms=0, decode_time_ms=1,
            measurement="client_streaming",
        ),
    ]

    _with_fake_turns(
        turns,
        lambda: engine.run_episode(
            trace_path=trace_path,
            task_input="some task",
        ),
    )

    rows = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines()]
    step_starts = [r for r in rows if r["type"] == "step_start"]
    assert step_starts
    assert all("task_input" not in r for r in step_starts)


def test_future_subagent_parallelism_modes_raise():
    import pytest
    for mode in ("worktree", "shared+optimistic"):
        with pytest.raises(NotImplementedError):
            engine.run_episode(
                config={"subagent_parallelism": mode},
                task_input="x",
            )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_run_episode_creates_trace_file(tmp_path)
        test_run_episode_writes_multi_step_events(tmp_path)
        test_run_episode_tracks_history_and_task_input(tmp_path)
        test_tool_call_event_has_timing_offsets(tmp_path)
        test_runs_per_task_produces_distinct_episodes(tmp_path)
        test_run_episode_ignores_finish_reason_for_loop_control(tmp_path)
    print("All engine tests passed.")


if __name__ == "__main__":
    main()
