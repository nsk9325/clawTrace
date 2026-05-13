from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module(module_name: str, file_name: str):
    module_path = Path(__file__).resolve().parent.parent / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


analyzer = _load_module("analyzer", "analyzer.py")


def _episode_start(episode_id="ep_a", **overrides):
    base = {
        "t_ms": 0,
        "wall_clock_start": "2026-04-26T10:00:00.000000+00:00",
        "type": "episode_start",
        "step_id": None,
        "episode_id": episode_id,
        "run_id": "run_a",
        "depth": 0,
        "root_episode_id": episode_id,
        "parent_episode_id": None,
        "parent_run_id": None,
        "parent_step_id": None,
        "task_input": "do it",
        "model": "gpt-4o-mini",
        "backend": "openai",
        "system_prompt_name": "agent",
        "system_prompt_chars": 500,
        "cfg": {},
        "workload_info": None,
    }
    base.update(overrides)
    return base


def _episode_end(episode_id="ep_a", **overrides):
    base = {
        "t_ms": 5000,
        "type": "episode_end",
        "step_id": None,
        "episode_id": episode_id,
        "run_id": "run_a",
        "depth": 0,
        "root_episode_id": episode_id,
        "status": "completed",
        "stop_reason": "no_tool_calls",
        "step_count": 1,
        "history_length": 2,
    }
    base.update(overrides)
    return base


def _llm_call(latency_ms=100, prompt_tokens=50, completion_tokens=10, **overrides):
    base = {
        "type": "llm_call",
        "step_id": 0,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    base.update(overrides)
    return base


def _tool_call(step_id, tool_name, started_at_ms, ended_at_ms, **overrides):
    base = {
        "type": "tool_call",
        "step_id": step_id,
        "tool_name": tool_name,
        "started_at_ms": started_at_ms,
        "ended_at_ms": ended_at_ms,
        "latency_ms": ended_at_ms - started_at_ms,
        "exit_status": "ok",
    }
    base.update(overrides)
    return base


def test_summarize_minimal_trace():
    rows = [_episode_start(), _llm_call(latency_ms=200), _episode_end()]
    s = analyzer.summarize(rows)
    assert s.episode_id == "ep_a"
    assert s.status == "completed"
    assert s.step_count == 1
    assert s.total_llm_ms == 200
    assert s.total_tool_ms == 0
    assert s.total_wall_ms == 5000  # 5 seconds between start and end
    assert s.prompt_tokens_total == 50
    assert s.completion_tokens_total == 10
    assert s.tool_call_count_by_name == {}
    assert s.subagent_count == 0


def test_summarize_falls_back_to_iso_timestamps_for_legacy_traces():
    """Pre-t_ms traces have only ``timestamp`` (ISO). The analyzer's
    ``_episode_wall_ms`` helper must still compute wall time from those."""
    start = _episode_start()
    end = _episode_end()
    # Strip the new fields, leave only the legacy ISO timestamps.
    start.pop("t_ms")
    start["timestamp"] = start.pop("wall_clock_start")
    end.pop("t_ms")
    end["timestamp"] = "2026-04-26T10:00:05.000000+00:00"
    s = analyzer.summarize([start, _llm_call(latency_ms=200), end])
    assert s.total_wall_ms == 5000


def test_summarize_serial_tools_sum_to_wall_time():
    rows = [
        _episode_start(),
        _tool_call(step_id=0, tool_name="read_file", started_at_ms=0, ended_at_ms=50),
        _tool_call(step_id=0, tool_name="bash", started_at_ms=50, ended_at_ms=150),
        _episode_end(),
    ]
    s = analyzer.summarize(rows)
    # serial: max(150, 50) - min(0, 50) = 150 - 0 = 150 (sum of latencies)
    assert s.total_tool_ms == 150
    assert s.tool_call_count_by_name == {"read_file": 1, "bash": 1}


def test_summarize_parallel_tools_take_max_not_sum():
    rows = [
        _episode_start(),
        _tool_call(step_id=0, tool_name="read_file", started_at_ms=0, ended_at_ms=80),
        _tool_call(step_id=0, tool_name="glob", started_at_ms=2, ended_at_ms=100),
        _episode_end(),
    ]
    s = analyzer.summarize(rows)
    # parallel: max(80, 100) - min(0, 2) = 100 - 0 = 100, not 80+98=178
    assert s.total_tool_ms == 100


def test_summarize_multi_step_tool_times_sum_across_steps():
    rows = [
        _episode_start(),
        _tool_call(step_id=0, tool_name="read_file", started_at_ms=0, ended_at_ms=50),
        _tool_call(step_id=1, tool_name="read_file", started_at_ms=0, ended_at_ms=30),
        _episode_end(step_count=2),
    ]
    s = analyzer.summarize(rows)
    assert s.total_tool_ms == 80


def test_summarize_pulls_workload_instance_id():
    start = _episode_start(workload_info={"instance_id": "django__django-1", "repo": "django/django"})
    rows = [start, _episode_end()]
    s = analyzer.summarize(rows)
    assert s.workload_instance_id == "django__django-1"


def test_summarize_counts_subagents():
    rows = [
        _episode_start(),
        {"type": "subagent_start", "step_id": 0, "child_episode_id": "ep_child_1"},
        {"type": "subagent_start", "step_id": 1, "child_episode_id": "ep_child_2"},
        _episode_end(),
    ]
    s = analyzer.summarize(rows)
    assert s.subagent_count == 2


def test_summarize_raises_on_empty_trace():
    with pytest.raises(ValueError):
        analyzer.summarize([])


def test_summarize_raises_when_episode_start_missing():
    rows = [_llm_call(), _episode_end()]
    with pytest.raises(ValueError):
        analyzer.summarize(rows)


def test_summarize_raises_when_episode_end_missing():
    rows = [_episode_start(), _llm_call()]
    with pytest.raises(ValueError):
        analyzer.summarize(rows)


def test_render_workload_with_payload_lists_fields():
    out = analyzer.render_workload({"instance_id": "x", "repo": "y/z"})
    assert "Workload:" in out
    assert "instance_id" in out
    assert "'x'" in out


def test_render_workload_with_none_says_none():
    assert analyzer.render_workload(None) == "Workload: (none)"


def test_render_config_lists_all_fields_sorted():
    out = analyzer.render_config({"max_steps": 20, "model": "gpt-4o-mini"})
    assert "Config:" in out
    assert "max_steps: 20" in out
    assert "model: 'gpt-4o-mini'" in out


def test_render_summary_includes_key_fields():
    rows = [_episode_start(), _llm_call(latency_ms=200), _episode_end()]
    s = analyzer.summarize(rows)
    out = analyzer.render_summary(s)
    assert "ep_a" in out
    assert "completed" in out
    assert "gpt-4o-mini" in out
    assert "Wall time:" in out
    assert "LLM time:" in out


def test_build_tree_no_subagents_returns_leaf(tmp_path):
    trace = tmp_path / "ep_a.jsonl"
    trace.write_text(
        "\n".join(json.dumps(r) for r in [_episode_start(), _episode_end()]) + "\n",
        encoding="utf-8",
    )
    node = analyzer.build_tree(trace)
    assert node.summary.episode_id == "ep_a"
    assert node.children == []
    assert node.missing == []


def test_build_tree_follows_child_trace_path(tmp_path):
    # Child trace
    child_trace = tmp_path / "ep_child.jsonl"
    child_trace.write_text(
        "\n".join(json.dumps(r) for r in [
            _episode_start(episode_id="ep_child", depth=1, parent_episode_id="ep_parent"),
            _episode_end(episode_id="ep_child", depth=1, root_episode_id="ep_parent"),
        ]) + "\n",
        encoding="utf-8",
    )
    # Parent trace references child
    parent_trace = tmp_path / "ep_parent.jsonl"
    parent_trace.write_text(
        "\n".join(json.dumps(r) for r in [
            _episode_start(episode_id="ep_parent"),
            {
                "type": "subagent_end",
                "step_id": 0,
                "child_episode_id": "ep_child",
                "child_trace_path": str(child_trace),
            },
            _episode_end(episode_id="ep_parent"),
        ]) + "\n",
        encoding="utf-8",
    )
    root = analyzer.build_tree(parent_trace)
    assert root.summary.episode_id == "ep_parent"
    assert len(root.children) == 1
    assert root.children[0].summary.episode_id == "ep_child"
    assert root.children[0].spawned_at_step == 0


def test_build_tree_resolves_relative_child_path_as_sibling(tmp_path):
    child_trace = tmp_path / "ep_child.jsonl"
    child_trace.write_text(
        "\n".join(json.dumps(r) for r in [
            _episode_start(episode_id="ep_child", depth=1),
            _episode_end(episode_id="ep_child"),
        ]) + "\n",
        encoding="utf-8",
    )
    parent_trace = tmp_path / "ep_parent.jsonl"
    parent_trace.write_text(
        "\n".join(json.dumps(r) for r in [
            _episode_start(episode_id="ep_parent"),
            {
                "type": "subagent_end",
                "step_id": 0,
                "child_episode_id": "ep_child",
                "child_trace_path": "some/random/cwd-relative/prefix/ep_child.jsonl",
            },
            _episode_end(episode_id="ep_parent"),
        ]) + "\n",
        encoding="utf-8",
    )
    root = analyzer.build_tree(parent_trace)
    assert len(root.children) == 1
    assert root.children[0].summary.episode_id == "ep_child"
    assert root.missing == []


def test_build_tree_records_missing_child(tmp_path):
    parent_trace = tmp_path / "ep_parent.jsonl"
    parent_trace.write_text(
        "\n".join(json.dumps(r) for r in [
            _episode_start(episode_id="ep_parent"),
            {
                "type": "subagent_end",
                "step_id": 1,
                "child_episode_id": "ep_missing",
                "child_trace_path": str(tmp_path / "does_not_exist.jsonl"),
            },
            _episode_end(episode_id="ep_parent"),
        ]) + "\n",
        encoding="utf-8",
    )
    root = analyzer.build_tree(parent_trace)
    assert root.children == []
    assert len(root.missing) == 1
    assert root.missing[0]["child_episode_id"] == "ep_missing"
    assert root.missing[0]["spawned_at_step"] == 1
    assert "not found" in root.missing[0]["error"]


def test_main_writes_analysis_file_next_to_trace(tmp_path):
    trace = tmp_path / "ep_a.jsonl"
    trace.write_text(
        "\n".join(json.dumps(r) for r in [_episode_start(), _episode_end()]) + "\n",
        encoding="utf-8",
    )
    rc = analyzer.main([str(trace)])
    assert rc == 0
    saved = tmp_path / "ep_a.analysis.txt"
    assert saved.exists()
    content = saved.read_text(encoding="utf-8")
    assert "ep_a" in content
    assert "completed" in content


def test_render_tree_contains_indented_children(tmp_path):
    child_trace = tmp_path / "ep_child.jsonl"
    child_trace.write_text(
        "\n".join(json.dumps(r) for r in [
            _episode_start(episode_id="ep_child", depth=1),
            _episode_end(episode_id="ep_child"),
        ]) + "\n",
        encoding="utf-8",
    )
    parent_trace = tmp_path / "ep_parent.jsonl"
    parent_trace.write_text(
        "\n".join(json.dumps(r) for r in [
            _episode_start(episode_id="ep_parent"),
            {
                "type": "subagent_end",
                "step_id": 2,
                "child_episode_id": "ep_child",
                "child_trace_path": str(child_trace),
            },
            _episode_end(episode_id="ep_parent"),
        ]) + "\n",
        encoding="utf-8",
    )
    root = analyzer.build_tree(parent_trace)
    out = analyzer.render_tree(root)
    assert "Subagent tree:" in out
    assert "ep_parent" in out
    assert "ep_child" in out
    assert "└─" in out  # tree connector for the (only, last) child
    assert "spawned at step 2" in out
