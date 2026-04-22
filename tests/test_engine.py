from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
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


def _with_fake_turns(turns, fn):
    original = engine.run_assistant_turn

    def fake_run_assistant_turn(history, tool_schemas, config):
        _ = history
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
    assert rows[1]["tool_call_count"] == 1
    assert rows[2]["tool_name"] == "bash"
    assert rows[2]["params"] == {"command": "printf hello"}
    assert rows[6]["tool_call_count"] == 0


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


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_run_episode_creates_trace_file(tmp_path)
        test_run_episode_writes_multi_step_events(tmp_path)
        test_run_episode_tracks_history_and_task_input(tmp_path)
    print("All engine tests passed.")


if __name__ == "__main__":
    main()
