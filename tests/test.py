from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


def _load_trace_module():
    trace_path = Path(__file__).resolve().parent.parent / "trace.py"
    spec = importlib.util.spec_from_file_location("clawtrace_trace", trace_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {trace_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


trace = _load_trace_module()


def test_make_event_contains_core_fields():
    event = trace.make_event(
        "step_start",
        episode_id="ep_1",
        run_id="run_1",
        step_id=0,
        action_type="start",
    )

    assert event["type"] == "step_start"
    assert event["episode_id"] == "ep_1"
    assert event["run_id"] == "run_1"
    assert event["step_id"] == 0
    assert event["action_type"] == "start"
    assert "timestamp" in event


def test_trace_writer_writes_jsonl(tmp_path):
    output_path = tmp_path / "trace.jsonl"
    event = trace.make_event(
        "llm_call",
        episode_id="ep_1",
        run_id="run_1",
        step_id=1,
        latency_ms=123,
    )

    with trace.TraceWriter(output_path) as writer:
        writer.write(event)

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    stored = json.loads(lines[0])
    assert stored["type"] == "llm_call"
    assert stored["episode_id"] == "ep_1"
    assert stored["run_id"] == "run_1"
    assert stored["step_id"] == 1
    assert stored["latency_ms"] == 123


def main() -> None:
    test_make_event_contains_core_fields()
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_trace_writer_writes_jsonl(Path(tmp_dir))
    print("All tests passed.")


if __name__ == "__main__":
    main()
