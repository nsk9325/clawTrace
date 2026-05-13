from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load(name: str, fname: str):
    path = Path(__file__).resolve().parent.parent / fname
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


token_trace_gen = _load("token_trace_gen", "token_trace_gen.py")


def _episode_start(episode_id="ep_root", model="gpt-4o-mini",
                   wall_clock_start="2026-04-26T12:00:00.000000+00:00", depth=0,
                   t_ms=0):
    return {
        "type": "episode_start",
        "t_ms": t_ms,
        "wall_clock_start": wall_clock_start,
        "episode_id": episode_id,
        "model": model,
        "depth": depth,
    }


def _llm_call(t_ms, latency_ms, ttft_ms, prompt_tokens, completion_tokens,
              cached_tokens=None, episode_id="ep_root", step_id=0):
    return {
        "type": "llm_call",
        "t_ms": t_ms,
        "step_id": step_id,
        "episode_id": episode_id,
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
    }


def _episode_end(episode_id="ep_root", t_ms=10000):
    return {"type": "episode_end", "t_ms": t_ms, "episode_id": episode_id}


def test_single_episode_two_calls(tmp_path):
    trace_path = tmp_path / "ep_root.jsonl"
    rows = [
        _episode_start(),
        _llm_call(
            t_ms=1450,
            latency_ms=1450, ttft_ms=320,
            prompt_tokens=1842, completion_tokens=47, cached_tokens=0,
        ),
        _llm_call(
            t_ms=4200, step_id=1,
            latency_ms=1100, ttft_ms=280,
            prompt_tokens=2030, completion_tokens=89, cached_tokens=1800,
        ),
        _episode_end(t_ms=5000),
    ]
    trace_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out = token_trace_gen.write_tokens_for(trace_path)
    assert out == tmp_path / "ep_root.tokens.jsonl"

    lines = out.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    records = [json.loads(l) for l in lines[1:]]

    assert header == {
        "root_episode_id": "ep_root",
        "model": "gpt-4o-mini",
        "started_at": "2026-04-26T12:00:00.000000+00:00",
        "n_calls": 2,
    }
    assert len(records) == 2
    # First call: end at +1450ms, latency 1450ms → started at 0
    assert records[0] == {
        "t_ms": 0,
        "ttft_ms": 320,
        "duration_ms": 1450,
        "in_tokens": 1842,
        "cached_tokens": 0,
        "out_tokens": 47,
    }
    # Second call: end at +4200ms, latency 1100ms → started at 3100
    assert records[1]["t_ms"] == 3100
    assert records[1]["cached_tokens"] == 1800


def test_subagent_calls_merged_on_root_timeline(tmp_path):
    child_path = tmp_path / "ep_child.jsonl"
    child_rows = [
        # Child episode starts t_ms=0 in its own frame (parent tracks the
        # root-relative offset on subagent_end).
        _episode_start(episode_id="ep_child", depth=1, t_ms=0),
        _llm_call(
            t_ms=500,
            latency_ms=500, ttft_ms=100,
            prompt_tokens=300, completion_tokens=10,
            episode_id="ep_child",
        ),
        _episode_end(episode_id="ep_child", t_ms=1000),
    ]
    child_path.write_text("\n".join(json.dumps(r) for r in child_rows) + "\n", encoding="utf-8")

    parent_path = tmp_path / "ep_root.jsonl"
    parent_rows = [
        _episode_start(),
        _llm_call(
            t_ms=1000,
            latency_ms=1000, ttft_ms=200,
            prompt_tokens=500, completion_tokens=20,
        ),
        {
            "type": "subagent_end",
            "t_ms": 3000,
            "child_episode_offset_ms": 2000,
            "child_trace_path": str(child_path),
        },
        _episode_end(t_ms=4000),
    ]
    parent_path.write_text("\n".join(json.dumps(r) for r in parent_rows) + "\n", encoding="utf-8")

    out = token_trace_gen.write_tokens_for(parent_path)
    lines = out.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    records = [json.loads(l) for l in lines[1:]]

    assert header["n_calls"] == 2
    assert len(records) == 2
    # Records sorted by t_ms ascending: parent at 0, child at 2000
    assert records[0]["t_ms"] == 0
    assert records[1]["t_ms"] == 2000
    assert records[1]["in_tokens"] == 300


def test_missing_cached_tokens_serialized_as_null(tmp_path):
    trace_path = tmp_path / "ep_root.jsonl"
    rows = [
        _episode_start(),
        _llm_call(
            t_ms=1000,
            latency_ms=1000, ttft_ms=200,
            prompt_tokens=100, completion_tokens=10,
            cached_tokens=None,
        ),
        _episode_end(t_ms=2000),
    ]
    trace_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out = token_trace_gen.write_tokens_for(trace_path)
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[1])
    assert record["cached_tokens"] is None


def test_falls_back_to_iso_timestamps_for_legacy_traces(tmp_path):
    """Legacy traces have no ``t_ms`` — only ISO ``timestamp``. Verify the
    ISO fallback path still produces the same root-relative t_ms records."""
    trace_path = tmp_path / "ep_root.jsonl"
    rows = [
        {
            "type": "episode_start",
            "timestamp": "2026-04-26T12:00:00.000000+00:00",
            "episode_id": "ep_root",
            "model": "gpt-4o-mini",
            "depth": 0,
        },
        {
            "type": "llm_call",
            "timestamp": "2026-04-26T12:00:01.450000+00:00",
            "step_id": 0,
            "episode_id": "ep_root",
            "latency_ms": 1450,
            "ttft_ms": 320,
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "cached_tokens": None,
        },
        {
            "type": "episode_end",
            "timestamp": "2026-04-26T12:00:02.000000+00:00",
            "episode_id": "ep_root",
        },
    ]
    trace_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out = token_trace_gen.write_tokens_for(trace_path)
    record = json.loads(out.read_text(encoding="utf-8").splitlines()[1])
    assert record["t_ms"] == 0
    assert record["duration_ms"] == 1450
