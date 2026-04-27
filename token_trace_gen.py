"""Generate a vanilla token trace from a finished episode.

A token trace is a JSONL view focused on LLM-call timing and token volumes,
with calls from the parent episode and all subagent descendants flattened
onto a single timeline anchored at the root episode's start (t=0).

File layout (`<root_stem>.tokens.jsonl`):
  - Line 1:  header dict {root_episode_id, model, started_at, n_calls}
  - Lines 2+: one record per LLM call, sorted ascending by t_ms

Per-record fields:
  t_ms          — call start, ms relative to root episode start
  ttft_ms       — time to first token (None if the call produced no tokens)
  duration_ms   — total LLM call latency
  in_tokens     — total prompt tokens (the cached prefix is counted here too)
  cached_tokens — prefix-cache hit count, if the provider exposed it; else None
  out_tokens    — completion tokens

The header's `model` is the root episode's model. Subagent model overrides
are not represented per-record — keep the trace schema flat. If you start
running mixed-model trees, revisit.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _collect_llm_calls(trace_path: Path) -> list[dict[str, Any]]:
    """Walk a trace + all subagent traces; return all llm_call events."""
    rows = _load_trace(trace_path)
    calls = [r for r in rows if r.get("type") == "llm_call"]

    parent_dir = trace_path.parent
    for r in rows:
        if r.get("type") != "subagent_end":
            continue
        child_path_str = r.get("child_trace_path")
        if not child_path_str:
            continue
        child_path = Path(child_path_str)
        if not child_path.is_absolute():
            # Per-root-episode dir layout: child trace is a sibling of parent.
            child_path = parent_dir / child_path.name
        if child_path.exists():
            calls.extend(_collect_llm_calls(child_path))
    return calls


def _llm_call_to_record(call: dict[str, Any], t0: datetime) -> dict[str, Any]:
    # The llm_call event is emitted after run_assistant_turn returns, so its
    # timestamp ≈ end-of-call. Subtract latency to recover the start anchor.
    end_ms = int((datetime.fromisoformat(call["timestamp"]) - t0).total_seconds() * 1000)
    duration_ms = int(call.get("latency_ms", 0))
    t_ms = end_ms - duration_ms
    ttft_ms = call.get("ttft_ms")
    cached = call.get("cached_tokens")
    return {
        "t_ms": t_ms,
        "ttft_ms": int(ttft_ms) if ttft_ms is not None else None,
        "duration_ms": duration_ms,
        "in_tokens": int(call.get("prompt_tokens", 0)),
        "cached_tokens": int(cached) if cached is not None else None,
        "out_tokens": int(call.get("completion_tokens", 0)),
    }


def write_tokens_for(root_trace_path: Path) -> Path:
    """Generate the token trace for a finished root episode. Returns the
    path of the JSONL file written next to the root trace."""
    root_path = Path(root_trace_path)
    rows = _load_trace(root_path)
    if not rows or rows[0].get("type") != "episode_start":
        raise ValueError(f"{root_path}: missing episode_start as first event")

    start = rows[0]
    t0 = datetime.fromisoformat(start["timestamp"])

    llm_calls = _collect_llm_calls(root_path)
    records = sorted(
        (_llm_call_to_record(c, t0) for c in llm_calls),
        key=lambda r: r["t_ms"],
    )

    header = {
        "root_episode_id": start.get("episode_id"),
        "model": start.get("model"),
        "started_at": start["timestamp"],
        "n_calls": len(records),
    }

    out_path = root_path.parent / f"{root_path.stem}.tokens.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for r in records:
            f.write(json.dumps(r) + "\n")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trace", type=Path, help="Path to the root episode's .jsonl trace")
    args = p.parse_args(argv)

    if not args.trace.exists():
        print(f"Error: trace file not found: {args.trace}", file=sys.stderr)
        return 1

    try:
        out = write_tokens_for(args.trace)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
