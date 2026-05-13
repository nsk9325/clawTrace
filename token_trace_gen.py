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


def _collect_llm_calls(
    trace_path: Path,
    *,
    anchor_in_root_ms: int,
    root_iso: str | None,
) -> list[tuple[dict[str, Any], int]]:
    """Walk a trace + subagent traces; return ``(call_event, end_ms_in_root)``
    tuples. Each end_ms_in_root is computed against the root episode start
    using ``t_ms`` (new traces) or ISO arithmetic (legacy fallback)."""
    rows = _load_trace(trace_path)
    if not rows or rows[0].get("type") != "episode_start":
        return []
    start = rows[0]

    if "t_ms" in start:
        start_t_ms = int(start["t_ms"])
        def to_root_ms(e: dict[str, Any]) -> int:
            return anchor_in_root_ms + (int(e["t_ms"]) - start_t_ms)
    else:
        if root_iso is None:
            return []
        root_dt = datetime.fromisoformat(root_iso)
        def to_root_ms(e: dict[str, Any]) -> int:
            return int(
                (datetime.fromisoformat(e["timestamp"]) - root_dt).total_seconds() * 1000
            )

    out: list[tuple[dict[str, Any], int]] = []
    parent_dir = trace_path.parent
    for r in rows:
        if r.get("type") == "llm_call":
            out.append((r, to_root_ms(r)))
        elif r.get("type") == "subagent_end":
            child_path_str = r.get("child_trace_path")
            if not child_path_str:
                continue
            child_path = Path(child_path_str)
            if not child_path.is_absolute():
                # Per-root-episode dir layout: child trace is a sibling of parent.
                child_path = parent_dir / child_path.name
            if not child_path.exists():
                continue
            child_offset = int(r.get("child_episode_offset_ms", 0))
            out.extend(_collect_llm_calls(
                child_path,
                anchor_in_root_ms=anchor_in_root_ms + child_offset,
                root_iso=root_iso,
            ))
    return out


def _llm_call_to_record(call: dict[str, Any], end_ms: int) -> dict[str, Any]:
    # The llm_call event is emitted after run_assistant_turn returns, so its
    # end_ms ≈ end-of-call. Subtract latency to recover the call start.
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
    started_at = start.get("wall_clock_start") or start.get("timestamp")
    root_iso = start.get("timestamp")  # legacy-fallback anchor only

    calls = _collect_llm_calls(root_path, anchor_in_root_ms=0, root_iso=root_iso)
    records = sorted(
        (_llm_call_to_record(c, end_ms) for c, end_ms in calls),
        key=lambda r: r["t_ms"],
    )

    header = {
        "root_episode_id": start.get("episode_id"),
        "model": start.get("model"),
        "started_at": started_at,
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
