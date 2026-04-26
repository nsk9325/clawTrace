"""Analyze a ClawTrace episode trace.

Reads a JSONL trace, computes derived per-episode metrics, prints a
human-readable summary. When the episode spawned subagents, also walks
the subagent tree (loading co-located child trace files) and prints it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class EpisodeSummary:
    episode_id: str
    depth: int
    status: str
    stop_reason: str | None
    step_count: int
    total_wall_ms: int
    total_llm_ms: int
    total_tool_ms: int
    prompt_tokens_total: int
    completion_tokens_total: int
    tool_call_count_by_name: dict[str, int]
    subagent_count: int
    model: str
    backend: str
    system_prompt_name: str
    workload_instance_id: str | None
    cfg: dict[str, Any]
    workload_info: dict[str, Any] | None


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _ms_between(start_iso: str, end_iso: str) -> int:
    return int((datetime.fromisoformat(end_iso) - datetime.fromisoformat(start_iso)).total_seconds() * 1000)


def _tool_wall_ms_for_step(tool_events: list[dict[str, Any]]) -> int:
    """Wall-clock tool time for one step, accounting for parallel execution.

    started_at_ms / ended_at_ms are origin-relative per step (fresh ToolExecutor
    timer per step), so max(ended) - min(started) gives the batch wall time:
    equals the sum of latencies under serial execution and equals the longest
    tool's latency under fully parallel execution.
    """
    if not tool_events:
        return 0
    starts = [int(t["started_at_ms"]) for t in tool_events]
    ends = [int(t["ended_at_ms"]) for t in tool_events]
    return max(ends) - min(starts)


def summarize(rows: list[dict[str, Any]]) -> EpisodeSummary:
    if not rows:
        raise ValueError("Empty trace.")
    if rows[0]["type"] != "episode_start":
        raise ValueError(f"Expected episode_start as first event, got {rows[0]['type']!r}")
    start = rows[0]
    end = next((r for r in reversed(rows) if r["type"] == "episode_end"), None)
    if end is None:
        raise ValueError("Trace has no episode_end event (run may have been killed).")

    tool_by_step: dict[int, list[dict[str, Any]]] = {}
    llm_total_ms = 0
    prompt_tokens = 0
    completion_tokens = 0
    tool_counts: Counter = Counter()
    subagent_starts = 0

    for r in rows:
        t = r["type"]
        if t == "llm_call":
            llm_total_ms += int(r.get("latency_ms", 0))
            prompt_tokens += int(r.get("prompt_tokens", 0))
            completion_tokens += int(r.get("completion_tokens", 0))
        elif t == "tool_call":
            tool_counts[r.get("tool_name", "<unknown>")] += 1
            step_id = r.get("step_id")
            if step_id is not None:
                tool_by_step.setdefault(step_id, []).append(r)
        elif t == "subagent_start":
            subagent_starts += 1

    total_tool_ms = sum(_tool_wall_ms_for_step(tools) for tools in tool_by_step.values())
    total_wall_ms = _ms_between(start["timestamp"], end["timestamp"])

    raw_workload = start.get("workload_info")
    workload_info = raw_workload if isinstance(raw_workload, dict) else None
    workload_instance_id = workload_info.get("instance_id") if workload_info else None
    cfg = start.get("cfg") if isinstance(start.get("cfg"), dict) else {}

    return EpisodeSummary(
        episode_id=start["episode_id"],
        depth=int(start.get("depth", 0)),
        status=str(end.get("status", "unknown")),
        stop_reason=end.get("stop_reason"),
        step_count=int(end.get("step_count", 0)),
        total_wall_ms=total_wall_ms,
        total_llm_ms=llm_total_ms,
        total_tool_ms=total_tool_ms,
        prompt_tokens_total=prompt_tokens,
        completion_tokens_total=completion_tokens,
        tool_call_count_by_name=dict(tool_counts),
        subagent_count=subagent_starts,
        model=str(start.get("model", "?")),
        backend=str(start.get("backend", "?")),
        system_prompt_name=str(start.get("system_prompt_name", "?")),
        workload_instance_id=workload_instance_id,
        cfg=cfg,
        workload_info=workload_info,
    )


def _pct(part: int, whole: int) -> str:
    if whole <= 0:
        return "-"
    return f"{100.0 * part / whole:.1f}%"


def render_workload(workload: dict[str, Any] | None) -> str:
    if not workload:
        return "Workload: (none)"
    lines = ["Workload:"]
    for k in sorted(workload.keys()):
        lines.append(f"  {k}: {workload[k]!r}")
    return "\n".join(lines)


def render_config(cfg: dict[str, Any]) -> str:
    if not cfg:
        return "Config: (not recorded)"
    lines = ["Config:"]
    for k in sorted(cfg.keys()):
        lines.append(f"  {k}: {cfg[k]!r}")
    return "\n".join(lines)


def render_summary(summary: EpisodeSummary) -> str:
    lines = [
        f"=== {summary.episode_id} ===",
        f"  Status:       {summary.status} ({summary.stop_reason or '-'})",
        f"  Model:        {summary.model} ({summary.backend})",
        f"  Prompt:       {summary.system_prompt_name}",
    ]
    if summary.workload_instance_id:
        lines.append(f"  Instance:     {summary.workload_instance_id}")
    lines.extend([
        f"  Steps:        {summary.step_count}",
        f"  Wall time:    {summary.total_wall_ms} ms",
        f"  LLM time:     {summary.total_llm_ms} ms ({_pct(summary.total_llm_ms, summary.total_wall_ms)})",
        f"  Tool time:    {summary.total_tool_ms} ms ({_pct(summary.total_tool_ms, summary.total_wall_ms)})",
        f"  Tokens:       in={summary.prompt_tokens_total} out={summary.completion_tokens_total}",
    ])
    if summary.tool_call_count_by_name:
        tools_str = ", ".join(f"{n}={c}" for n, c in sorted(summary.tool_call_count_by_name.items()))
        lines.append(f"  Tool calls:   {tools_str}")
    if summary.subagent_count:
        lines.append(f"  Subagents:    {summary.subagent_count} spawned")
    return "\n".join(lines)


@dataclass
class TreeNode:
    summary: EpisodeSummary
    spawned_at_step: int | None
    children: list["TreeNode"] = field(default_factory=list)
    missing: list[dict[str, Any]] = field(default_factory=list)


def build_tree(trace_path: Path, spawned_at_step: int | None = None) -> TreeNode:
    rows = load_trace(trace_path)
    summary = summarize(rows)
    node = TreeNode(summary=summary, spawned_at_step=spawned_at_step)

    parent_dir = Path(trace_path).parent
    for r in rows:
        if r.get("type") != "subagent_end":
            continue
        child_path_str = r.get("child_trace_path")
        child_step = r.get("step_id")
        if not child_path_str:
            continue
        child_path = Path(child_path_str)
        if not child_path.is_absolute():
            # Parent and child are siblings in both the new per-root-episode
            # layout and the old flat layout — resolve by basename in parent_dir.
            child_path = parent_dir / child_path.name
        if not child_path.exists():
            node.missing.append({
                "child_episode_id": r.get("child_episode_id"),
                "spawned_at_step": child_step,
                "error": "trace file not found",
            })
            continue
        try:
            node.children.append(build_tree(child_path, spawned_at_step=child_step))
        except Exception as exc:
            node.missing.append({
                "child_episode_id": r.get("child_episode_id"),
                "spawned_at_step": child_step,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return node


def render_tree(node: TreeNode) -> str:
    lines = ["Subagent tree:"]
    _render_tree_lines(node, prefix="", is_last=True, is_root=True, out=lines)
    return "\n".join(lines)


def _render_tree_lines(
    node: TreeNode,
    prefix: str,
    is_last: bool,
    is_root: bool,
    out: list[str],
) -> None:
    if is_root:
        connector = ""
        new_prefix = ""
    else:
        connector = "└─ " if is_last else "├─ "
        new_prefix = prefix + ("   " if is_last else "│  ")

    s = node.summary
    spawn_info = f", spawned at step {node.spawned_at_step}" if node.spawned_at_step is not None else ""
    out.append(
        f"{prefix}{connector}{s.episode_id} (depth {s.depth}{spawn_info}, "
        f"{s.status}, {s.step_count} steps)"
    )

    items: list[tuple[str, Any]] = [("real", c) for c in node.children]
    items.extend(("missing", m) for m in node.missing)

    for i, (kind, item) in enumerate(items):
        last = i == len(items) - 1
        if kind == "real":
            _render_tree_lines(item, new_prefix, last, False, out)
        else:
            con = "└─ " if last else "├─ "
            spawn = (
                f"spawned at step {item['spawned_at_step']}, "
                if item.get("spawned_at_step") is not None
                else ""
            )
            out.append(
                f"{new_prefix}{con}{item.get('child_episode_id') or '<unknown>'} "
                f"({spawn}MISSING: {item['error']})"
            )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trace", type=Path, help="Path to a single .jsonl trace file")
    args = p.parse_args(argv)

    if not args.trace.exists():
        print(f"Error: trace file not found: {args.trace}", file=sys.stderr)
        return 1

    try:
        rows = load_trace(args.trace)
        summary = summarize(rows)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parts = [
        render_summary(summary),
        render_workload(summary.workload_info),
        render_config(summary.cfg),
    ]
    if summary.subagent_count > 0:
        try:
            tree = build_tree(args.trace)
            parts.append(render_tree(tree))
        except Exception as exc:
            print(f"(Tree rendering failed: {exc})", file=sys.stderr)

    rendered = "\n\n".join(parts)
    print(rendered)

    analysis_path = args.trace.parent / f"{args.trace.stem}.analysis.txt"
    analysis_path.write_text(rendered + "\n", encoding="utf-8")
    print(f"\nSaved analysis to {analysis_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
