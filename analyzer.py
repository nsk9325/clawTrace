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


# spawn_subagent is the trace-vocabulary tag for subagent dispatch. Subagents
# aren't tools, so the analyzer never counts them as such — even on legacy
# traces that wrote a tool_call event for the spawn (pre-refactor). Subagent
# metrics come exclusively from subagent_start / subagent_end events.
_SUBAGENT_TOOL_NAME = "spawn_subagent"


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
    total_subagent_ms: int
    prompt_tokens_total: int
    completion_tokens_total: int
    tool_call_count_by_name: dict[str, int]
    tool_ms_by_name: dict[str, int]
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


def _episode_wall_ms(start: dict[str, Any], end: dict[str, Any]) -> int:
    """Episode-relative ms from start→end. Prefers ``t_ms`` (single
    perf_counter clock; new traces). Falls back to ISO ``timestamp`` math
    for legacy traces written before the t_ms switch."""
    if "t_ms" in start and "t_ms" in end:
        return int(end["t_ms"]) - int(start["t_ms"])
    return _ms_between(start["timestamp"], end["timestamp"])


def _wall_ms_for_step(events_in_step: list[dict[str, Any]]) -> int:
    """Wall-clock time for a batch of timed events in one step.

    started_at_ms / ended_at_ms are origin-relative per step (fresh
    ToolExecutor timer per step), so max(ended) - min(started) gives the
    batch wall time: equals the sum of latencies under serial execution and
    equals the longest event's latency under fully parallel execution.
    """
    if not events_in_step:
        return 0
    starts = [int(e["started_at_ms"]) for e in events_in_step if "started_at_ms" in e]
    ends = [int(e["ended_at_ms"]) for e in events_in_step if "ended_at_ms" in e]
    if not starts or not ends:
        return 0
    return max(ends) - min(starts)


def _subagent_ms_for_step(events_in_step: list[dict[str, Any]]) -> int:
    """Wall-clock subagent time for one step, with a legacy-trace fallback.

    Pre-refactor subagent_end events carry only ``duration_ms`` (no origin-
    relative start/end). Under the default serial subagent_parallelism, the
    sum of durations equals wall — so summing duration_ms is exact. New
    traces use parallel-aware wall via ``_wall_ms_for_step``.
    """
    if not events_in_step:
        return 0
    if all("started_at_ms" in e and "ended_at_ms" in e for e in events_in_step):
        return _wall_ms_for_step(events_in_step)
    return sum(int(e.get("duration_ms", 0)) for e in events_in_step)


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
    subagent_by_step: dict[int, list[dict[str, Any]]] = {}
    llm_total_ms = 0
    prompt_tokens = 0
    completion_tokens = 0
    tool_counts: Counter = Counter()
    tool_ms: Counter = Counter()
    subagent_starts = 0

    for r in rows:
        t = r["type"]
        if t == "llm_call":
            llm_total_ms += int(r.get("latency_ms", 0))
            prompt_tokens += int(r.get("prompt_tokens", 0))
            completion_tokens += int(r.get("completion_tokens", 0))
        elif t == "tool_call":
            name = r.get("tool_name", "<unknown>")
            if name == _SUBAGENT_TOOL_NAME:
                continue  # legacy trace; subagent metrics come from subagent_* events
            tool_counts[name] += 1
            tool_ms[name] += int(r.get("latency_ms", 0))
            step_id = r.get("step_id")
            if step_id is not None:
                tool_by_step.setdefault(step_id, []).append(r)
        elif t == "subagent_start":
            subagent_starts += 1
        elif t == "subagent_end":
            step_id = r.get("step_id")
            if step_id is not None:
                subagent_by_step.setdefault(step_id, []).append(r)

    total_tool_ms = sum(_wall_ms_for_step(tools) for tools in tool_by_step.values())
    total_subagent_ms = sum(_subagent_ms_for_step(subs) for subs in subagent_by_step.values())
    total_wall_ms = _episode_wall_ms(start, end)

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
        total_subagent_ms=total_subagent_ms,
        prompt_tokens_total=prompt_tokens,
        completion_tokens_total=completion_tokens,
        tool_call_count_by_name=dict(tool_counts),
        tool_ms_by_name=dict(tool_ms),
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


@dataclass
class MetricsBreakdown:
    """The metric-portion of an episode summary, isolated for aggregation
    and reuse across the main / subagents / total sections."""
    total_llm_ms: int
    total_tool_ms: int
    prompt_tokens_total: int
    completion_tokens_total: int
    tool_call_count_by_name: dict[str, int]
    tool_ms_by_name: dict[str, int]


def _summary_to_metrics(s: EpisodeSummary) -> MetricsBreakdown:
    return MetricsBreakdown(
        total_llm_ms=s.total_llm_ms,
        total_tool_ms=s.total_tool_ms,
        prompt_tokens_total=s.prompt_tokens_total,
        completion_tokens_total=s.completion_tokens_total,
        tool_call_count_by_name=dict(s.tool_call_count_by_name),
        tool_ms_by_name=dict(s.tool_ms_by_name),
    )


def _merge_metrics(*breakdowns: MetricsBreakdown) -> MetricsBreakdown:
    out = MetricsBreakdown(
        total_llm_ms=0,
        total_tool_ms=0,
        prompt_tokens_total=0,
        completion_tokens_total=0,
        tool_call_count_by_name={},
        tool_ms_by_name={},
    )
    for b in breakdowns:
        out.total_llm_ms += b.total_llm_ms
        out.total_tool_ms += b.total_tool_ms
        out.prompt_tokens_total += b.prompt_tokens_total
        out.completion_tokens_total += b.completion_tokens_total
        for name, count in b.tool_call_count_by_name.items():
            out.tool_call_count_by_name[name] = out.tool_call_count_by_name.get(name, 0) + count
        for name, ms in b.tool_ms_by_name.items():
            out.tool_ms_by_name[name] = out.tool_ms_by_name.get(name, 0) + ms
    return out


def _render_metrics_lines(m: MetricsBreakdown, wall_for_pct: int | None) -> list[str]:
    """The shared LLM/Tool/per-tool/Tokens block. Used by all three sections."""
    def pct(part: int) -> str:
        return f" ({_pct(part, wall_for_pct)})" if wall_for_pct else ""

    lines = [
        f"  LLM time:     {m.total_llm_ms} ms{pct(m.total_llm_ms)}",
        f"  Tool time:    {m.total_tool_ms} ms{pct(m.total_tool_ms)}",
    ]
    if m.tool_call_count_by_name:
        # Sort by cumulative ms descending — bottleneck pops to the top.
        # Cumulative ms can exceed total_tool_ms when tools run in parallel,
        # since per-tool ms is the sum of latencies, not parallel-aware wall.
        names_by_ms = sorted(
            m.tool_call_count_by_name.keys(),
            key=lambda n: (-m.tool_ms_by_name.get(n, 0), n),
        )
        name_w = max(len(n) for n in names_by_ms)
        for name in names_by_ms:
            calls = m.tool_call_count_by_name[name]
            ms = m.tool_ms_by_name.get(name, 0)
            lines.append(f"    {name:<{name_w}}  {calls:>2}×  {ms:>7} ms")
    lines.append(f"  Tokens:       in={m.prompt_tokens_total} out={m.completion_tokens_total}")
    return lines


def render_summary(summary: EpisodeSummary) -> str:
    lines = [
        f"=== {summary.episode_id} (main agent) ===",
        f"  Status:       {summary.status} ({summary.stop_reason or '-'})",
        f"  Model:        {summary.model} ({summary.backend})",
        f"  Prompt:       {summary.system_prompt_name}",
    ]
    if summary.workload_instance_id:
        lines.append(f"  Instance:     {summary.workload_instance_id}")
    lines.append(f"  Steps:        {summary.step_count}")
    lines.append(f"  Wall time:    {summary.total_wall_ms} ms")
    lines.extend(_render_metrics_lines(_summary_to_metrics(summary), summary.total_wall_ms))
    if summary.subagent_count:
        lines.append(
            f"  Subagents:    {summary.subagent_count} spawned, "
            f"{summary.total_subagent_ms} ms ({_pct(summary.total_subagent_ms, summary.total_wall_ms)})"
        )
    return "\n".join(lines)


def render_subagent_section(
    aggregate: MetricsBreakdown,
    *,
    episode_count: int,
    step_count: int,
    total_subagent_ms: int,
    parent_wall_ms: int,
) -> str:
    plural = "episode" if episode_count == 1 else "episodes"
    lines = [
        f"=== Subagents ({episode_count} {plural}, aggregated) ===",
        f"  Steps:        {step_count}",
        f"  Wall time:    {total_subagent_ms} ms ({_pct(total_subagent_ms, parent_wall_ms)})",
    ]
    lines.extend(_render_metrics_lines(aggregate, parent_wall_ms))
    return "\n".join(lines)


def render_total_section(total: MetricsBreakdown, parent_wall_ms: int) -> str:
    lines = [
        "=== Total (main + subagents) ===",
        f"  Wall time:    {parent_wall_ms} ms (main episode wall, subagents ran within it)",
    ]
    lines.extend(_render_metrics_lines(total, parent_wall_ms))
    return "\n".join(lines)


@dataclass
class TreeNode:
    summary: EpisodeSummary
    spawned_at_step: int | None
    children: list["TreeNode"] = field(default_factory=list)
    missing: list[dict[str, Any]] = field(default_factory=list)


def aggregate_subagents(root: TreeNode) -> tuple[MetricsBreakdown, int, int]:
    """Sum metrics across all subagent descendants of `root` (excluding root).
    Returns (merged_metrics, total_step_count, episode_count)."""
    descendants: list[TreeNode] = []

    def visit(node: TreeNode) -> None:
        for child in node.children:
            descendants.append(child)
            visit(child)

    visit(root)
    metrics = _merge_metrics(*(_summary_to_metrics(d.summary) for d in descendants))
    step_count = sum(d.summary.step_count for d in descendants)
    return metrics, step_count, len(descendants)


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

    parts = [render_summary(summary)]

    tree: TreeNode | None = None
    if summary.subagent_count > 0:
        try:
            tree = build_tree(args.trace)
            sub_metrics, sub_steps, sub_episodes = aggregate_subagents(tree)
            parts.append(render_subagent_section(
                sub_metrics,
                episode_count=sub_episodes,
                step_count=sub_steps,
                total_subagent_ms=summary.total_subagent_ms,
                parent_wall_ms=summary.total_wall_ms,
            ))
            total = _merge_metrics(_summary_to_metrics(summary), sub_metrics)
            parts.append(render_total_section(total, summary.total_wall_ms))
        except Exception as exc:
            print(f"(Subagent aggregation failed: {exc})", file=sys.stderr)

    parts.append(render_workload(summary.workload_info))
    parts.append(render_config(summary.cfg))
    if tree is not None:
        parts.append(render_tree(tree))

    rendered = "\n\n".join(parts)
    print(rendered)

    analysis_path = args.trace.parent / f"{args.trace.stem}.analysis.txt"
    analysis_path.write_text(rendered + "\n", encoding="utf-8")
    print(f"\nSaved analysis to {analysis_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
