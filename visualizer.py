"""Render a clawTrace episode as a server/client Gantt chart.

Two-layer design:
- ``build_timeline()`` walks the root trace + every recursively reachable
  subagent trace and emits a ``Timeline`` of ``Bar``s on ``Lane``s, anchored
  to the root episode's start. Pure stdlib.
- ``render_png()`` consumes the Timeline. The ``.gantt.json`` sidecar mirrors
  the Timeline shape, so a future render_html can read it without touching
  Layer 1.

Lane convention: each episode contributes ``<ep>:server`` and ``<ep>:client``.
LLM activity lands on server; tool activity on client. A spawn shows as a
gray hatched ``subagent_span`` bar on the parent's client lane, with the
child's own server+client lanes appearing below, time-aligned on the same
root timeline. Overlapping client-lane bars (parallel tools / parallel
subagents) split into ``<ep>:client:N`` sub-lanes.

CLI: python visualizer.py traces/episode_<id>/episode_<id>.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# --------------------------------------------------------------- data model

@dataclass
class Bar:
    lane: str
    kind: str            # ttft | decode | llm | tool | subagent_span
    start_ms: int        # all times are relative to root episode_start
    end_ms: int
    label: str
    detail: str
    step_id: int | None
    status: str          # ok | error | crashed | completed | incomplete


@dataclass
class Lane:
    key: str
    title: str
    episode_id: str
    depth: int


@dataclass
class Timeline:
    lanes: list[Lane]
    bars: list[Bar]
    root_episode_id: str
    root_started_at: str
    total_wall_ms: int
    step_boundaries_ms: list[int]


# --------------------------------------------------------------- helpers

def _load_trace(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _ms_since(t0: datetime, iso: str) -> int:
    return int((datetime.fromisoformat(iso) - t0).total_seconds() * 1000)


def _short(ep_id: str) -> str:
    return ep_id.replace("episode_", "")


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"


def _tool_detail(tool_name: str, params: dict[str, Any]) -> str:
    """One-line summary of a tool call's params, for chart labels."""
    if tool_name == "bash":
        cmd = str(params.get("command", "")).strip()
        return cmd.splitlines()[0] if cmd else ""
    if tool_name in ("read_file", "write_file", "edit_file"):
        return str(params.get("file_path", ""))
    if tool_name == "glob":
        pattern = str(params.get("pattern", ""))
        path = params.get("path")
        return f"{pattern} in {path}" if path else pattern
    if tool_name == "grep":
        return str(params.get("pattern", ""))
    return ""


# --------------------------------------------------------------- Layer 1: build

def _bars_from_llm_call(r: dict[str, Any], t0: datetime, server_lane: str) -> list[Bar]:
    end_ms = _ms_since(t0, r["timestamp"])
    start_ms = end_ms - int(r.get("latency_ms", 0))
    label = str(r.get("model", ""))
    step = r.get("step_id")
    ttft = r.get("ttft_ms")
    decode = r.get("decode_time_ms")
    if ttft is not None and decode is not None and (int(ttft) + int(decode)) > 0:
        ttft_end = start_ms + int(ttft)
        return [
            Bar(server_lane, "ttft",   start_ms, ttft_end, label, "", step, "ok"),
            Bar(server_lane, "decode", ttft_end, end_ms,   label, "", step, "ok"),
        ]
    return [Bar(server_lane, "llm", start_ms, end_ms, label, "", step, "ok")]


def _bar_from_tool_call(r: dict[str, Any], t0: datetime, client_lane: str) -> Bar | None:
    # Legacy traces (pre-refactor) emitted a tool_call event for spawn_subagent
    # alongside subagent_start/end; the spawn is already represented by the
    # subagent_span bar, so skip the legacy duplicate.
    if r.get("tool_name") == "spawn_subagent":
        return None
    end_ms = _ms_since(t0, r["timestamp"])
    name = str(r.get("tool_name", ""))
    return Bar(
        lane=client_lane, kind="tool",
        start_ms=end_ms - int(r.get("latency_ms", 0)), end_ms=end_ms,
        label=name, detail=_tool_detail(name, r.get("params", {})),
        step_id=r.get("step_id"),
        status=str(r.get("exit_status", "ok")),
    )


def _subagent_span_bar(r: dict[str, Any], t0: datetime, client_lane: str) -> Bar:
    end_ms = _ms_since(t0, r["timestamp"])
    child_ep = r.get("child_episode_id", "<unknown>")
    return Bar(
        lane=client_lane, kind="subagent_span",
        start_ms=end_ms - int(r.get("duration_ms", 0)), end_ms=end_ms,
        label=f"spawn {_short(child_ep)}",
        detail=str(r.get("child_status", "")),
        step_id=r.get("step_id"),
        status=str(r.get("child_status", "ok")),
    )


def _resolve_child_path(child_path_str: str | None, parent_dir: Path) -> Path | None:
    if not child_path_str:
        return None
    p = Path(child_path_str)
    if not p.is_absolute():
        # Per-root-episode dir: child trace is a sibling of parent.
        p = parent_dir / p.name
    return p if p.exists() else None


def _walk_trace(
    trace_path: Path,
    *,
    t0: datetime,
    depth: int,
    lanes: list[Lane],
    bars: list[Bar],
    step_boundaries: list[int] | None,  # populated only at depth=0
) -> None:
    rows = _load_trace(trace_path)
    if not rows or rows[0].get("type") != "episode_start":
        return

    ep_id = rows[0]["episode_id"]
    server_lane = f"{ep_id}:server"
    client_lane = f"{ep_id}:client"
    indent = "  " * depth
    arrow = "↳ " if depth > 0 else ""
    name = "main" if depth == 0 else f"sub {_short(ep_id)}"
    lanes.append(Lane(server_lane, f"{indent}{arrow}{name} — server", ep_id, depth))
    lanes.append(Lane(client_lane, f"{indent}{arrow}{name} — client", ep_id, depth))

    parent_dir = trace_path.parent

    for r in rows:
        kind = r["type"]
        if kind == "step_start" and step_boundaries is not None:
            step_boundaries.append(_ms_since(t0, r["timestamp"]))
        elif kind == "llm_call":
            bars.extend(_bars_from_llm_call(r, t0, server_lane))
        elif kind == "tool_call":
            bar = _bar_from_tool_call(r, t0, client_lane)
            if bar is not None:
                bars.append(bar)
        elif kind == "subagent_end":
            bars.append(_subagent_span_bar(r, t0, client_lane))
            child = _resolve_child_path(r.get("child_trace_path"), parent_dir)
            if child is not None:
                _walk_trace(child, t0=t0, depth=depth + 1, lanes=lanes, bars=bars,
                            step_boundaries=None)


def _split_client_lanes(lanes: list[Lane], bars: list[Bar]) -> list[Lane]:
    """When bars on the same client lane overlap (parallel tools or parallel
    subagents), split into ``<ep>:client:N`` sub-lanes via greedy lane-packing.
    Mutates ``bar.lane`` in place; returns the rebuilt lane list."""
    new_lanes: list[Lane] = []

    for lane in lanes:
        if not lane.key.endswith(":client"):
            new_lanes.append(lane)
            continue

        bar_idxs = sorted(
            (i for i, b in enumerate(bars) if b.lane == lane.key),
            key=lambda i: (bars[i].start_ms, bars[i].end_ms),
        )
        sublane_end: list[int] = []
        assigned: dict[int, int] = {}
        for i in bar_idxs:
            b = bars[i]
            for s, end in enumerate(sublane_end):
                if end <= b.start_ms:
                    sublane_end[s] = b.end_ms
                    assigned[i] = s
                    break
            else:
                sublane_end.append(b.end_ms)
                assigned[i] = len(sublane_end) - 1

        n_sub = max(len(sublane_end), 1)
        for s in range(n_sub):
            suffix = "" if n_sub == 1 else f".{s}"
            new_lanes.append(Lane(
                key=f"{lane.key}:{s}",
                title=f"{lane.title}{suffix}",
                episode_id=lane.episode_id,
                depth=lane.depth,
            ))
        for i, s in assigned.items():
            bars[i].lane = f"{lane.key}:{s}"

    return new_lanes


def build_timeline(root_trace_path: Path) -> Timeline:
    root_trace_path = Path(root_trace_path)
    rows = _load_trace(root_trace_path)
    if not rows or rows[0].get("type") != "episode_start":
        raise ValueError(f"{root_trace_path}: missing episode_start as first event")
    end = next((r for r in reversed(rows) if r["type"] == "episode_end"), None)
    if end is None:
        raise ValueError(f"{root_trace_path}: missing episode_end (run may have been killed)")

    root_iso = rows[0]["timestamp"]
    t0 = datetime.fromisoformat(root_iso)

    lanes: list[Lane] = []
    bars: list[Bar] = []
    step_boundaries: list[int] = []
    _walk_trace(root_trace_path, t0=t0, depth=0, lanes=lanes, bars=bars,
                step_boundaries=step_boundaries)
    lanes = _split_client_lanes(lanes, bars)

    return Timeline(
        lanes=lanes,
        bars=bars,
        root_episode_id=rows[0]["episode_id"],
        root_started_at=root_iso,
        total_wall_ms=_ms_since(t0, end["timestamp"]),
        step_boundaries_ms=step_boundaries,
    )


def dump_json(timeline: Timeline, out_path: Path) -> Path:
    out_path.write_text(json.dumps(asdict(timeline), indent=2), encoding="utf-8")
    return out_path


# --------------------------------------------------------------- Layer 2: render

# Bar.kind → fill color and (optional) hatch pattern. Renderer-private —
# Layer 1 stays purely structural.
_KIND_COLOR = {
    "ttft":          "#A6CEE3",
    "decode":        "#1F78B4",
    "llm":           "#1F78B4",
    "tool":          "#FF7F0E",
    "subagent_span": "#999999",
}
_KIND_HATCH = {"subagent_span": "//"}
_ERROR_COLOR = "#D62728"

_FIG_WIDTH_IN = 14
_BAR_HEIGHT = 0.4
_FONTSIZE_INSIDE = 7
_FONTSIZE_ABOVE = 6


def _parse_detail_tools(arg: str) -> frozenset[str] | None:
    """CLI arg → set passed to render_png. ``None`` shows every detail; an
    empty frozenset shows none."""
    s = arg.strip().lower()
    if s == "all":
        return None
    if s == "none" or not s:
        return frozenset()
    return frozenset(t.strip() for t in s.split(",") if t.strip())


def _label_text(bar: Bar, budget: int, show_detail: bool) -> str:
    if not show_detail:
        return bar.label
    return f"{bar.label}\n{_truncate(bar.detail, budget)}"


def render_png(
    timeline: Timeline,
    out_path: Path,
    *,
    dpi: int = 130,
    detail_tools: frozenset[str] | None = frozenset({"bash"}),
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    n_lanes = len(timeline.lanes)
    if n_lanes == 0:
        raise ValueError("Timeline has no lanes — nothing to render")

    fig_height = max(3.0, 0.9 * n_lanes + 1.6)
    fig, ax = plt.subplots(figsize=(_FIG_WIDTH_IN, fig_height))

    # Lane y placement: list-order [0..n-1] reads top→bottom.
    lane_y = {lane.key: (n_lanes - 1) - i for i, lane in enumerate(timeline.lanes)}

    use_seconds = timeline.total_wall_ms > 5000
    scale = 1000.0 if use_seconds else 1.0
    unit_label = "seconds" if use_seconds else "ms"

    # Char budgets driven by figure dimensions.
    axis_chars = int(_FIG_WIDTH_IN * dpi * 0.85 / (_FONTSIZE_INSIDE * 0.7))
    gap_in = (1 - _BAR_HEIGHT) * (fig_height - 1.6) / max(n_lanes, 1)
    v_budget = max(4, int(gap_in * dpi / _FONTSIZE_ABOVE))

    def h_budget_for(width_ms: int) -> int:
        if timeline.total_wall_ms <= 0:
            return 0
        return max(0, int(width_ms / timeline.total_wall_ms * axis_chars) - 1)

    # Sub-resolution tools (engine.py rounds latency_ms to 0 — see review
    # issue #2) would render at width 0 and disappear. Visual clamp only;
    # the JSON sidecar still carries the real (zero) timing.
    min_visible_ms = max(int(timeline.total_wall_ms * 0.002), 1)

    for bar in timeline.bars:
        y = lane_y.get(bar.lane)
        if y is None:
            continue
        real_w = max(0, bar.end_ms - bar.start_ms)
        visible_w = (
            min_visible_ms if bar.kind == "tool" and real_w < min_visible_ms else real_w
        )
        is_error = bar.status not in ("ok", "completed")

        ax.barh(
            y=y,
            width=visible_w / scale,
            left=bar.start_ms / scale,
            height=_BAR_HEIGHT,
            color=_ERROR_COLOR if is_error else _KIND_COLOR.get(bar.kind, "#888888"),
            hatch=_KIND_HATCH.get(bar.kind, ""),
            edgecolor="none", linewidth=0,
        )

        if bar.kind != "tool" or not bar.label:
            continue

        h_budget = h_budget_for(real_w)
        fits_inside = h_budget >= len(bar.label)
        show_detail = bool(bar.detail) and (
            detail_tools is None or bar.label in detail_tools
        )
        text = _label_text(bar, h_budget if fits_inside else v_budget, show_detail)
        bar_center_x = (bar.start_ms + real_w / 2) / scale

        if fits_inside:
            ax.text(
                bar_center_x, y, text,
                ha="center", va="center",
                fontsize=_FONTSIZE_INSIDE, color="black",
            )
        else:
            # Multi-line + rotation=90: matplotlib lays out the unrotated
            # lines stacked, then rotates the block — so label and detail
            # become two parallel vertical columns.
            ax.text(
                bar_center_x, y + _BAR_HEIGHT / 2 + 0.05, text,
                ha="center", va="bottom", rotation=90,
                fontsize=_FONTSIZE_ABOVE, color="black",
            )

    # Step boundaries from the root episode — coherent global anchor since
    # all bars share the root timeline.
    for boundary_ms in timeline.step_boundaries_ms:
        ax.axvline(boundary_ms / scale, color="#dddddd", linewidth=0.6, zorder=0)

    ax.set_yticks(list(range(n_lanes)))
    ax.set_yticklabels(list(reversed([l.title for l in timeline.lanes])), fontsize=9)
    ax.set_ylim(-0.6, n_lanes - 0.4)
    ax.set_xlabel(f"Time ({unit_label})", fontsize=9)
    ax.set_xlim(0, timeline.total_wall_ms / scale * 1.005)

    n_episodes = len({lane.episode_id for lane in timeline.lanes})
    ax.set_title(
        f"{timeline.root_episode_id} — {timeline.total_wall_ms / 1000:.1f}s wall, "
        f"{n_episodes} episode(s)",
        fontsize=11,
    )

    ax.legend(
        handles=[
            mpatches.Patch(facecolor=_KIND_COLOR["ttft"],   label="LLM: prefill / TTFT"),
            mpatches.Patch(facecolor=_KIND_COLOR["decode"], label="LLM: decode"),
            mpatches.Patch(facecolor=_KIND_COLOR["tool"],   label="Tool"),
            mpatches.Patch(facecolor=_ERROR_COLOR,          label="Error"),
            mpatches.Patch(facecolor=_KIND_COLOR["subagent_span"], hatch="//",
                           label="Subagent span (parent's view)"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),  # just outside the right edge
        fontsize=8, framealpha=0.9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trace", type=Path, help="Path to a root episode .jsonl trace")
    p.add_argument(
        "--detail-tools",
        default="bash",
        help="Comma-separated tool names to show details for in the PNG "
             "(e.g. 'bash,grep'). 'all' shows every tool's detail; 'none' "
             "shows no details. JSON sidecar always carries full detail. "
             "Default: bash.",
    )
    args = p.parse_args(argv)

    if not args.trace.exists():
        print(f"Error: trace file not found: {args.trace}", file=sys.stderr)
        return 1

    try:
        timeline = build_timeline(args.trace)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    stem = args.trace.stem
    out_dir = args.trace.parent
    json_path = dump_json(timeline, out_dir / f"{stem}.gantt.json")
    png_path = render_png(
        timeline,
        out_dir / f"{stem}.gantt.png",
        detail_tools=_parse_detail_tools(args.detail_tools),
    )

    print(
        f"Built timeline: {len(timeline.lanes)} lanes, {len(timeline.bars)} bars, "
        f"{timeline.total_wall_ms} ms wall, "
        f"{len(timeline.step_boundaries_ms)} root steps"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
