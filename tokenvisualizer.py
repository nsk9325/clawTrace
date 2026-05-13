"""Plot a token trace: per-call token volumes + prefix-cache hit rate.

Input is a ``.tokens.jsonl`` file produced by ``token_trace_gen.py`` (one
header line followed by one record per LLM call, flattened across the
parent episode and its subagent descendants on a single root timeline).

Chart:
  X axis:      call start time (ms or s, relative to root episode start)
  Left Y:      token counts — three line series, one marker per call:
                 - input tokens (incl. cached prefix)
                 - output tokens
                 - cached tokens
  Right Y:     cache hit rate as a percent of input tokens — drawn as a
               triangle at (t_ms, 100 * cached / in_tokens) for each call
               that reports ``cached_tokens``.

CLI:
  python tokenvisualizer.py traces/episode_<id>/episode_<id>.tokens.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TokenCall:
    t_ms: int
    in_tokens: int
    out_tokens: int
    cached_tokens: int | None  # None when the provider didn't expose it
    cache_pct: float | None    # 100 * cached / in_tokens; None if unmeasured


def load_token_trace(path: Path) -> tuple[dict[str, Any], list[TokenCall]]:
    with path.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if not lines:
        raise ValueError(f"{path}: empty token trace")
    header = lines[0]
    calls: list[TokenCall] = []
    for r in lines[1:]:
        in_t = int(r.get("in_tokens", 0))
        out_t = int(r.get("out_tokens", 0))
        cached = r.get("cached_tokens")
        cached_int: int | None = int(cached) if cached is not None else None
        cache_pct: float | None
        if cached_int is not None and in_t > 0:
            cache_pct = 100.0 * cached_int / in_t
        else:
            cache_pct = None
        calls.append(TokenCall(
            t_ms=int(r["t_ms"]),
            in_tokens=in_t,
            out_tokens=out_t,
            cached_tokens=cached_int,
            cache_pct=cache_pct,
        ))
    return header, calls


_COLOR_IN = "#1F78B4"        # blue — input
_COLOR_OUT = "#FF7F0E"       # orange — output
_COLOR_CACHED = "#33A02C"    # green — cached
_COLOR_CACHE_PCT = "#DAA520" # goldenrod — cache hit % marker


def render_png(
    header: dict[str, Any],
    calls: list[TokenCall],
    out_path: Path,
    *,
    dpi: int = 130,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not calls:
        raise ValueError("Token trace has no LLM calls — nothing to plot")

    t_max_ms = max((c.t_ms for c in calls), default=0)
    use_seconds = t_max_ms > 5000
    scale = 1000.0 if use_seconds else 1.0
    unit = "seconds" if use_seconds else "ms"

    span = max(t_max_ms / scale, 1.0)

    xs = [c.t_ms / scale for c in calls]
    in_vals = [c.in_tokens for c in calls]
    out_vals = [c.out_tokens for c in calls]
    cached_vals = [c.cached_tokens or 0 for c in calls]

    fig, ax_left = plt.subplots(figsize=(12, 5))
    ax_right = ax_left.twinx()

    ax_left.plot(xs, in_vals,     marker="o", color=_COLOR_IN,     label="Input tokens",  linewidth=1.5, markersize=4)
    ax_left.plot(xs, out_vals,    marker="o", color=_COLOR_OUT,    label="Output tokens", linewidth=1.5, markersize=4)
    ax_left.plot(xs, cached_vals, marker="o", color=_COLOR_CACHED, label="Cached tokens", linewidth=1.5, markersize=4)

    ax_left.set_xlabel(f"Call start ({unit} from root episode start)", fontsize=9)
    ax_left.set_ylabel("Tokens", fontsize=9)
    ax_left.set_xlim(-span * 0.02, span * 1.02)
    ax_left.set_ylim(bottom=0)
    ax_left.grid(True, axis="y", alpha=0.25)
    ax_left.spines["top"].set_visible(False)

    # Right axis: cache hit % per call (only where measured).
    pct_xs = [c.t_ms / scale for c in calls if c.cache_pct is not None]
    pct_ys = [c.cache_pct for c in calls if c.cache_pct is not None]
    if pct_xs:
        ax_right.scatter(
            pct_xs, pct_ys,
            marker="^", s=80, color=_COLOR_CACHE_PCT,
            edgecolor="black", linewidth=0.6,
            zorder=5, label="Cache hit %",
        )
    ax_right.set_ylabel("Cache hit % (cached / input)", fontsize=9)
    ax_right.set_ylim(0, 105)
    ax_right.spines["top"].set_visible(False)

    # Single combined legend in the corner.
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.9)

    model = header.get("model", "?")
    n_calls = header.get("n_calls", len(calls))
    root_id = header.get("root_episode_id", "?")
    ax_left.set_title(f"{root_id} — {model} — {n_calls} LLM call(s)", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("tokens_trace", type=Path,
                   help="Path to a .tokens.jsonl file (from token_trace_gen.py)")
    args = p.parse_args(argv)

    if not args.tokens_trace.exists():
        print(f"Error: token trace not found: {args.tokens_trace}", file=sys.stderr)
        return 1

    try:
        header, calls = load_token_trace(args.tokens_trace)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # <root>.tokens.jsonl → <root>.tokens.png next to it.
    out_path = args.tokens_trace.with_suffix(".png")
    try:
        render_png(header, calls, out_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
