from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from config import load_config
from engine import run_episode


def _run_all(task_input: str, config: dict[str, Any]) -> list[tuple[dict[str, Any], int]]:
    merged = dict(load_config())
    merged.update(config)
    runs = int(merged.get("runs_per_task", 1))

    out: list[tuple[dict[str, Any], int]] = []
    for _ in range(runs):
        started = time.perf_counter()
        result = run_episode(task_input=task_input, config=config)
        wall_ms = int((time.perf_counter() - started) * 1000)
        out.append((result, wall_ms))
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one ClawTrace episode against the agent loop.")
    p.add_argument("task", nargs="+", help="The task input to send to the agent.")
    p.add_argument(
        "--config",
        default=None,
        help='Arbitrary cfg overrides as JSON, e.g. \'{"backend":"custom","custom_base_url":"http://localhost:8000/v1"}\'',
    )
    return p.parse_args(argv)


def main() -> None:
    args = _parse_args()
    task_input = " ".join(args.task).strip()
    # Only override defaults that the CLI is opinionated about.
    # `enable_subagents` and `max_parallel_tools` were redundant after
    # DEFAULT_CONFIG was tuned; `allow_parallel_tools=True` is kept here as
    # a runner-specific opinion (DEFAULT_CONFIG leaves it False).
    config: dict[str, Any] = {
        "allow_parallel_tools": True,
    }
    if args.config is not None:
        config.update(json.loads(args.config))

    total_start = time.perf_counter()
    try:
        results = _run_all(task_input, config)
    except Exception as exc:
        print(f"Error: {exc}")
        return
    total_wall_ms = int((time.perf_counter() - total_start) * 1000)

    for i, (result, wall_ms) in enumerate(results, start=1):
        print(
            f"Run {i}/{len(results)}: {result['episode_id']} "
            f"status={result['status']} "
            f"steps={result['step_count']} "
            f"wall={wall_ms}ms "
            f"trace={result['trace_path']}"
        )
    print(f"Completed {len(results)} run(s) in {total_wall_ms}ms")


if __name__ == "__main__":
    main()
