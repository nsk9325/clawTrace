from __future__ import annotations

import sys
import time
from typing import Any

from config import load_config
from engine import run_episode
import subagent  # noqa: F401  (registers spawn_subagent)


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


def main() -> None:
    task_input = " ".join(sys.argv[1:]).strip()
    config = {
        "allow_parallel_tools": True,
        "max_parallel_tools": 4,
        "enable_subagents": True,
    }

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
