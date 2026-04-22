from __future__ import annotations

import sys

from engine import run_episode


def main() -> None:
    task_input = " ".join(sys.argv[1:]).strip()
    try:
        result = run_episode(task_input=task_input)
    except Exception as exc:
        print(f"Error: {exc}")
        return
    print(f"Episode: {result['episode_id']}")
    print(f"Run: {result['run_id']}")
    print(f"Status: {result['status']}")
    if result["task_input"]:
        print(f"Task: {result['task_input']}")
    print(f"Trace: {result['trace_path']}")


if __name__ == "__main__":
    main()
