"""Profile a ClawTrace agent against SWE-bench instances.

Reads a JSONL of instances, optionally filters them, then runs each through
ClawTrace's run_episode, capturing trace + diff + predictions files for
later analysis or evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import subagent  # noqa: F401  (registers spawn_subagent)
from engine import run_episode
from swebench_dispenser import (
    build_workload,
    capture_diff,
    load_instances,
    reset_repo,
    select,
    write_predictions,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--instances", required=True, type=Path, help="Path to JSONL of SWE-bench instances")
    p.add_argument("--repos-dir", required=True, type=Path, help="Directory containing per-instance clones")
    p.add_argument("--instance-id", default=None, help="Run only the instance with this id")
    p.add_argument("--repo", default=None, help="Filter to instances from this repo (e.g. django/django)")
    p.add_argument("--min-ps-len", default=None, type=int, help="Minimum problem_statement length (chars)")
    p.add_argument("--max-ps-len", default=None, type=int, help="Maximum problem_statement length (chars)")
    p.add_argument("--limit", default=None, type=int, help="Cap the number of instances run")
    p.add_argument("--system-prompt", default=None, help="Shortcut for cfg.system_prompt")
    p.add_argument("--output-dir", default=None, type=Path, help="Shortcut for cfg.output_dir")
    p.add_argument("--config", default=None, help='Arbitrary cfg overrides as JSON, e.g. \'{"model":"gpt-4o","max_steps":20}\'')
    return p.parse_args(argv)


def _build_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.system_prompt is not None:
        cfg["system_prompt"] = args.system_prompt
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir.resolve())
    if args.config is not None:
        cfg.update(json.loads(args.config))
    # The per-instance loop chdirs into each repo before run_episode runs, so
    # any relative output_dir would resolve inside the wrong cwd and traces
    # would land inside the SWE-bench repo (and get swept into capture_diff).
    out_dir = Path(cfg.get("output_dir", "traces"))
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    cfg["output_dir"] = str(out_dir.resolve())
    return cfg


def _run_one(
    instance: dict[str, Any],
    repos_dir: Path,
    cfg_overrides: dict[str, Any],
    original_cwd: Path,
) -> tuple[bool, str]:
    """Run one instance end-to-end. Returns (success, summary_line)."""
    instance_id = instance.get("instance_id", "<unknown>")

    try:
        workload = build_workload(instance, repos_dir)
        reset_repo(workload.repo_path, workload.characteristics.base_commit)
    except Exception as exc:
        return False, f"{instance_id} status=skipped reason={type(exc).__name__}: {exc}"

    started = time.perf_counter()
    os.chdir(workload.repo_path)
    try:
        result = run_episode(
            task_input=workload.task_input,
            config=cfg_overrides,
            workload_info=asdict(workload.characteristics),
        )
        wall_ms = int((time.perf_counter() - started) * 1000)
    except Exception as exc:
        return False, f"{workload.instance_id} status=crashed reason={type(exc).__name__}: {exc}"
    finally:
        os.chdir(original_cwd)

    warning = ""
    try:
        diff = capture_diff(workload.repo_path, workload.characteristics.base_commit)
        trace_path = Path(result["trace_path"])
        trace_dir = trace_path.parent
        trace_stem = trace_path.stem
        (trace_dir / f"{trace_stem}.diff").write_text(diff, encoding="utf-8")
        write_predictions(
            workload.instance_id,
            diff,
            trace_dir / f"{trace_stem}.predictions.json",
        )
    except Exception as exc:
        warning = f" warning=capture_failed:{type(exc).__name__}"

    return True, (
        f"{workload.instance_id} status={result['status']} "
        f"steps={result['step_count']} wall={wall_ms}ms "
        f"trace={result['trace_path']}{warning}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    instances = load_instances(args.instances)
    selected = select(
        instances,
        min_ps_len=args.min_ps_len,
        max_ps_len=args.max_ps_len,
        repo=args.repo,
        instance_id=args.instance_id,
        limit=args.limit,
    )

    if not selected:
        print("No instances match the selection criteria.", file=sys.stderr)
        return 1

    cfg_overrides = _build_config(args)
    original_cwd = Path.cwd()
    print(f"Running {len(selected)} instance(s)...")

    total_started = time.perf_counter()
    success_count = 0

    for instance in selected:
        ok, summary = _run_one(instance, args.repos_dir, cfg_overrides, original_cwd)
        print(summary)
        if ok:
            success_count += 1

    total_wall_ms = int((time.perf_counter() - total_started) * 1000)
    print(f"Completed {success_count}/{len(selected)} instances; total wall={total_wall_ms}ms")

    return 0 if success_count == len(selected) else 1


if __name__ == "__main__":
    sys.exit(main())
