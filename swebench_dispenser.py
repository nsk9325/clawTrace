"""SWE-bench workload dispenser.

Loads SWE-bench instances from a JSONL file, characterizes them, selects subsets,
renders them into ClawTrace-friendly Workload objects, and exposes git helpers
for repo reset and diff capture between runs.

Convention: each instance has its own clone at <repos_dir>/<instance_id>/, with
HEAD verified to contain the instance's base_commit before use.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InstanceCharacteristics:
    instance_id: str
    repo: str
    problem_statement_length: int
    hints_present: bool
    num_fail_to_pass: int
    num_pass_to_pass: int
    base_commit: str
    created_at: str | None = None


@dataclass(frozen=True)
class Workload:
    instance_id: str
    repo_path: Path
    task_input: str
    characteristics: InstanceCharacteristics
    raw_instance: dict[str, Any]


def load_instances(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def characterize(instance: dict[str, Any]) -> InstanceCharacteristics:
    fail_raw = instance.get("FAIL_TO_PASS", "[]")
    pass_raw = instance.get("PASS_TO_PASS", "[]")
    fail_list = json.loads(fail_raw) if isinstance(fail_raw, str) else fail_raw
    pass_list = json.loads(pass_raw) if isinstance(pass_raw, str) else pass_raw

    return InstanceCharacteristics(
        instance_id=str(instance["instance_id"]),
        repo=str(instance["repo"]),
        problem_statement_length=len(str(instance.get("problem_statement", ""))),
        hints_present=bool(str(instance.get("hints_text", "")).strip()),
        num_fail_to_pass=len(fail_list),
        num_pass_to_pass=len(pass_list),
        base_commit=str(instance["base_commit"]),
        created_at=instance.get("created_at"),
    )


def select(
    instances: list[dict[str, Any]],
    *,
    min_ps_len: int | None = None,
    max_ps_len: int | None = None,
    repo: str | None = None,
    instance_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for instance in instances:
        chars = characterize(instance)
        if instance_id is not None and chars.instance_id != instance_id:
            continue
        if repo is not None and chars.repo != repo:
            continue
        if min_ps_len is not None and chars.problem_statement_length < min_ps_len:
            continue
        if max_ps_len is not None and chars.problem_statement_length > max_ps_len:
            continue
        out.append(instance)
        if limit is not None and len(out) >= limit:
            break
    return out


def build_workload(instance: dict[str, Any], repos_dir: Path) -> Workload:
    chars = characterize(instance)
    repo_path = Path(repos_dir) / chars.instance_id
    if not repo_path.exists():
        raise FileNotFoundError(
            f"Repo not found: {repo_path} "
            f"(expected per-instance clone at <repos_dir>/<instance_id>/)"
        )
    result = subprocess.run(
        ["git", "-C", str(repo_path), "cat-file", "-e", f"{chars.base_commit}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"base_commit {chars.base_commit} not found in {repo_path}; "
            f"clone may be shallow or wrong"
        )
    return Workload(
        instance_id=chars.instance_id,
        repo_path=repo_path,
        task_input=render_task(instance, repo_path),
        characteristics=chars,
        raw_instance=instance,
    )


def reset_repo(repo_path: Path, base_commit: str) -> None:
    """Pin to base_commit and wipe staged changes, untracked files, and dirs."""
    subprocess.run(
        ["git", "-C", str(repo_path), "reset", "--hard", base_commit],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_path), "clean", "-fd"],
        capture_output=True,
        check=True,
    )


def capture_diff(repo_path: Path, base_commit: str) -> str:
    """Cumulative diff from base_commit to current state (incl. untracked)."""
    subprocess.run(
        ["git", "-C", str(repo_path), "add", "-A"],
        capture_output=True,
        check=True,
    )
    result = subprocess.run(
        ["git", "-C", str(repo_path), "diff", "--cached", base_commit],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def render_task(instance: dict[str, Any], repo_path: Path) -> str:
    problem = str(instance.get("problem_statement", "")).strip()
    return f"{problem}\n\nThe repo is checked out at {repo_path}. Modify files in-place."


def write_predictions(instance_id: str, model_patch: str, output_path: Path) -> None:
    """Emit a SWE-bench predictions file the eval harness can consume later."""
    payload = [{
        "instance_id": instance_id,
        "model_patch": model_patch,
        "model_name_or_path": "clawtrace",
    }]
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
