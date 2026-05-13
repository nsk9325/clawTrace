"""Local SWE-bench-style evaluation, no Docker.

For each prediction, applies the model_patch to the per-instance repo
checkout, runs the canonical eval script (test_patch + filtered test
command) inside the per-instance venv that ``env_setup.py`` built, and
scores the outcome against ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` using
swebench's per-repo log parser.

This is **not** byte-identical to the official Docker-based harness —
no container, no install-step re-verification, no curated base image.
It produces the same per-test pass/fail signal and the same
``resolved`` verdict, which is what's needed when iterating on agent
correctness locally.

Usage::

    python local_eval.py traces/episode_<id>/episode_<id>.predictions.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class InstanceReport:
    instance_id: str
    resolved: bool
    fail_to_pass_passing: list[str] = field(default_factory=list)
    fail_to_pass_failing: list[str] = field(default_factory=list)
    pass_to_pass_passing: list[str] = field(default_factory=list)
    pass_to_pass_failing: list[str] = field(default_factory=list)
    error: str | None = None
    log_path: str | None = None


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _index_instances(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["instance_id"]] = row
    return rows


def _slug(repo: str, version: str) -> str:
    return f"{repo.replace('/', '__')}__v{version}"


def _reset_repo(repo: Path, base_commit: str) -> None:
    subprocess.run(["git", "-C", str(repo), "reset", "--hard", base_commit],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "clean", "-fd"],
                   check=True, capture_output=True)


def _apply_patch(repo: Path, patch: str) -> tuple[bool, str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), "apply", "--whitespace=nowarn", "-"],
        input=patch, text=True, capture_output=True,
    )
    return (proc.returncode == 0, proc.stderr)


# Lines in eval_script_list that assume the Docker container layout. We strip
# or rewrite them so the same script runs against a local repo + venv.
_CONDA_LINE = re.compile(r"^\s*(source /opt/miniconda3|conda activate|conda create)\b")
_LOCALE_LINE = re.compile(r"locale-gen|sed -i '/en_US\.UTF-8/")
_NOOP_LINE = re.compile(r"^\s*git (status|show|diff|config)\b")


def _localize_script(script_lines: list[str], repo: Path, venv: Path) -> str:
    out: list[str] = [
        f"set -e",
        f"source {venv}/bin/activate",
        f"cd {repo}",
    ]
    for line in script_lines:
        if _CONDA_LINE.search(line):
            continue
        if _LOCALE_LINE.search(line):
            continue
        if _NOOP_LINE.search(line):
            continue
        # Note: we intentionally keep `python -m pip install -e .` lines.
        # env_setup caches venvs per (repo, version) but editable installs
        # bind to one repo path; re-running pip install -e . per eval points
        # the venv at the current instance's repo, matching the official
        # harness's behavior.
        out.append(line.replace("/testbed", str(repo)))
    return "\n".join(out) + "\n"


_EVAL_START = ">>>>> Start Test Output"
_EVAL_END = ">>>>> End Test Output"

# Python 3.11+ unittest verbose output appends the method name a second time
# inside the parens (`test_x (mod.Class.test_x)`); ≤3.10 omits it
# (`test_x (mod.Class)`). SWE-bench's curated FAIL_TO_PASS / PASS_TO_PASS
# strings are keyed to the legacy form, so we alias the new shape back.
_PY311_UNITTEST_NAME = re.compile(r"^(\w+) \(([\w.]+)\.\1\)$")


def _alias_py311_unittest_names(outcomes: dict[str, str]) -> dict[str, str]:
    aliased = dict(outcomes)
    for key, status in outcomes.items():
        m = _PY311_UNITTEST_NAME.match(key)
        if m:
            aliased.setdefault(f"{m.group(1)} ({m.group(2)})", status)
    return aliased


def _between_markers(log: str) -> str:
    start = log.find(_EVAL_START)
    end = log.find(_EVAL_END)
    if start < 0 or end < 0 or end < start:
        return log
    return log[start + len(_EVAL_START):end]


def _score(test_outcomes: dict[str, str], spec: Any) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split FAIL_TO_PASS / PASS_TO_PASS into passing / failing lists."""
    def passes(name: str) -> bool:
        return test_outcomes.get(name) == "PASSED"
    f2p_pass = [t for t in spec.FAIL_TO_PASS if passes(t)]
    f2p_fail = [t for t in spec.FAIL_TO_PASS if not passes(t)]
    p2p_pass = [t for t in spec.PASS_TO_PASS if passes(t)]
    p2p_fail = [t for t in spec.PASS_TO_PASS if not passes(t)]
    return f2p_pass, f2p_fail, p2p_pass, p2p_fail


def evaluate_one(
    prediction: dict[str, Any],
    instance: dict[str, Any],
    repo_path: Path,
    venv_path: Path,
    log_dir: Path,
) -> InstanceReport:
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
    from swebench.harness.test_spec.test_spec import make_test_spec

    instance_id = instance["instance_id"]
    log_path = log_dir / f"{instance_id}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight existence checks first — clearer error than the KeyError
    # make_test_spec raises on a malformed instance row.
    if not repo_path.is_dir():
        return InstanceReport(instance_id=instance_id, resolved=False,
                              error=f"repo not found: {repo_path}")
    if not venv_path.is_dir():
        return InstanceReport(instance_id=instance_id, resolved=False,
                              error=f"venv not found: {venv_path}")

    spec = make_test_spec(instance)
    parser = MAP_REPO_TO_PARSER.get(spec.repo)
    if parser is None:
        return InstanceReport(instance_id=instance_id, resolved=False,
                              error=f"no log parser for {spec.repo}")

    try:
        _reset_repo(repo_path, instance["base_commit"])
    except subprocess.CalledProcessError as exc:
        return InstanceReport(instance_id=instance_id, resolved=False,
                              error=f"reset failed: {exc.stderr.decode() if exc.stderr else exc}")

    patch = prediction.get("model_patch") or ""
    if patch.strip():
        ok, err = _apply_patch(repo_path, patch)
        if not ok:
            return InstanceReport(instance_id=instance_id, resolved=False,
                                  error=f"model_patch apply failed:\n{err}")

    script = _localize_script(list(spec.eval_script_list), repo_path, venv_path)
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=1800)
    log = (proc.stdout or "") + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")
    log_path.write_text(log, encoding="utf-8")

    outcomes = _alias_py311_unittest_names(parser(_between_markers(log), spec))
    f2p_pass, f2p_fail, p2p_pass, p2p_fail = _score(outcomes, spec)
    resolved = not f2p_fail and not p2p_fail

    return InstanceReport(
        instance_id=instance_id,
        resolved=resolved,
        fail_to_pass_passing=f2p_pass,
        fail_to_pass_failing=f2p_fail,
        pass_to_pass_passing=p2p_pass,
        pass_to_pass_failing=p2p_fail,
        log_path=str(log_path),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("predictions", type=Path, help="Path to predictions JSON")
    p.add_argument("--instances", type=Path, default=Path("swebench_lite.jsonl"))
    p.add_argument("--repos-dir", type=Path, default=Path("swebench-repos"))
    p.add_argument("--envs-dir", type=Path, default=Path("swebench-envs"))
    p.add_argument("--run-id", default="local")
    p.add_argument("--output-dir", type=Path, default=Path("eval-logs"))
    p.add_argument("--instance-ids", nargs="*", default=None,
                   help="Restrict to a subset (default: all in predictions file)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    predictions = _load_predictions(args.predictions)
    instances = _index_instances(args.instances)

    if args.instance_ids:
        wanted = set(args.instance_ids)
        predictions = [p for p in predictions if p["instance_id"] in wanted]

    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"

    reports: list[InstanceReport] = []
    for pred in predictions:
        iid = pred["instance_id"]
        inst = instances.get(iid)
        if inst is None:
            reports.append(InstanceReport(instance_id=iid, resolved=False,
                                          error=f"instance not found in {args.instances}"))
            continue

        version = str(inst.get("version") or "")
        # Absolute paths: the eval script does `cd /testbed → cd <repo>`
        # multiple times; a relative path would resolve against the new
        # cwd on the second invocation and fail.
        repo_path = (args.repos_dir / iid).resolve()
        venv_path = (args.envs_dir / _slug(inst["repo"], version)).resolve()

        print(f"==> {iid} repo={repo_path} venv={venv_path}", file=sys.stderr)
        report = evaluate_one(pred, inst, repo_path, venv_path, log_dir)
        status = "RESOLVED" if report.resolved else "FAILED" if report.error is None else "ERROR"
        suffix = (f" F2P_pass={len(report.fail_to_pass_passing)}/{len(report.fail_to_pass_passing) + len(report.fail_to_pass_failing)}"
                  f" P2P_pass={len(report.pass_to_pass_passing)}/{len(report.pass_to_pass_passing) + len(report.pass_to_pass_failing)}")
        if report.error:
            suffix = f" error={report.error.splitlines()[0]}"
        print(f"    {status}{suffix}", file=sys.stderr)
        reports.append(report)

    summary = {
        "run_id": args.run_id,
        "total": len(reports),
        "resolved": sum(1 for r in reports if r.resolved),
        "errored": sum(1 for r in reports if r.error is not None),
        "instances": [asdict(r) for r in reports],
    }
    (run_dir / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {run_dir / 'report.json'} — resolved {summary['resolved']}/{summary['total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
