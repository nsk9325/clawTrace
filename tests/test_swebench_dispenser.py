from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_module(module_name: str, file_name: str):
    module_path = Path(__file__).resolve().parent.parent / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dispenser = _load_module("swebench_dispenser", "swebench_dispenser.py")


def _stub_instance(**overrides):
    base = {
        "repo": "fake/repo",
        "instance_id": "fake__repo-001",
        "base_commit": "0" * 40,
        "patch": "diff --git a/x b/x\n",
        "test_patch": "diff --git a/y b/y\n",
        "problem_statement": "Fix a bug.",
        "hints_text": "",
        "created_at": "2024-01-01T00:00:00Z",
        "FAIL_TO_PASS": "[\"test_a\", \"test_b\"]",
        "PASS_TO_PASS": "[]",
    }
    base.update(overrides)
    return base


def _init_git_repo(repo_path: Path) -> str:
    repo_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repo_path), "init", "-q"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo_path), "config", "user.email", "t@t.t"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo_path), "config", "user.name", "t"], check=True, capture_output=True)
    (repo_path / "file.py").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo_path), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo_path), "commit", "-q", "-m", "init"], check=True, capture_output=True)
    sha = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return sha


def test_load_instances_parses_jsonl(tmp_path):
    p = tmp_path / "tiny.jsonl"
    p.write_text(
        json.dumps(_stub_instance(instance_id="a")) + "\n"
        + json.dumps(_stub_instance(instance_id="b")) + "\n",
        encoding="utf-8",
    )
    rows = dispenser.load_instances(p)
    assert len(rows) == 2
    assert [r["instance_id"] for r in rows] == ["a", "b"]


def test_characterize_extracts_fields():
    inst = _stub_instance(
        problem_statement="Hello world",
        hints_text="some hint",
        FAIL_TO_PASS="[\"a\", \"b\", \"c\"]",
        PASS_TO_PASS="[\"d\", \"e\"]",
    )
    c = dispenser.characterize(inst)
    assert c.instance_id == "fake__repo-001"
    assert c.repo == "fake/repo"
    assert c.problem_statement_length == len("Hello world")
    assert c.hints_present is True
    assert c.num_fail_to_pass == 3
    assert c.num_pass_to_pass == 2
    assert c.base_commit == "0" * 40
    assert c.created_at == "2024-01-01T00:00:00Z"


def test_characterize_treats_blank_hints_as_absent():
    inst = _stub_instance(hints_text="   ")
    assert dispenser.characterize(inst).hints_present is False


def test_select_filters_by_repo_length_id_and_limit():
    instances = [
        _stub_instance(instance_id="a", repo="r1", problem_statement="x" * 100),
        _stub_instance(instance_id="b", repo="r2", problem_statement="x" * 200),
        _stub_instance(instance_id="c", repo="r2", problem_statement="x" * 300),
        _stub_instance(instance_id="d", repo="r2", problem_statement="x" * 400),
    ]
    assert [i["instance_id"] for i in dispenser.select(instances, repo="r2")] == ["b", "c", "d"]
    assert [i["instance_id"] for i in dispenser.select(instances, min_ps_len=250)] == ["c", "d"]
    assert [i["instance_id"] for i in dispenser.select(instances, max_ps_len=200)] == ["a", "b"]
    assert [i["instance_id"] for i in dispenser.select(instances, instance_id="b")] == ["b"]
    assert [i["instance_id"] for i in dispenser.select(instances, repo="r2", limit=2)] == ["b", "c"]


def test_render_task_includes_problem_statement_and_repo_path():
    inst = _stub_instance(problem_statement="The bug is X.")
    out = dispenser.render_task(inst, Path("/tmp/some/repo"))
    assert "The bug is X." in out
    assert "/tmp/some/repo" in out


def test_reset_repo_restores_modified_and_removes_untracked(tmp_path):
    repo = tmp_path / "fake_repo"
    sha = _init_git_repo(repo)
    (repo / "file.py").write_text("MODIFIED\n", encoding="utf-8")
    (repo / "new_file.py").write_text("untracked\n", encoding="utf-8")

    dispenser.reset_repo(repo, sha)

    assert (repo / "file.py").read_text(encoding="utf-8") == "original\n"
    assert not (repo / "new_file.py").exists()


def test_capture_diff_covers_modifications_and_new_files(tmp_path):
    repo = tmp_path / "fake_repo"
    sha = _init_git_repo(repo)
    (repo / "file.py").write_text("MODIFIED\n", encoding="utf-8")
    (repo / "new_file.py").write_text("untracked\n", encoding="utf-8")

    diff = dispenser.capture_diff(repo, sha)

    assert "file.py" in diff
    assert "new_file.py" in diff
    assert "MODIFIED" in diff
    assert "untracked" in diff


def test_build_workload_constructs_workload_when_commit_present(tmp_path):
    inst_id = "fake__repo-001"
    repo = tmp_path / inst_id
    sha = _init_git_repo(repo)

    inst = _stub_instance(instance_id=inst_id, base_commit=sha)
    w = dispenser.build_workload(inst, tmp_path)
    assert w.repo_path == repo
    assert w.characteristics.base_commit == sha
    assert "Fix a bug." in w.task_input


def test_build_workload_raises_on_missing_repo(tmp_path):
    inst = _stub_instance(instance_id="missing__instance-1")
    with pytest.raises(FileNotFoundError):
        dispenser.build_workload(inst, tmp_path)


def test_build_workload_raises_on_unknown_base_commit(tmp_path):
    inst_id = "fake__repo-001"
    repo = tmp_path / inst_id
    _init_git_repo(repo)
    inst = _stub_instance(instance_id=inst_id, base_commit="0" * 40)
    with pytest.raises(ValueError):
        dispenser.build_workload(inst, tmp_path)


def test_write_predictions_writes_swebench_format(tmp_path):
    out = tmp_path / "predictions.json"
    dispenser.write_predictions("foo__bar-1", "diff content", out)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 1
    assert payload[0]["instance_id"] == "foo__bar-1"
    assert payload[0]["model_patch"] == "diff content"
    assert payload[0]["model_name_or_path"] == "clawtrace"
