from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from conftest import load_module

local_eval = load_module("local_eval", "local_eval.py")


def test_score_classifies_pass_to_pass_regression():
    class _Spec:
        FAIL_TO_PASS = ["test_callable_path"]
        PASS_TO_PASS = ["test_existing"]

    outcomes = {
        "test_callable_path": "PASSED",
        "test_existing": "FAILED",
    }
    f2p_p, f2p_f, p2p_p, p2p_f = local_eval._score(outcomes, _Spec())
    assert f2p_p == ["test_callable_path"]
    assert f2p_f == []
    assert p2p_p == []
    assert p2p_f == ["test_existing"]


def test_score_resolves_when_all_targets_pass():
    class _Spec:
        FAIL_TO_PASS = ["a", "b"]
        PASS_TO_PASS = ["c"]

    outcomes = {"a": "PASSED", "b": "PASSED", "c": "PASSED"}
    f2p_p, f2p_f, p2p_p, p2p_f = local_eval._score(outcomes, _Spec())
    assert not f2p_f and not p2p_f


def test_alias_py311_unittest_names_back_to_legacy_form():
    outcomes = {
        "test_callable_path (model_fields.test_filepathfield.FilePathFieldTests.test_callable_path)": "PASSED",
        "test_path (model_fields.test_filepathfield.FilePathFieldTests.test_path)": "FAILED",
    }
    aliased = local_eval._alias_py311_unittest_names(outcomes)
    assert aliased["test_callable_path (model_fields.test_filepathfield.FilePathFieldTests)"] == "PASSED"
    assert aliased["test_path (model_fields.test_filepathfield.FilePathFieldTests)"] == "FAILED"
    # New-shape keys preserved too.
    assert aliased["test_callable_path (model_fields.test_filepathfield.FilePathFieldTests.test_callable_path)"] == "PASSED"


def test_alias_py311_unittest_names_passes_pytest_format_through():
    outcomes = {"tests/foo.py::test_a": "PASSED"}
    assert local_eval._alias_py311_unittest_names(outcomes) == outcomes


def test_between_markers_extracts_relevant_section():
    log = (
        "preamble lines\n"
        ">>>>> Start Test Output\n"
        "PASSED tests/x.py::test_a\n"
        "FAILED tests/x.py::test_b\n"
        ">>>>> End Test Output\n"
        "trailing junk\n"
    )
    section = local_eval._between_markers(log)
    assert "PASSED tests/x.py::test_a" in section
    assert "preamble lines" not in section
    assert "trailing junk" not in section


def test_between_markers_falls_back_to_full_log_when_markers_missing():
    log = "no markers at all\nPASSED x\n"
    assert local_eval._between_markers(log) == log


def test_localize_script_strips_conda_and_rewrites_paths(tmp_path):
    venv = tmp_path / "venv"
    venv.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()

    docker_lines = [
        "source /opt/miniconda3/bin/activate",
        "conda activate testbed",
        "cd /testbed",
        "sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen",
        "export LANG=en_US.UTF-8",
        "git config --global --add safe.directory /testbed",
        "git status",
        "python -m pip install -e .",
        "./tests/runtests.py model_fields.test_filepathfield",
    ]
    script = local_eval._localize_script(docker_lines, repo, venv)

    assert "conda activate" not in script
    assert "/opt/miniconda3" not in script
    assert "locale-gen" not in script
    assert "git config" not in script
    assert "git status" not in script
    # `pip install -e .` is kept on purpose — re-binds the editable install
    # to the current instance's repo (env_setup caches venvs per version).
    assert "pip install -e ." in script
    assert f"source {venv}/bin/activate" in script
    assert f"cd {repo}" in script
    assert "/testbed" not in script
    assert "./tests/runtests.py model_fields.test_filepathfield" in script


def test_evaluate_one_reports_missing_repo(tmp_path):
    pred = {"instance_id": "fake__repo-1", "model_patch": ""}
    inst = {"instance_id": "fake__repo-1", "repo": "astropy/astropy",
            "version": "4.3", "base_commit": "deadbeef"}
    venv = tmp_path / "venv"
    venv.mkdir()
    report = local_eval.evaluate_one(
        prediction=pred, instance=inst,
        repo_path=tmp_path / "not-there",
        venv_path=venv,
        log_dir=tmp_path / "logs",
    )
    assert report.resolved is False
    assert "repo not found" in (report.error or "")


def test_evaluate_one_reports_missing_venv(tmp_path):
    pred = {"instance_id": "fake__repo-2", "model_patch": ""}
    inst = {"instance_id": "fake__repo-2", "repo": "astropy/astropy",
            "version": "4.3", "base_commit": "deadbeef"}
    repo = tmp_path / "repo"
    repo.mkdir()
    report = local_eval.evaluate_one(
        prediction=pred, instance=inst,
        repo_path=repo,
        venv_path=tmp_path / "not-there",
        log_dir=tmp_path / "logs",
    )
    assert report.resolved is False
    assert "venv not found" in (report.error or "")
