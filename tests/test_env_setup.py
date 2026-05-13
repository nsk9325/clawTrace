from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from conftest import load_module

env_setup = load_module("env_setup", "env_setup.py")


def test_skipped_when_swebench_missing(tmp_path):
    with patch.object(env_setup, "_load_spec", return_value=None):
        result = env_setup.setup_env(
            repo="fake/repo",
            version="1.0",
            repo_path=tmp_path,
            envs_root=tmp_path / "envs",
        )
    assert result.status == "skipped"
    assert result.venv_path is None
    assert result.env_overrides() == {}


def test_cached_short_circuits_when_ready_marker_present(tmp_path):
    envs_root = tmp_path / "envs"
    cached_venv = envs_root / "fake__repo__v1.0"
    cached_venv.mkdir(parents=True)
    (cached_venv / ".ready").write_text("ok\n", encoding="utf-8")

    fake_spec = {"python": "3.11", "install": "pip install -e ."}
    with patch.object(env_setup, "_load_spec", return_value=fake_spec):
        result = env_setup.setup_env(
            repo="fake/repo",
            version="1.0",
            repo_path=tmp_path,
            envs_root=envs_root,
        )
    assert result.status == "cached"
    assert result.venv_path == cached_venv
    overrides = result.env_overrides()
    assert overrides["VIRTUAL_ENV"] == str(cached_venv)
    assert str(cached_venv / "bin") in overrides["PATH"]


def test_failed_when_repo_path_missing(tmp_path):
    fake_spec = {"python": "3.11", "install": "pip install -e ."}
    with patch.object(env_setup, "_load_spec", return_value=fake_spec):
        result = env_setup.setup_env(
            repo="fake/repo",
            version="1.0",
            repo_path=tmp_path / "not_there",
            envs_root=tmp_path / "envs",
        )
    assert result.status == "failed"
    assert "repo_path does not exist" in (result.error or "")


def test_failed_when_venv_creation_fails(tmp_path):
    fake_spec = {"python": "3.99", "install": "pip install -e ."}
    with patch.object(env_setup, "_load_spec", return_value=fake_spec), \
         patch.object(env_setup, "_make_venv", return_value=(False, "no python 3.99")):
        result = env_setup.setup_env(
            repo="fake/repo",
            version="1.0",
            repo_path=tmp_path,
            envs_root=tmp_path / "envs",
        )
    assert result.status == "failed"
    assert result.venv_path is None
    assert "no python 3.99" in result.error


def test_install_failure_propagates_tail_and_does_not_mark_ready(tmp_path):
    fake_spec = {"python": "3.11", "install": "pip install -e ."}
    envs_root = tmp_path / "envs"

    class _FakeProc:
        returncode = 1
        stdout = "boom: cannot resolve package\n"
        stderr = "ERROR: install fail\n"

    def _fake_make_venv(venv_path, python_version):
        venv_path.mkdir(parents=True, exist_ok=True)
        (venv_path / "bin").mkdir(exist_ok=True)
        return True, ""

    with patch.object(env_setup, "_load_spec", return_value=fake_spec), \
         patch.object(env_setup, "_make_venv", side_effect=_fake_make_venv), \
         patch.object(env_setup, "_run_in_venv", return_value=_FakeProc()):
        result = env_setup.setup_env(
            repo="fake/repo",
            version="1.0",
            repo_path=tmp_path,
            envs_root=envs_root,
        )
    assert result.status == "failed"
    assert "install fail" in (result.error or "")
    assert not (envs_root / "fake__repo__v1.0" / ".ready").exists()


def test_successful_install_writes_ready_marker(tmp_path):
    fake_spec = {
        "python": "3.11",
        "install": "pip install -e .",
        "pre_install": ["echo pre"],
        "pip_packages": ["pytest"],
    }
    envs_root = tmp_path / "envs"

    class _FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_make_venv(venv_path, python_version):
        venv_path.mkdir(parents=True, exist_ok=True)
        (venv_path / "bin").mkdir(exist_ok=True)
        return True, ""

    with patch.object(env_setup, "_load_spec", return_value=fake_spec), \
         patch.object(env_setup, "_make_venv", side_effect=_fake_make_venv), \
         patch.object(env_setup, "_run_in_venv", return_value=_FakeProc()) as run_mock:
        result = env_setup.setup_env(
            repo="fake/repo",
            version="1.0",
            repo_path=tmp_path,
            envs_root=envs_root,
        )
    assert result.status == "ok"
    assert (envs_root / "fake__repo__v1.0" / ".ready").exists()
    # Pre_install + a single pip install line for pip_packages + the install command.
    assert run_mock.call_count == 3
    assert "echo pre" in result.install_commands
    assert any("pip install" in c and "pytest" in c for c in result.install_commands)
    assert "pip install -e ." in result.install_commands


def _init_git_repo_with_pyproject(repo: Path, content: str) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t.t"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True, capture_output=True)
    (repo / "pyproject.toml").write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True, capture_output=True)


def test_post_install_restores_tracked_files_modified_by_pre_install(tmp_path):
    """pre_install commands (e.g. sed-edit pyproject.toml) must not leak into
    the captured diff — env_setup restores the working tree after install."""
    repo = tmp_path / "repo"
    original_pyproject = 'requires = ["setuptools",\n            "wheel"]\n'
    _init_git_repo_with_pyproject(repo, original_pyproject)

    fake_spec = {
        "python": "3.11",
        "install": "pip install -e .",
        "pre_install": ["sed -i 's/setuptools,/setuptools==68.0.0,/' pyproject.toml"],
    }

    class _OkProc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_make_venv(venv_path, python_version):
        venv_path.mkdir(parents=True, exist_ok=True)
        (venv_path / "bin").mkdir(exist_ok=True)
        return True, ""

    def _fake_run_in_venv(venv_path, repo_path, command, timeout):
        # Honor the sed pre_install for real so we can verify the restore step.
        subprocess.run(command, shell=True, cwd=str(repo_path), check=False, capture_output=True)
        return _OkProc()

    with patch.object(env_setup, "_load_spec", return_value=fake_spec), \
         patch.object(env_setup, "_make_venv", side_effect=_fake_make_venv), \
         patch.object(env_setup, "_run_in_venv", side_effect=_fake_run_in_venv):
        result = env_setup.setup_env(
            repo="fake/repo",
            version="1.0",
            repo_path=repo,
            envs_root=tmp_path / "envs",
        )

    assert result.status == "ok"
    # Working tree must be restored to its committed state.
    assert (repo / "pyproject.toml").read_text(encoding="utf-8") == original_pyproject
    # `git diff` should report nothing.
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet"],
        capture_output=True,
    )
    assert diff.returncode == 0, "working tree has uncommitted changes after env_setup"


def test_to_event_payload_serializes_strings(tmp_path):
    result = env_setup.EnvSetupResult(
        status="ok",
        venv_path=tmp_path / "v",
        python_version="3.11",
        install_commands=["pip install -e ."],
        duration_ms=12,
    )
    payload = result.to_event_payload()
    assert payload["status"] == "ok"
    assert payload["venv_path"] == str(tmp_path / "v")
    assert payload["python_version"] == "3.11"
    assert payload["install_commands"] == ["pip install -e ."]
    assert payload["duration_ms"] == 12
