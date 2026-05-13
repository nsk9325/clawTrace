"""Per-instance Python env build for SWE-bench workloads.

Looks up the install recipe for ``(repo, version)`` from the official
``swebench`` package's ``MAP_REPO_VERSION_TO_SPECS`` table, then materializes
a cached venv under ``<envs_root>/<repo_slug>__v<version>/`` and runs the
``pre_install`` / ``pip install`` / ``install`` commands inside the repo.

The venv is reused across instances of the same (repo, version) pair; a
``.ready`` marker file at the venv root distinguishes a finished build from a
half-built one (only ``.ready`` venvs are considered cached).

Returns an :class:`EnvSetupResult` carrying status, venv path, and timing.
Callers translate that into a trace ``env_setup`` event and into the
``RuntimeContext.env_overrides`` dict consumed by ``tools._bash``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EnvSetupResult:
    status: str  # "ok" | "cached" | "skipped" | "failed"
    venv_path: Path | None
    python_version: str | None
    install_commands: list[str] = field(default_factory=list)
    duration_ms: int = 0
    error: str | None = None

    def env_overrides(self) -> dict[str, str]:
        """Env vars to inject into the agent's bash subprocesses."""
        if self.venv_path is None:
            return {}
        return _venv_env_overrides(self.venv_path)

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "venv_path": str(self.venv_path) if self.venv_path else None,
            "python_version": self.python_version,
            "install_commands": list(self.install_commands),
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


def _venv_env_overrides(venv_path: Path) -> dict[str, str]:
    bin_dir = venv_path / "bin"
    return {
        "VIRTUAL_ENV": str(venv_path),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
    }


def _load_spec(repo: str, version: str) -> dict[str, Any] | None:
    try:
        from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
    except ImportError:
        return None
    repo_specs = MAP_REPO_VERSION_TO_SPECS.get(repo)
    if not repo_specs:
        return None
    return repo_specs.get(str(version))


def _slug(repo: str, version: str) -> str:
    return f"{repo.replace('/', '__')}__v{version}"


def _make_venv(venv_path: Path, python_version: str) -> tuple[bool, str]:
    """Create a venv at ``venv_path`` with pip seeded.

    Without pip in the venv, the spec's ``pip install ...`` commands resolve
    to whatever ``pip`` is next on PATH (typically the *parent* venv) and
    silently install into the wrong place — which is exactly the dependency
    isolation failure mode this whole module exists to prevent.
    """
    venv_path.parent.mkdir(parents=True, exist_ok=True)
    uv = shutil.which("uv")
    if uv:
        result = subprocess.run(
            [uv, "venv", "--seed", "--python", python_version, str(venv_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return True, ""
        # Fall through to system python on uv failure (e.g. offline, no matching cpython).

    result = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()
    return True, ""


def _restore_tracked_files(repo_path: Path) -> None:
    """`git checkout -- .` inside repo_path; ignore failure if it isn't a git tree."""
    subprocess.run(
        ["git", "-C", str(repo_path), "checkout", "--", "."],
        capture_output=True,
    )


def _run_in_venv(
    venv_path: Path,
    repo_path: Path,
    command: str,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run a shell command with the venv on PATH and inside ``repo_path``."""
    env = os.environ.copy()
    env.update(_venv_env_overrides(venv_path))
    # Avoid touching unrelated GPUs on shared hosts: builds don't need a GPU.
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    # pip's PyPI version check fails noisily when offline or rate-limited and
    # adds nothing useful to env_setup's output.
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    return subprocess.run(
        command,
        shell=True,
        cwd=str(repo_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def setup_env(
    *,
    repo: str,
    version: str,
    repo_path: Path,
    envs_root: Path,
    install_timeout_s: int = 900,
) -> EnvSetupResult:
    """Build (or reuse) a cached venv for ``(repo, version)`` and install deps."""
    started = time.perf_counter()
    elapsed_ms = lambda: int((time.perf_counter() - started) * 1000)

    spec = _load_spec(repo, version)
    if spec is None:
        return EnvSetupResult(
            status="skipped",
            venv_path=None,
            python_version=None,
            error=f"no spec for ({repo}, {version}) — swebench package missing or unknown pair",
            duration_ms=elapsed_ms(),
        )

    python_version = str(spec.get("python", ""))
    venv_path = envs_root / _slug(repo, version)
    ready_marker = venv_path / ".ready"

    if not repo_path.is_dir():
        return EnvSetupResult(
            status="failed",
            venv_path=None,
            python_version=python_version,
            error=f"repo_path does not exist: {repo_path}",
            duration_ms=elapsed_ms(),
        )

    pre_install = list(spec.get("pre_install") or [])
    pip_packages = list(spec.get("pip_packages") or [])
    install_cmd = str(spec.get("install") or "").strip()
    install_commands: list[str] = []
    if pre_install:
        install_commands.extend(pre_install)
    if pip_packages:
        install_commands.append("pip install " + " ".join(repr(p) for p in pip_packages))
    if install_cmd:
        install_commands.append(install_cmd)

    if ready_marker.exists():
        return EnvSetupResult(
            status="cached",
            venv_path=venv_path,
            python_version=python_version,
            install_commands=install_commands,
            duration_ms=elapsed_ms(),
        )

    # Stale half-built venv from a prior failed attempt — wipe before retrying.
    if venv_path.exists():
        shutil.rmtree(venv_path, ignore_errors=True)

    ok, err = _make_venv(venv_path, python_version)
    if not ok:
        return EnvSetupResult(
            status="failed",
            venv_path=None,
            python_version=python_version,
            install_commands=install_commands,
            error=f"venv creation failed: {err}",
            duration_ms=elapsed_ms(),
        )

    for cmd in install_commands:
        try:
            proc = _run_in_venv(venv_path, repo_path, cmd, timeout=install_timeout_s)
        except subprocess.TimeoutExpired:
            return EnvSetupResult(
                status="failed",
                venv_path=None,
                python_version=python_version,
                install_commands=install_commands,
                error=f"install timed out after {install_timeout_s}s on: {cmd}",
                duration_ms=elapsed_ms(),
            )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-20:]
            return EnvSetupResult(
                status="failed",
                venv_path=None,
                python_version=python_version,
                install_commands=install_commands,
                error=f"install failed on `{cmd}`:\n" + "\n".join(tail),
                duration_ms=elapsed_ms(),
            )

    # Revert tracked-file mutations from pre_install commands (e.g. sed edits
    # to pyproject.toml) so they don't leak into the agent's working tree or
    # the diff captured for the SWE-bench predictions file.
    _restore_tracked_files(repo_path)

    ready_marker.write_text("ok\n", encoding="utf-8")
    return EnvSetupResult(
        status="ok",
        venv_path=venv_path,
        python_version=python_version,
        install_commands=install_commands,
        duration_ms=elapsed_ms(),
    )
