from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class ToolDef:
    name: str
    schema: dict[str, Any]
    func: Callable[[dict[str, Any], dict[str, Any]], str]
    read_only: bool = False
    concurrent_safe: bool = False


_registry: dict[str, ToolDef] = {}


def register_tool(tool_def: ToolDef) -> None:
    _registry[tool_def.name] = tool_def


def get_tool(name: str) -> ToolDef | None:
    return _registry.get(name)


def get_all_tools() -> list[ToolDef]:
    return list(_registry.values())


def get_tool_schemas() -> list[dict[str, Any]]:
    return [tool.schema for tool in _registry.values()]


def clear_registry() -> None:
    _registry.clear()


def execute_tool(
    name: str,
    params: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> str:
    tool = get_tool(name)
    if tool is None:
        return f"Error: tool '{name}' not found."

    cfg = config or {}
    max_output = int(cfg.get("max_tool_output", 32000))

    try:
        result = tool.func(params, cfg)
    except Exception as exc:
        return f"Error executing {name}: {exc}"

    if len(result) > max_output:
        first_half = max_output // 2
        last_quarter = max_output // 4
        truncated = len(result) - first_half - last_quarter
        result = (
            result[:first_half]
            + f"\n[... {truncated} chars truncated ...]\n"
            + result[-last_quarter:]
        )

    return result


def _read_file(file_path: str, limit: int | None = None, offset: int | None = None) -> str:
    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found: {file_path}"
    if path.is_dir():
        return f"Error: {file_path} is a directory"

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        start = offset or 0
        chunk = lines[start:start + limit] if limit else lines[start:]
        if not chunk:
            return "(empty file)"
        return "".join(f"{start + index + 1:6}\t{line}" for index, line in enumerate(chunk))
    except Exception as exc:
        return f"Error: {exc}"


def _kill_proc_tree(pid: int) -> None:
    import sys

    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        return

    import signal

    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _bash(command: str, timeout: int = 30) -> str:
    import sys

    kwargs: dict[str, Any] = {
        "shell": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "cwd": os.getcwd(),
    }
    if sys.platform != "win32":
        kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(command, **kwargs)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_proc_tree(proc.pid)
            proc.wait()
            return f"Error: timed out after {timeout}s (process killed)"

        output = stdout
        if stderr:
            output += ("\n" if output else "") + "[stderr]\n" + stderr
        return output.strip() or "(no output)"
    except Exception as exc:
        return f"Error: {exc}"


def _register_builtins() -> None:
    register_tool(
        ToolDef(
            name="read_file",
            schema={
                "name": "read_file",
                "description": "Read a file's contents with line numbers.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                    },
                    "required": ["file_path"],
                },
            },
            func=lambda params, config: _read_file(
                file_path=params["file_path"],
                limit=params.get("limit"),
                offset=params.get("offset"),
            ),
            read_only=True,
            concurrent_safe=True,
        )
    )

    register_tool(
        ToolDef(
            name="bash",
            schema={
                "name": "bash",
                "description": "Execute a shell command and return stdout/stderr.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                    "required": ["command"],
                },
            },
            func=lambda params, config: _bash(
                command=params["command"],
                timeout=params.get("timeout", config.get("tool_timeout_s", 30)),
            ),
            read_only=False,
            concurrent_safe=False,
        )
    )

_register_builtins()