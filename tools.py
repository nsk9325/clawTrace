from __future__ import annotations

import difflib
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


def _unified_diff(old: str, new: str, filename: str, context_lines: int = 3) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            n=context_lines,
        )
    )


def _truncate_diff(diff_text: str, max_lines: int = 80) -> str:
    lines = diff_text.splitlines()
    if len(lines) <= max_lines:
        return diff_text
    remaining = len(lines) - max_lines
    return "\n".join(lines[:max_lines]) + f"\n\n[... {remaining} more lines ...]"


def _write_file(file_path: str, content: str) -> str:
    path = Path(file_path)
    try:
        is_new = not path.exists()
        old_content = "" if is_new else path.read_text(encoding="utf-8", errors="replace")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        if is_new:
            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Created {file_path} ({line_count} lines)"

        diff = _unified_diff(old_content, content, path.name)
        if not diff:
            return f"No changes in {file_path}"
        return f"File updated — {file_path}:\n\n{_truncate_diff(diff)}"
    except Exception as exc:
        return f"Error: {exc}"


def _edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found: {file_path}"

    try:
        content = path.read_text(encoding="utf-8", errors="replace")

        crlf_count = content.count("\r\n")
        lf_count = content.count("\n")
        is_pure_crlf = crlf_count > 0 and crlf_count == lf_count

        content_norm = content.replace("\r\n", "\n")
        old_norm = old_string.replace("\r\n", "\n")
        new_norm = new_string.replace("\r\n", "\n")

        count = content_norm.count(old_norm)
        if count == 0:
            return "Error: old_string not found in file. Ensure EXACT match including whitespace."
        if count > 1 and not replace_all:
            return (
                f"Error: old_string appears {count} times. "
                "Provide more context or use replace_all=true."
            )

        new_content_norm = (
            content_norm.replace(old_norm, new_norm)
            if replace_all
            else content_norm.replace(old_norm, new_norm, 1)
        )

        if is_pure_crlf:
            final_content = new_content_norm.replace("\n", "\r\n")
            old_for_diff = content
        else:
            final_content = new_content_norm
            old_for_diff = content_norm

        path.write_text(final_content, encoding="utf-8", newline="")
        diff = _unified_diff(old_for_diff, final_content, path.name)
        return f"Changes applied to {path.name}:\n\n{_truncate_diff(diff)}"
    except Exception as exc:
        return f"Error: {exc}"


def _glob_tool(pattern: str, path: str | None = None) -> str:
    base = Path(path) if path else Path.cwd()
    try:
        matches = sorted(base.glob(pattern))
        if not matches:
            return "No files matched"
        return "\n".join(str(m) for m in matches[:500])
    except Exception as exc:
        return f"Error: {exc}"


def _has_ripgrep() -> bool:
    try:
        subprocess.run(["rg", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _grep_tool(
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    output_mode: str = "files_with_matches",
    case_insensitive: bool = False,
    context: int = 0,
) -> str:
    use_rg = _has_ripgrep()
    cmd: list[str] = ["rg", "--no-heading"] if use_rg else ["grep", "-r"]
    if case_insensitive:
        cmd.append("-i")
    if output_mode == "files_with_matches":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")
    else:
        cmd.append("-n")
        if context:
            cmd += ["-C", str(context)]
    if glob:
        cmd += (["--glob", glob] if use_rg else ["--include", glob])
    cmd.append(pattern)
    cmd.append(path or str(Path.cwd()))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        return output[:20000] if output else "No matches found"
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

    register_tool(
        ToolDef(
            name="write_file",
            schema={
                "name": "write_file",
                "description": "Write content to a file, creating parent directories as needed.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["file_path", "content"],
                },
            },
            func=lambda params, config: _write_file(
                file_path=params["file_path"],
                content=params["content"],
            ),
            read_only=False,
            concurrent_safe=False,
        )
    )

    register_tool(
        ToolDef(
            name="edit_file",
            schema={
                "name": "edit_file",
                "description": (
                    "Replace exact text in a file. old_string must match exactly. "
                    "Use replace_all=true for multiple occurrences."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            },
            func=lambda params, config: _edit_file(
                file_path=params["file_path"],
                old_string=params["old_string"],
                new_string=params["new_string"],
                replace_all=params.get("replace_all", False),
            ),
            read_only=False,
            concurrent_safe=False,
        )
    )

    register_tool(
        ToolDef(
            name="glob",
            schema={
                "name": "glob",
                "description": "Find files matching a glob pattern. Returns sorted list of matching paths.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["pattern"],
                },
            },
            func=lambda params, config: _glob_tool(
                pattern=params["pattern"],
                path=params.get("path"),
            ),
            read_only=True,
            concurrent_safe=True,
        )
    )

    register_tool(
        ToolDef(
            name="grep",
            schema={
                "name": "grep",
                "description": "Search file contents with regex. Uses ripgrep when available.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                        "glob": {"type": "string"},
                        "output_mode": {
                            "type": "string",
                            "enum": ["content", "files_with_matches", "count"],
                        },
                        "case_insensitive": {"type": "boolean"},
                        "context": {"type": "integer"},
                    },
                    "required": ["pattern"],
                },
            },
            func=lambda params, config: _grep_tool(
                pattern=params["pattern"],
                path=params.get("path"),
                glob=params.get("glob"),
                output_mode=params.get("output_mode", "files_with_matches"),
                case_insensitive=params.get("case_insensitive", False),
                context=params.get("context", 0),
            ),
            read_only=True,
            concurrent_safe=True,
        )
    )

_register_builtins()