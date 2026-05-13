from __future__ import annotations

import platform
import subprocess
from datetime import date
from pathlib import Path


MINIMAL = """\
You are an assistant inside ClawTrace.
Use the provided tools when they help you complete the task.
You may call zero, one, or multiple tools in a turn.
When you have enough information, answer normally without calling tools.
"""


AGENT_TEMPLATE = """\
You are a coding agent operating inside ClawTrace.
Your job is to investigate, change, and verify code in the working directory using the tools provided.

# Environment
- Date: {date}
- Working directory: {cwd}
- Platform: {platform}
{git_info}
# How to work
- Read before you write. Inspect relevant files before making changes.
- Make small, verifiable changes. Prefer editing existing files to creating new ones.
- Reference code as path:line when discussing it.
- Be concise. Assume the reader can read the diff.

# Exploration discipline
- Search before reading. Use `grep` for symbols and `glob` for file names to locate code; don't read whole files just to find a function.
- When a file is large, use `read_file` with `offset` and `limit`. Every successful read tells you the total line count (`# file has N lines`) — plan offsets from that, not from guesses.
- Don't re-read content already in your transcript. If you just read lines 1–100, lines 1–100 are still in your context — don't request them again.
{parallel_tools_block}{subagents_block}
# When to stop
- When the task is complete, respond without calling any tools.
"""


PARALLEL_TOOLS_BLOCK = """
# Parallel tool use
- You may emit multiple tool calls in one turn; concurrent-safe ones run in parallel.
- Batch independent reads (read_file, glob, grep) into a single turn instead of serializing them.
"""


SUBAGENTS_BLOCK = """
# Subagents
- Use spawn_subagent for self-contained sub-tasks that benefit from a fresh context window.
- Subagents do not see your transcript and return only their final text as the tool result.
- Delegate when a sub-task is large enough to warrant its own context, not for trivial work.
"""


REGISTRY: dict[str, str] = {
    "minimal": MINIMAL,
    "agent": AGENT_TEMPLATE,
}


def build_system_prompt(cfg: "RunConfig", is_subagent: bool = False) -> str:
    name = "minimal" if is_subagent else cfg.system_prompt
    template = REGISTRY.get(name)

    if template is None:
        return _append(cfg.system_prompt, cfg.append_system_prompt)

    rendered = template.format(
        date=date.today().isoformat(),
        cwd=str(Path.cwd()),
        platform=platform.system(),
        git_info=_git_info(),
        parallel_tools_block=PARALLEL_TOOLS_BLOCK if cfg.allow_parallel_tools else "",
        subagents_block=SUBAGENTS_BLOCK if cfg.enable_subagents else "",
    )
    return _append(rendered, cfg.append_system_prompt)


def _append(prompt: str, append: str) -> str:
    extra = append.strip()
    if not extra:
        return prompt
    return prompt.rstrip() + "\n\n" + extra + "\n"


def _git_info() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    branch = result.stdout.strip()
    if not branch:
        return ""
    return f"- Git branch: {branch}\n"
