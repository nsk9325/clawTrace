from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    max_steps: int = 20
    timeout_s: int = 300
    backend: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    tool_timeout_s: int = 30
    output_dir: str = "traces"
    openai_base_url: str = ""
    allow_parallel_tools: bool = False
    max_parallel_tools: int = 4
    enable_subagents: bool = True
    max_subagents_per_parent: int = 2
    max_subagents_total: int = 2
    max_subagent_depth: int = 1
    max_concurrent_subagents: int = 2
    subagent_parallelism: str = "serial"
    runs_per_task: int = 1
    system_prompt: str = "agent"
    append_system_prompt: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG: dict[str, Any] = asdict(RunConfig())


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)

    if path is None:
        return config

    config_path = Path(path)
    if not config_path.exists():
        return config

    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(loaded)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
