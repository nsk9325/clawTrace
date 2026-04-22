from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "max_steps": 8,
    "timeout_s": 300,
    "backend": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "max_tokens": 512,
    "tool_timeout_s": 30,
    "output_dir": "traces",
    "trace_level": "standard",
    "openai_base_url": "",
}


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
