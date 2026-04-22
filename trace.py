from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_event(
    event_type: str,
    episode_id: str,
    run_id: str,
    step_id: int | None = None,
    **fields: Any,
) -> dict[str, Any]:
    event = {
        "timestamp": utc_timestamp(),
        "type": event_type,
        "episode_id": episode_id,
        "run_id": run_id,
        "step_id": step_id,
    }
    event.update(fields)
    return event


class TraceWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, event: dict[str, Any]) -> None:
        json.dump(event, self._fh, ensure_ascii=True)
        self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
