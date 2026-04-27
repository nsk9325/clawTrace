from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, event: dict[str, Any]) -> None:
        with self._lock:
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
