from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class JsonStorage:
    """Atomic JSON persistence for DSM state."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.replace(self.path)
