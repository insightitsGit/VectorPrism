"""Shared JSONL helpers for VectorPrism data contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Line {lineno} of {path} is not a JSON object")
            yield obj


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"No records loaded from {path}")
    return rows


def require_keys(obj: Dict[str, Any], keys: List[str], *, path: str, lineno: int) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ValueError(f"{path}:{lineno} missing keys {missing}: {obj!r}")
