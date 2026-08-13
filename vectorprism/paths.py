"""Resolve packaged schema.sql and example JSONL (wheel + editable)."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SCHEMA_SQL = PACKAGE_DIR / "schema.sql"
EXAMPLE_DATA_DIR = PACKAGE_DIR / "data"


def schema_sql_path() -> Path:
    path = SCHEMA_SQL
    if not path.is_file():
        raise FileNotFoundError(
            f"schema.sql missing from package at {path}. "
            "Reinstall vectorprism or apply schema from the git repo."
        )
    return path


def read_schema_sql() -> str:
    return schema_sql_path().read_text(encoding="utf-8")


def example_data_dir() -> Path:
    path = EXAMPLE_DATA_DIR
    if not path.is_dir():
        raise FileNotFoundError(
            f"example data missing from package at {path}. "
            "Reinstall vectorprism or clone the git repo."
        )
    return path


def example_jsonl(name: str) -> Path:
    path = example_data_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"Packaged example not found: {path}")
    return path
