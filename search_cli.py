"""Shim: prefer `from vectorprism.search_cli import ...`."""
from vectorprism import search_cli as _mod
from vectorprism.search_cli import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
