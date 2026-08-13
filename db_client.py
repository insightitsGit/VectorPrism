"""Shim: prefer `from vectorprism.db_client import ...`."""
from vectorprism import db_client as _mod
from vectorprism.db_client import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
