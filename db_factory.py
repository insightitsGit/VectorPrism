"""Shim: prefer `from vectorprism.db_factory import ...`."""
from vectorprism import db_factory as _mod
from vectorprism.db_factory import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
