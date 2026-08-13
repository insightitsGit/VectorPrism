"""Shim: prefer `from vectorprism.training import ...`."""
from vectorprism import training as _mod
from vectorprism.training import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
