"""Shim: prefer `from vectorprism.checkpointing import ...`."""
from vectorprism import checkpointing as _mod
from vectorprism.checkpointing import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
