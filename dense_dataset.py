"""Shim: prefer `from vectorprism.dense_dataset import ...`."""
from vectorprism import dense_dataset as _mod
from vectorprism.dense_dataset import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
