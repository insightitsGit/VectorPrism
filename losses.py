"""Shim: prefer `from vectorprism.losses import ...`."""
from vectorprism import losses as _mod
from vectorprism.losses import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
