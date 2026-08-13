"""Shim: prefer `from vectorprism.base_encoder import ...`."""
from vectorprism import base_encoder as _mod
from vectorprism.base_encoder import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
