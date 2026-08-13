"""Shim: prefer `from vectorprism.live_benchmark import ...`."""
from vectorprism import live_benchmark as _mod
from vectorprism.live_benchmark import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
