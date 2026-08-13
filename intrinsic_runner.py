"""Shim: prefer `from vectorprism.intrinsic_runner import ...`."""
from vectorprism import intrinsic_runner as _mod
from vectorprism.intrinsic_runner import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
