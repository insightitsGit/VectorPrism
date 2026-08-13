"""Shim: prefer `from vectorprism.intrinsic_validation import ...`."""
from vectorprism import intrinsic_validation as _mod
from vectorprism.intrinsic_validation import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
