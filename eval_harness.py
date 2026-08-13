"""Shim: prefer `from vectorprism.eval_harness import ...`."""
from vectorprism import eval_harness as _mod
from vectorprism.eval_harness import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
