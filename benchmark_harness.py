"""Shim: prefer `from vectorprism.benchmark_harness import ...`."""
from vectorprism import benchmark_harness as _mod
from vectorprism.benchmark_harness import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
