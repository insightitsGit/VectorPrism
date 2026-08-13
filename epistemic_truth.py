"""Shim: prefer `from vectorprism.epistemic_truth import ...`."""
from vectorprism import epistemic_truth as _mod
from vectorprism.epistemic_truth import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
