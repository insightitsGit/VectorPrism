"""Shim: prefer `from vectorprism.tensor_contract import ...`."""
from vectorprism import tensor_contract as _mod
from vectorprism.tensor_contract import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
