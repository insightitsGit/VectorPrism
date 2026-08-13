"""Shim: prefer `from vectorprism.structure_index import ...`."""
from vectorprism import structure_index as _mod
from vectorprism.structure_index import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
