"""Shim: prefer `from vectorprism.identity_anchor import ...`."""
from vectorprism import identity_anchor as _mod
from vectorprism.identity_anchor import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
