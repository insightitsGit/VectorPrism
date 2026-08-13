"""Shim: prefer `from vectorprism.jsonl_utils import ...`."""
from vectorprism import jsonl_utils as _mod
from vectorprism.jsonl_utils import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
