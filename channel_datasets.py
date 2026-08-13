"""Shim: prefer `from vectorprism.channel_datasets import ...`."""
from vectorprism import channel_datasets as _mod
from vectorprism.channel_datasets import *  # noqa: F401,F403

# re-export private helpers (import * skips leading underscore)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
