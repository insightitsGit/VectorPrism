"""Shim: prefer `python -m vectorprism train`."""
from vectorprism import train as _mod
from vectorprism.train import *  # noqa: F401,F403
from vectorprism.train import main

globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(main() or 0)
