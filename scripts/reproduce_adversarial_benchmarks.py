#!/usr/bin/env python3
"""Reproduce adversarial benchmark scorecards from pack data.

Checkpoints are gitignored (*.pt). This script trains them then scores.
For a machine that already has checkpoints, pass --skip-train.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos" / "finance_demo"
CKPT = ROOT / "checkpoints"


def _run(cmd: list[str]) -> None:
    print("\n+", " ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=str(ROOT))
    if code != 0:
        raise SystemExit(code)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reproduce VectorPrism adversarial benchmarks")
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--with-post", action="store_true")
    ap.add_argument("--with-robustness", action="store_true")
    args = ap.parse_args(argv)

    multi = CKPT / "finance_hard_adversarial_multi.pt"
    dense = CKPT / "finance_hard_adversarial.pt"
    causal = CKPT / "finance_hard_adversarial_causal.pt"

    if args.skip_train:
        if not multi.exists():
            raise SystemExit(
                "--skip-train requires checkpoints/finance_hard_adversarial_multi.pt. "
                "Checkpoints are gitignored — omit --skip-train to retrain from pack data."
            )
    else:
        # Dense adversarial checkpoint
        if not dense.exists():
            _run(
                [
                    sys.executable,
                    str(DEMO / "run_hard_eval.py"),
                    "--mode",
                    "isolated",
                    "--pack",
                    "adversarial",
                    "--encoder",
                    args.encoder,
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                ]
            )
        # Causal on top of dense
        if not causal.exists():
            _run(
                [
                    sys.executable,
                    str(DEMO / "run_causal_recovery.py"),
                    "--pack",
                    "adversarial",
                    "--encoder",
                    args.encoder,
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                ]
            )
        # Hyp+rel multi
        _run(
            [
                sys.executable,
                str(DEMO / "run_multichannel_recovery.py"),
                "--encoder",
                args.encoder,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
            ]
        )

    if args.skip_train:
        _run(
            [
                sys.executable,
                str(DEMO / "run_multichannel_recovery.py"),
                "--skip-train",
                "--encoder",
                args.encoder,
            ]
        )

    if args.with_post:
        _run([sys.executable, str(DEMO / "run_post_validation.py"), "--encoder", args.encoder])
    if args.with_robustness:
        _run(
            [sys.executable, str(DEMO / "run_robustness_validation.py"), "--encoder", args.encoder]
        )

    print("\n[reproduce] done. See demos/finance_demo/results/hard_eval/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
