"""Backward-compatible Phase-1 entrypoint. Prefer: python train.py --channel dense ... """

from __future__ import annotations

import argparse
from vectorprism.train import main as train_main


def main() -> None:
    p = argparse.ArgumentParser(description="VectorPrism Phase-1 dense training (wrapper)")
    p.add_argument("--pairs", required=True)
    p.add_argument("--out", default="checkpoints/dense_v1.pt")
    p.add_argument("--encoder", default="hash")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--model-version", type=int, default=1)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    train_main([
        "--channel", "dense",
        "--data", args.pairs,
        "--out", args.out,
        "--encoder", args.encoder,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--model-version", str(args.model_version),
        "--device", args.device,
    ])


if __name__ == "__main__":
    main()
