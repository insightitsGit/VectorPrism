"""
epistemic_truth.py - Phase 4 Epistemic Truth classifier.

Trains a calibrated P(is_true | text) head on labeled passages.
Hard Stage-1 filtering must stay disabled until ECE is acceptably low.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from vectorprism.base_encoder import BaseTextEncoder
from vectorprism.channel_datasets import load_truth_jsonl
from vectorprism.intrinsic_validation import expected_calibration_error


class EpistemicTruthClassifier(nn.Module):
    def __init__(self, in_dim: int = 768, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns P(true) in [0, 1], shape (batch,)."""
        return torch.sigmoid(self.net(x)).squeeze(-1)


class _TruthDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[str, int]], encoder: BaseTextEncoder):
        self.pairs = list(pairs)
        self.encoder = encoder

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        text, y = self.pairs[idx]
        emb = self.encoder.encode_one(text)
        return emb, torch.tensor(float(y), dtype=torch.float32)


@dataclass
class TruthTrainReport:
    train_loss: float
    ece: float
    accuracy: float
    n_eval: int
    hard_filter_allowed: bool
    path: str


def train_truth_classifier(
    pairs_path: str,
    encoder: BaseTextEncoder,
    out_path: str,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    ece_threshold: float = 0.05,
    device: str = "cpu",
    seed: int = 0,
) -> TruthTrainReport:
    pairs = load_truth_jsonl(pairs_path)
    if len(pairs) < 10:
        raise ValueError("Need at least 10 labeled truth examples")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_val = max(1, int(len(pairs) * val_fraction))
    val_idx = set(idx[:n_val].tolist())
    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in val_idx]
    val_pairs = [pairs[i] for i in range(len(pairs)) if i in val_idx]
    if not train_pairs:
        train_pairs, val_pairs = pairs[:-1], pairs[-1:]

    model = EpistemicTruthClassifier(in_dim=encoder.embedding_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    train_loader = DataLoader(
        _TruthDataset(train_pairs, encoder),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda xs: (
            torch.stack([a for a, _ in xs]),
            torch.stack([b for _, b in xs]),
        ),
    )

    model.train()
    last_loss = 0.0
    for epoch in range(epochs):
        running = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            n += 1
        last_loss = running / max(n, 1)
        print(f"[truth epoch {epoch}] loss={last_loss:.4f}")

    metrics = evaluate_truth_classifier(model, encoder, val_pairs, device=device)
    hard_ok = metrics["ECE"] <= ece_threshold
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "in_dim": encoder.embedding_dim,
        "ece": metrics["ECE"],
        "accuracy": metrics["accuracy"],
        "ece_threshold": ece_threshold,
        "hard_filter_allowed": hard_ok,
        "trained_at": int(time.time()),
    }
    torch.save(payload, out)
    with out.with_suffix(out.suffix + ".json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "ece": metrics["ECE"],
                "accuracy": metrics["accuracy"],
                "hard_filter_allowed": hard_ok,
                "ece_threshold": ece_threshold,
                "n_eval": metrics["n"],
            },
            f,
            indent=2,
        )

    return TruthTrainReport(
        train_loss=last_loss,
        ece=metrics["ECE"],
        accuracy=metrics["accuracy"],
        n_eval=metrics["n"],
        hard_filter_allowed=hard_ok,
        path=str(out),
    )


@torch.no_grad()
def evaluate_truth_classifier(
    model: EpistemicTruthClassifier,
    encoder: BaseTextEncoder,
    pairs: Sequence[Tuple[str, int]],
    device: str = "cpu",
) -> Dict[str, Any]:
    model.eval()
    probs, labels = [], []
    for text, y in pairs:
        emb = encoder.encode_one(text).unsqueeze(0).to(device)
        p = float(model(emb)[0].cpu())
        probs.append(p)
        labels.append(y)
    probs_a = np.asarray(probs, dtype=np.float64)
    labels_a = np.asarray(labels, dtype=np.float64)
    ece = expected_calibration_error(probs_a, labels_a)
    acc = float(((probs_a >= 0.5) == labels_a).mean()) if len(labels_a) else 0.0
    return {"ECE": ece["ECE"], "bins": ece["bins"], "accuracy": acc, "n": len(labels_a)}


def load_truth_classifier(
    path: str | Path,
    map_location: str = "cpu",
    *,
    unsafe_pickle: bool = False,
) -> Tuple[EpistemicTruthClassifier, dict]:
    """Load truth classifier; default ``weights_only=True`` (no arbitrary pickle)."""
    payload = torch.load(
        path,
        map_location=map_location,
        weights_only=not unsafe_pickle,
    )
    model = EpistemicTruthClassifier(in_dim=int(payload["in_dim"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    meta = {
        "ece": payload.get("ece"),
        "accuracy": payload.get("accuracy"),
        "hard_filter_allowed": bool(payload.get("hard_filter_allowed", False)),
        "ece_threshold": payload.get("ece_threshold", 0.05),
    }
    return model, meta


@torch.no_grad()
def score_texts(
    model: EpistemicTruthClassifier,
    encoder: BaseTextEncoder,
    texts: Sequence[str],
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    out: List[float] = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        emb = encoder.encode(batch).to(device)
        pred = model(emb).cpu().numpy().tolist()
        out.extend(pred)
    return np.asarray(out, dtype=np.float32)


def shadow_filter_report(
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Phase-4 shadow mode: log what a hard filter WOULD reject, without rejecting."""
    reject_mask = probs < threshold
    return {
        "threshold": threshold,
        "n": int(len(probs)),
        "would_reject": int(reject_mask.sum()),
        "would_reject_rate": float(reject_mask.mean()) if len(probs) else 0.0,
        "mean_prob": float(probs.mean()) if len(probs) else 0.0,
    }
