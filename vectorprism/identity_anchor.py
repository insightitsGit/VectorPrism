"""
identity_anchor.py - Compute and install the frozen in-domain identity anchor v0.
"""

from __future__ import annotations

from typing import Sequence

import torch

from vectorprism.base_encoder import BaseTextEncoder
from vectorprism.ingestion_adapter import MultiTaskProjectionAdapter
from vectorprism.channel_datasets import load_identity_jsonl


@torch.no_grad()
def compute_identity_anchor(
    encoder: BaseTextEncoder,
    adapter: MultiTaskProjectionAdapter,
    texts: Sequence[str],
    batch_size: int = 64,
) -> torch.Tensor:
    """Mean of identity-head outputs over an in-domain reference corpus."""
    if not texts:
        raise ValueError("Need at least one in-domain text to compute identity anchor")
    adapter.eval()
    chunks = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        base = encoder.encode(batch)
        header = torch.zeros(base.shape[0], 16)
        _, raw = adapter(base, header)
        chunks.append(raw["identity"])
    mean = torch.cat(chunks, dim=0).mean(dim=0)
    return mean


def set_identity_anchor_from_jsonl(
    path: str,
    encoder: BaseTextEncoder,
    adapter: MultiTaskProjectionAdapter,
) -> torch.Tensor:
    texts = load_identity_jsonl(path)
    v0 = compute_identity_anchor(encoder, adapter, texts)
    adapter.set_identity_anchor(v0)
    return v0
