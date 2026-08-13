"""
base_encoder.py - Frozen base text encoder interface for VectorPrism.

The MultiTaskProjectionAdapter consumes fixed 768d embeddings. This module
is the real encoder boundary — no fake embeddings in the production path.

Implementations:
  - HashingEncoder: deterministic local fallback for plumbing/tests only
  - SentenceTransformerEncoder: production path (e.g. nomic-embed-text)
"""

from abc import ABC, abstractmethod
from typing import List, Sequence
import hashlib

import numpy as np
import torch
import torch.nn.functional as F


class BaseTextEncoder(ABC):
    """Frozen encoder producing (N, embedding_dim) float32 torch tensors."""

    embedding_dim: int

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        ...

    def encode_one(self, text: str) -> torch.Tensor:
        return self.encode([text])[0]


class HashingEncoder(BaseTextEncoder):
    """Deterministic bag-of-hashed-ngrams encoder for tests / offline plumbing.

    NOT a semantic model. Do not use for quality evaluation or production
    retrieval claims. Useful so the rest of the pipeline can run without
    downloading a large sentence-transformer checkpoint.
    """

    def __init__(self, embedding_dim: int = 768, seed: int = 0):
        self.embedding_dim = embedding_dim
        self.seed = seed

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        rows = []
        for text in texts:
            vec = np.zeros(self.embedding_dim, dtype=np.float32)
            tokens = text.lower().split()
            if not tokens:
                tokens = [""]
            for tok in tokens:
                digest = hashlib.sha256(f"{self.seed}:{tok}".encode("utf-8")).digest()
                # Map token into a few dimensions for a sparse-ish signature
                for i in range(4):
                    idx = int.from_bytes(digest[i * 4:(i + 1) * 4], "little") % self.embedding_dim
                    sign = 1.0 if digest[16 + i] % 2 == 0 else -1.0
                    vec[idx] += sign
            n = np.linalg.norm(vec)
            if n > 0:
                vec /= n
            rows.append(vec)
        return torch.from_numpy(np.stack(rows, axis=0))


class SentenceTransformerEncoder(BaseTextEncoder):
    """Production frozen encoder via sentence-transformers.

    Example:
        enc = SentenceTransformerEncoder("nomic-ai/nomic-embed-text-v1.5")
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEncoder. "
                "Install with: pip install sentence-transformers"
            ) from e
        self.model_name = model_name
        self.device = device
        self._model = SentenceTransformer(model_name, device=device)
        probe = self._model.encode(["probe"], convert_to_numpy=True)
        self.embedding_dim = int(probe.shape[-1])

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        arr = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)
