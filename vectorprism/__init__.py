"""VectorPrism — Positional Subspace Multiplexing multi-channel retrieval."""

from __future__ import annotations

__version__ = "0.1.2"

# Public aliases kept for partner / docs imports
from vectorprism.tensor_contract import PSMTensorContract, VectorPrismTensorContract
from vectorprism.retrieval_engine import PSMRetrievalEngine, VectorPrismRetrievalEngine
from vectorprism.ingestion_adapter import (
    MultiTaskProjectionAdapter,
    VectorPrismProjectionAdapter,
)

__all__ = [
    "__version__",
    "PSMTensorContract",
    "VectorPrismTensorContract",
    "PSMRetrievalEngine",
    "VectorPrismRetrievalEngine",
    "MultiTaskProjectionAdapter",
    "VectorPrismProjectionAdapter",
]
