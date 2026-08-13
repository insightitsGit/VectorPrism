"""
ingest_pipeline.py - Text → frozen encoder → VectorPrism 1024d → vector DB.

Production upsert path (Phase 5). Epistemic truth defaults to 1.0 unless a
trained classifier is attached or the document supplies an explicit score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Dict, Any
import time

import numpy as np
import torch

from vectorprism.base_encoder import BaseTextEncoder
from vectorprism.ingestion_adapter import MultiTaskProjectionAdapter
from vectorprism.tensor_contract import PSMTensorContract as C
from vectorprism.db_client import VectorDBClient
from vectorprism.losses import anchor_distance_score


@dataclass
class IngestDocument:
    document_id: str
    chunk_text: str
    epistemic_truth: Optional[float] = None  # None => use classifier or default 1.0
    timestamp: Optional[int] = None


class VectorPrismIngestPipeline:
    def __init__(
        self,
        encoder: BaseTextEncoder,
        adapter: MultiTaskProjectionAdapter,
        db: VectorDBClient,
        model_version: int = 1,
        enabled_channels: Optional[Dict[str, bool]] = None,
        truth_classifier: Optional[torch.nn.Module] = None,
        default_truth: float = 1.0,
    ):
        if encoder.embedding_dim != adapter.base_dim:
            raise ValueError(
                f"encoder dim {encoder.embedding_dim} != adapter base_dim {adapter.base_dim}"
            )
        self.encoder = encoder
        self.adapter = adapter
        self.db = db
        self.model_version = int(model_version)
        self.truth_classifier = truth_classifier
        self.default_truth = float(default_truth)
        self.enabled_channels = enabled_channels or {
            "dense": True,
            "relational": False,
            "disentangled": False,
            "hyperbolic": False,
            "identity": True,
            "causal": False,
        }
        self.adapter.eval()
        if self.truth_classifier is not None:
            self.truth_classifier.eval()

    @torch.no_grad()
    def _truth_scores(self, texts: Sequence[str], base: torch.Tensor,
                      explicit: Optional[Sequence[Optional[float]]]) -> np.ndarray:
        n = len(texts)
        scores = np.full(n, self.default_truth, dtype=np.float32)
        if explicit is not None:
            for i, v in enumerate(explicit):
                if v is not None:
                    scores[i] = float(np.clip(v, 0.0, 1.0))
        need_cls = [i for i in range(n) if explicit is None or explicit[i] is None]
        if need_cls and self.truth_classifier is not None:
            pred = self.truth_classifier(base[need_cls]).detach().cpu().numpy()
            for i, p in zip(need_cls, pred):
                scores[i] = float(np.clip(p, 0.0, 1.0))
        return scores

    @torch.no_grad()
    def encode_texts(
        self,
        texts: Sequence[str],
        epistemic_truth: float | Sequence[Optional[float]] | None = None,
        timestamp: Optional[int] = None,
    ) -> np.ndarray:
        """Returns (N, 1024) float32 tensors ready for storage / query."""
        base = self.encoder.encode(texts)
        n = base.shape[0]
        ts = int(timestamp if timestamp is not None else time.time())

        if epistemic_truth is None:
            explicit = [None] * n
        elif isinstance(epistemic_truth, (float, int)):
            explicit = [float(epistemic_truth)] * n
        else:
            explicit = list(epistemic_truth)
        truths = self._truth_scores(texts, base, explicit)

        headers = []
        for i in range(n):
            headers.append(
                C.pack_header(
                    bitmask=C.default_channel_bitmask(self.enabled_channels),
                    epistemic_truth=float(truths[i]),
                    anchor_distance=0.0,
                    timestamp=ts,
                    model_version=self.model_version,
                )
            )
        header_t = torch.from_numpy(np.stack(headers)).to(dtype=torch.float32)
        tensor_1024d, raw = self.adapter(base, header_t)

        dists = anchor_distance_score(raw["identity"], self.adapter.identity_anchor_v0)
        out = tensor_1024d.cpu().numpy().astype(np.float32)
        for i in range(n):
            out[i, C.HDR_ANCHOR.start] = float(dists[i].item())
            out[i, C.HDR_TRUTH.start] = float(truths[i])
        return out

    def upsert_documents(self, docs: Iterable[IngestDocument], batch_size: int = 32) -> int:
        batch: List[IngestDocument] = []
        count = 0
        for doc in docs:
            batch.append(doc)
            if len(batch) >= batch_size:
                count += self._upsert_batch(batch)
                batch = []
        if batch:
            count += self._upsert_batch(batch)
        return count

    def _upsert_batch(self, docs: List[IngestDocument]) -> int:
        texts = [d.chunk_text for d in docs]
        explicit = [d.epistemic_truth for d in docs]
        # Use first timestamp if uniform else encode individually for stamps.
        stamps = {d.timestamp for d in docs}
        if len(stamps) == 1:
            tensors = self.encode_texts(
                texts,
                epistemic_truth=explicit,
                timestamp=docs[0].timestamp,
            )
        else:
            tensors = np.stack([
                self.encode_texts(
                    [d.chunk_text],
                    epistemic_truth=[d.epistemic_truth],
                    timestamp=d.timestamp,
                )[0]
                for d in docs
            ], axis=0)

        for doc, tensor in zip(docs, tensors):
            header = C.unpack_header(tensor)
            meta: Dict[str, Any] = {
                "epistemic_truth": header["epistemic_truth"],
                "anchor_dist": header["anchor_distance"],
                "valid_timestamp": header["timestamp"],
                "model_version": header["model_version"],
            }
            self.db.upsert(doc.document_id, doc.chunk_text, tensor, meta)
        return len(docs)

    def encode_query(self, query_text: str) -> np.ndarray:
        return self.encode_texts([query_text])[0]
