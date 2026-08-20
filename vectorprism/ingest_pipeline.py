"""
ingest_pipeline.py - Text → frozen encoder → VectorPrism 1024d → vector DB.

Production upsert path (Phase 5). Epistemic truth defaults to 1.0 unless a
trained classifier is attached or the document supplies an explicit score.

Integration contract (read this if you use Onyx / LangChain / another RAG stack):
  - ``chunk_text`` is *your* chunking. VectorPrism does not split documents.
  - Every upsert goes through the frozen base encoder + MultiTaskProjectionAdapter
    from a VectorPrism checkpoint → one contiguous **1024d** PSM tensor.
  - Storing Onyx / OpenAI / other foreign embeddings in ``tensor_1024d`` is
    unsupported. Stage-1 expects the 368d dense-core slice of that layout;
    Stage-2 scores the other channel slices. Foreign vectors will not recover
    the multi-channel benchmark results.
  - See ``vectorprism.encode_guards.INTEGRATION_BANNER``.
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
from vectorprism.temporal_types import coerce_unix_epoch_seconds


@dataclass
class IngestDocument:
    """One pre-chunked text unit. You own chunking; VectorPrism owns encoding.

    ``chunk_text`` may come from Onyx, LangChain, or any splitter. Do **not**
    pass an external embedding here — the pipeline always re-encodes text with
    the VectorPrism adapter into a 1024d multi-channel tensor.

    Bitemporal (exact unix seconds, not float embeddings):
      - ``timestamp`` / valid_from — when the fact became true
      - ``valid_to`` — exclusive end (None = still in force)
      - ``transaction_time`` — when this version was recorded in your system
    """

    document_id: str
    chunk_text: str
    epistemic_truth: Optional[float] = None  # None => use classifier or default 1.0
    timestamp: Optional[int] = None  # valid_from
    valid_to: Optional[int] = None
    transaction_time: Optional[int] = None


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
        transaction_time: Optional[int] = None,
    ) -> np.ndarray:
        """Returns (N, 1024) float32 tensors ready for storage / query."""
        base = self.encoder.encode(texts)
        n = base.shape[0]
        if timestamp is None:
            ts = int(time.time())
        else:
            ts = coerce_unix_epoch_seconds(timestamp, field_name="timestamp")
            assert ts is not None
        tx = None
        if transaction_time is not None:
            tx = coerce_unix_epoch_seconds(
                transaction_time, field_name="transaction_time"
            )

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
                    transaction_time=tx,
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
        norm_ts = [
            coerce_unix_epoch_seconds(
                d.timestamp, field_name="timestamp", allow_none=True, default=None
            )
            for d in docs
        ]
        norm_tx = [
            coerce_unix_epoch_seconds(
                d.transaction_time,
                field_name="transaction_time",
                allow_none=True,
                default=None,
            )
            for d in docs
        ]
        norm_to = [
            coerce_unix_epoch_seconds(
                d.valid_to, field_name="valid_to", allow_none=True, default=None
            )
            for d in docs
        ]
        for i, (vf, vt) in enumerate(zip(norm_ts, norm_to)):
            if vf is not None and vt is not None and vt <= vf:
                raise ValueError(
                    f"document {docs[i].document_id}: valid_to ({vt}) must be > "
                    f"valid_from/timestamp ({vf}) (half-open interval)"
                )

        # Encode one-by-one when temporal stamps differ across the batch
        keyset = set(zip(norm_ts, norm_tx))
        if len(keyset) == 1:
            ts0, tx0 = next(iter(keyset))
            tensors = self.encode_texts(
                texts,
                epistemic_truth=explicit,
                timestamp=ts0,
                transaction_time=tx0,
            )
        else:
            tensors = np.stack(
                [
                    self.encode_texts(
                        [d.chunk_text],
                        epistemic_truth=[d.epistemic_truth],
                        timestamp=ts,
                        transaction_time=tx,
                    )[0]
                    for d, ts, tx in zip(docs, norm_ts, norm_tx)
                ],
                axis=0,
            )

        for doc, tensor, vt in zip(docs, tensors, norm_to):
            header = C.unpack_header(tensor)
            meta: Dict[str, Any] = {
                "epistemic_truth": header["epistemic_truth"],
                "anchor_dist": header["anchor_distance"],
                "valid_timestamp": int(header["timestamp"]),
                "valid_to_timestamp": None if vt is None else int(vt),
                "transaction_timestamp": (
                    None
                    if header.get("transaction_time") is None
                    else int(header["transaction_time"])
                ),
                "model_version": int(header["model_version"]),
            }
            self.db.upsert(doc.document_id, doc.chunk_text, tensor, meta)
        return len(docs)

    def encode_query(self, query_text: str) -> np.ndarray:
        return self.encode_texts([query_text])[0]
