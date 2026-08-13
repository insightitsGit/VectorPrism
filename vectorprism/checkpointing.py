"""
checkpointing.py - Save/load VectorPrism adapter + relation table + metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List
import json

import torch
import numpy as np

from vectorprism.ingestion_adapter import MultiTaskProjectionAdapter
from vectorprism.training import RelationEmbeddingTable
from vectorprism.tensor_contract import PSMTensorContract as C
from vectorprism.channel_datasets import RelationVocab


def save_checkpoint(
    path: str | Path,
    adapter: MultiTaskProjectionAdapter,
    model_version: int,
    relation_table: Optional[RelationEmbeddingTable] = None,
    relation_vocab: Optional[RelationVocab] = None,
    enabled_channels: Optional[Dict[str, bool]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": int(model_version),
        "adapter_state": adapter.state_dict(),
        "base_dim": adapter.base_dim,
        "num_disentangled_classes": int(adapter.head_disentangled.classifier.out_features),
        "causal_dim": C.CAUSAL_TIME.length,
        "enabled_channels": enabled_channels or {"dense": True, "identity": True},
        "meta": meta or {},
    }
    if relation_table is not None:
        payload["relation_table_state"] = relation_table.state_dict()
        payload["num_relations"] = relation_table.table.num_embeddings
    if relation_vocab is not None:
        payload["relation_itos"] = relation_vocab.itos
    torch.save(payload, path)

    sidecar = path.with_suffix(path.suffix + ".json")
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_version": int(model_version),
                "base_dim": adapter.base_dim,
                "enabled_channels": payload["enabled_channels"],
                "relation_itos": payload.get("relation_itos"),
                "meta": meta or {},
            },
            f,
            indent=2,
        )


def load_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
    num_disentangled_classes: Optional[int] = None,
    *,
    unsafe_pickle: bool = False,
) -> Dict[str, Any]:
    """Load adapter checkpoint.

    Default uses ``weights_only=True`` (no arbitrary pickle). Pass
    ``unsafe_pickle=True`` only for trusted legacy local checkpoints.
    """
    path = Path(path)
    payload = torch.load(
        path,
        map_location=map_location,
        weights_only=not unsafe_pickle,
    )
    n_dis = num_disentangled_classes
    if n_dis is None:
        n_dis = int(payload.get("num_disentangled_classes", 32))
    adapter = MultiTaskProjectionAdapter(
        base_dim=int(payload.get("base_dim", 768)),
        num_disentangled_classes=n_dis,
    )
    adapter.load_state_dict(payload["adapter_state"])
    adapter.eval()

    relation_table = None
    if "relation_table_state" in payload:
        relation_table = RelationEmbeddingTable(
            num_relations=int(payload["num_relations"]),
            dim=C.RELATIONAL.length,
        )
        relation_table.load_state_dict(payload["relation_table_state"])
        relation_table.eval()

    relation_vocab = None
    if payload.get("relation_itos"):
        itos: List[str] = list(payload["relation_itos"])
        relation_vocab = RelationVocab(stoi={r: i for i, r in enumerate(itos)}, itos=itos)

    causal_matrix = adapter.causal_matrix.detach().cpu().numpy().astype(np.float32)
    return {
        "adapter": adapter,
        "relation_table": relation_table,
        "relation_vocab": relation_vocab,
        "model_version": int(payload.get("model_version", 0)),
        "causal_matrix": causal_matrix,
        "enabled_channels": payload.get("enabled_channels", {"dense": True, "identity": True}),
        "meta": payload.get("meta", {}),
    }
