"""
training.py - End-to-end training loop for MultiTaskProjectionAdapter.

THIS FILE DOES NOT GENERATE TRAINING DATA. It defines the exact schema
your data pipeline must produce (`PSMBatch`), and a loop that will
correctly train every head IF you supply real examples in that schema.
Running this on random tensors (as the old benchmark_harness.py did)
will execute without error and produce a completely meaningless model —
that's not a code bug, it's what "no labels" always means for a
supervised/contrastive objective. There is no way around needing labeled
data for the relational, disentangled, hyperbolic, identity, and causal
heads; only the dense head can bootstrap from generic (query, passage)
pairs, which are comparatively easy to source (e.g. from search-click
logs or existing sentence-pair datasets).
"""

from dataclasses import dataclass
from typing import Optional
import time

import torch
from torch.utils.data import DataLoader, Dataset

from vectorprism.tensor_contract import PSMTensorContract as C
from vectorprism.ingestion_adapter import MultiTaskProjectionAdapter
from vectorprism.losses import (
    info_nce_loss,
    transe_margin_loss,
    vib_loss,
    poincare_negative_sampling_loss,
    center_loss,
    causal_asymmetric_loss,
)


@dataclass
class PSMBatch:
    """
    Exact fields required per training step. `base_emb_*` fields are
    768d outputs of your FROZEN base encoder (run it upstream/offline;
    do not re-run it inside the training loop unless you're fine-tuning it).
    Any field can be None for a given batch if that channel has no data
    this step (the loop skips terms with None and only backprops the
    channels that had data — this is how you train channels on
    different, unaligned datasets without needing one mega-dataset that
    has every label type at once).
    """
    # Dense (InfoNCE): query/positive passage pairs
    dense_anchor: Optional[torch.Tensor] = None      # (B, 768)
    dense_positive: Optional[torch.Tensor] = None     # (B, 768)

    # Relational (TransE): KG-style triples + a relation id + one negative object
    rel_subject: Optional[torch.Tensor] = None         # (B, 768)
    rel_object: Optional[torch.Tensor] = None           # (B, 768)
    rel_neg_object: Optional[torch.Tensor] = None        # (B, 768)
    rel_relation_id: Optional[torch.Tensor] = None        # (B,) long, index into relation embedding table

    # Disentangled (VIB): input + the label you want Z to predict
    dis_input: Optional[torch.Tensor] = None           # (B, 768)
    dis_label: Optional[torch.Tensor] = None            # (B,) long

    # Hyperbolic (Poincare): taxonomy edges + k negatives per anchor
    hyp_parent: Optional[torch.Tensor] = None           # (B, 768)
    hyp_child: Optional[torch.Tensor] = None              # (B, 768)
    hyp_negatives: Optional[torch.Tensor] = None           # (B, k, 768) -> encoded upstream, then projected per-item

    # Identity (center loss): in-domain reference examples
    identity_in_domain: Optional[torch.Tensor] = None    # (B, 768)

    # Causal: known-ordered (earlier, later) event pairs
    causal_earlier: Optional[torch.Tensor] = None        # (B, 768)
    causal_later: Optional[torch.Tensor] = None           # (B, 768)

    # Header (metadata, not learned — see ingestion_adapter.forward docstring)
    header: torch.Tensor = None                          # (B, 16)


class RelationEmbeddingTable(torch.nn.Module):
    """Small closed vocabulary of relation types (e.g. 'causes', 'is_a',
    'part_of', 'contradicts', ...). TransE needs this — a per-passage
    linear projection cannot stand in for a relation vocabulary."""
    def __init__(self, num_relations: int, dim: int):
        super().__init__()
        self.table = torch.nn.Embedding(num_relations, dim)

    def forward(self, relation_ids: torch.Tensor) -> torch.Tensor:
        return self.table(relation_ids)


def train_step(
    model: MultiTaskProjectionAdapter,
    relation_table: RelationEmbeddingTable,
    batch: PSMBatch,
    loss_weights: dict,
) -> dict:
    """Computes only the loss terms for which this batch has data.
    Returns a dict of scalar losses (for logging) plus 'total' (for backward)."""
    losses = {}

    if batch.dense_anchor is not None:
        _, raw_a = model(batch.dense_anchor, torch.zeros(batch.dense_anchor.shape[0], 16))
        _, raw_p = model(batch.dense_positive, torch.zeros(batch.dense_positive.shape[0], 16))
        losses["dense"] = info_nce_loss(raw_a["dense"], raw_p["dense"])

    if batch.rel_subject is not None:
        _, raw_s = model(batch.rel_subject, torch.zeros(batch.rel_subject.shape[0], 16))
        _, raw_o = model(batch.rel_object, torch.zeros(batch.rel_object.shape[0], 16))
        _, raw_no = model(batch.rel_neg_object, torch.zeros(batch.rel_neg_object.shape[0], 16))
        rel_emb = relation_table(batch.rel_relation_id)
        losses["relational"] = transe_margin_loss(
            raw_s["relational"], rel_emb, raw_o["relational"], raw_no["relational"]
        )

    if batch.dis_input is not None:
        _, raw_d = model(batch.dis_input, torch.zeros(batch.dis_input.shape[0], 16))
        logits = model.head_disentangled.classifier(raw_d["disentangled_z"])
        losses["disentangled"] = vib_loss(
            logits, batch.dis_label, raw_d["disentangled_mu"], raw_d["disentangled_logvar"]
        )

    if batch.hyp_parent is not None:
        _, raw_par = model(batch.hyp_parent, torch.zeros(batch.hyp_parent.shape[0], 16))
        _, raw_chi = model(batch.hyp_child, torch.zeros(batch.hyp_child.shape[0], 16))
        B, k, _ = batch.hyp_negatives.shape
        flat_neg = batch.hyp_negatives.reshape(B * k, -1)
        _, raw_neg = model(flat_neg, torch.zeros(B * k, 16))
        neg_hyp = raw_neg["hyperbolic"].reshape(B, k, -1)
        losses["hyperbolic"] = poincare_negative_sampling_loss(
            raw_par["hyperbolic"], raw_chi["hyperbolic"], neg_hyp
        )

    if batch.identity_in_domain is not None:
        _, raw_id = model(batch.identity_in_domain, torch.zeros(batch.identity_in_domain.shape[0], 16))
        losses["identity"] = center_loss(raw_id["identity"], model.identity_anchor_v0)

    if batch.causal_earlier is not None:
        _, raw_e = model(batch.causal_earlier, torch.zeros(batch.causal_earlier.shape[0], 16))
        _, raw_l = model(batch.causal_later, torch.zeros(batch.causal_later.shape[0], 16))
        losses["causal"] = causal_asymmetric_loss(raw_e["causal"], raw_l["causal"], model.causal_matrix)

    if not losses:
        return {"total": torch.tensor(0.0)}

    total = sum(loss_weights.get(name, 1.0) * val for name, val in losses.items())
    losses["total"] = total
    return losses


def _batch_to_device(batch: PSMBatch, device: str) -> PSMBatch:
    def move(x):
        return x.to(device) if x is not None else None
    return PSMBatch(
        dense_anchor=move(batch.dense_anchor),
        dense_positive=move(batch.dense_positive),
        rel_subject=move(batch.rel_subject),
        rel_object=move(batch.rel_object),
        rel_neg_object=move(batch.rel_neg_object),
        rel_relation_id=move(batch.rel_relation_id),
        dis_input=move(batch.dis_input),
        dis_label=move(batch.dis_label),
        hyp_parent=move(batch.hyp_parent),
        hyp_child=move(batch.hyp_child),
        hyp_negatives=move(batch.hyp_negatives),
        identity_in_domain=move(batch.identity_in_domain),
        causal_earlier=move(batch.causal_earlier),
        causal_later=move(batch.causal_later),
        header=move(batch.header) if batch.header is not None else None,
    )


def run_training_loop(
    model: MultiTaskProjectionAdapter,
    relation_table: RelationEmbeddingTable,
    dataloader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-4,
    loss_weights: Optional[dict] = None,
    device: str = "cpu",
) -> list:
    """
    dataloader must yield PSMBatch objects. Returns per-epoch average loss dicts.
    """
    loss_weights = loss_weights or {}
    model.to(device)
    relation_table.to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(relation_table.parameters()), lr=lr
    )
    history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        running = {}
        n_batches = 0
        for batch in dataloader:
            batch = _batch_to_device(batch, device)
            optimizer.zero_grad()
            step_losses = train_step(model, relation_table, batch, loss_weights)
            if step_losses["total"].requires_grad:
                step_losses["total"].backward()
                optimizer.step()
            for k, v in step_losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        history.append(avg)
        print(f"[epoch {epoch}] ({time.time()-epoch_start:.1f}s) " +
              " ".join(f"{k}={v:.4f}" for k, v in avg.items()))
    return history
