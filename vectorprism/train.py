"""
train.py - Unified VectorPrism channel trainer (Phases 1-3).

Train ONE channel at a time (spec requirement). Resume from a prior checkpoint
so previously trained heads are preserved.

Examples:
  python train.py --channel dense --data data/dense_pairs.example.jsonl --out checkpoints/vp.pt
  python train.py --channel causal --data data/causal.example.jsonl --init checkpoints/vp.pt --out checkpoints/vp.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from vectorprism.base_encoder import HashingEncoder, SentenceTransformerEncoder
from vectorprism.channel_datasets import make_channel_dataloader
from vectorprism.ingestion_adapter import MultiTaskProjectionAdapter
from vectorprism.training import RelationEmbeddingTable, run_training_loop
from vectorprism.checkpointing import save_checkpoint, load_checkpoint
from vectorprism.tensor_contract import PSMTensorContract as C
from vectorprism.identity_anchor import set_identity_anchor_from_jsonl


CHANNEL_ORDER = ["dense", "causal", "relational", "hyperbolic", "disentangled", "identity"]


def build_encoder(name: str, device: str):
    if name in {"hash", "hashing", "test"}:
        return HashingEncoder(embedding_dim=768)
    return SentenceTransformerEncoder(model_name=name, device=device)


def main(argv: Optional[list] = None) -> dict:
    p = argparse.ArgumentParser(description="VectorPrism channel trainer")
    p.add_argument("--channel", required=True, choices=CHANNEL_ORDER)
    p.add_argument("--data", required=True, help="JSONL for this channel")
    p.add_argument("--out", default="checkpoints/vectorprism.pt")
    p.add_argument("--init", default=None, help="Optional prior checkpoint to resume")
    p.add_argument("--encoder", default="hash")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--model-version", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--identity-corpus", default=None,
                   help="Optional identity JSONL to refresh v0 after identity training")
    p.add_argument("--num-relations", type=int, default=64)
    p.add_argument("--num-disentangled-classes", type=int, default=32)
    p.add_argument(
        "--unsafe-pickle",
        action="store_true",
        help="Allow legacy full-pickle checkpoint load (trusted local files only)",
    )
    args = p.parse_args(argv)

    encoder = build_encoder(args.encoder, args.device)
    if encoder.embedding_dim != 768:
        raise SystemExit(f"Adapter expects 768d base embeddings; got {encoder.embedding_dim}")

    relation_vocab = None
    enabled = {"dense": False, "relational": False, "disentangled": False,
               "hyperbolic": False, "identity": True, "causal": False}

    if args.init:
        ckpt = load_checkpoint(args.init, unsafe_pickle=bool(args.unsafe_pickle))
        adapter = ckpt["adapter"].to(args.device)
        relation_table = ckpt["relation_table"]
        relation_vocab = ckpt.get("relation_vocab")
        enabled = dict(ckpt.get("enabled_channels") or enabled)
        model_version = args.model_version if args.model_version is not None else ckpt["model_version"] + 1
        if relation_table is None:
            relation_table = RelationEmbeddingTable(args.num_relations, C.RELATIONAL.length)
    else:
        # For disentangled-first runs, class count comes from data after load below.
        adapter = MultiTaskProjectionAdapter(
            base_dim=768,
            num_disentangled_classes=args.num_disentangled_classes,
        ).to(args.device)
        relation_table = RelationEmbeddingTable(args.num_relations, C.RELATIONAL.length).to(args.device)
        model_version = args.model_version if args.model_version is not None else 1

    loader, aux = make_channel_dataloader(
        args.channel,
        args.data,
        encoder,
        batch_size=args.batch_size,
        shuffle=True,
        relation_vocab=relation_vocab,
    )

    if args.channel == "relational":
        relation_vocab = aux["relation_vocab"]
        if relation_table is None or relation_table.table.num_embeddings < relation_vocab.size:
            relation_table = RelationEmbeddingTable(relation_vocab.size, C.RELATIONAL.length).to(args.device)
        relation_table = relation_table.to(args.device)
    if args.channel == "disentangled":
        n_classes = aux.get("num_classes", args.num_disentangled_classes)
        if adapter.head_disentangled.classifier.out_features != n_classes:
            from vectorprism.losses import VIBHead
            old = adapter.head_disentangled
            adapter.head_disentangled = VIBHead(adapter.base_dim, C.DISENTANGLED.length, n_classes).to(args.device)
            adapter.head_disentangled.mu_head.load_state_dict(old.mu_head.state_dict())
            adapter.head_disentangled.logvar_head.load_state_dict(old.logvar_head.state_dict())

    print(f"[VectorPrism] train channel={args.channel} n={len(loader.dataset)} "
          f"epochs={args.epochs} encoder={args.encoder}")
    history = run_training_loop(
        adapter,
        relation_table,
        loader,
        num_epochs=args.epochs,
        lr=args.lr,
        loss_weights={args.channel: 1.0},
        device=args.device,
    )

    if args.channel == "identity" or args.identity_corpus:
        corpus = args.identity_corpus or args.data
        set_identity_anchor_from_jsonl(corpus, encoder, adapter)
        print("[VectorPrism] identity anchor v0 refreshed")

    enabled[args.channel] = True
    if args.channel != "identity":
        enabled["identity"] = True  # always keep gate channel available

    out = Path(args.out)
    save_checkpoint(
        out,
        adapter,
        model_version=model_version,
        relation_table=relation_table,
        relation_vocab=relation_vocab,
        enabled_channels=enabled,
        meta={
            "last_channel": args.channel,
            "encoder": args.encoder,
            "history": history,
        },
    )
    print(f"[VectorPrism] Saved checkpoint -> {out} (model_version={model_version})")
    return {
        "out": str(out),
        "channel": args.channel,
        "model_version": model_version,
        "history": history,
        "enabled_channels": enabled,
    }


if __name__ == "__main__":
    main()
