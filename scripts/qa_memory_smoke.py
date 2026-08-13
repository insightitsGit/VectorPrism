"""BUG-001 smoke: two-step memory ingest then search via CLI helpers."""
from __future__ import annotations

from pathlib import Path

from vectorprism.checkpointing import save_checkpoint
from vectorprism.ingestion_adapter import MultiTaskProjectionAdapter
from vectorprism.ingest_cli import main as ingest_main
from vectorprism.search_cli import main as search_main

ROOT = Path(__file__).resolve().parents[1]
ckpt = ROOT / "checkpoints" / "dense_smoke.pt"
store = ROOT / "checkpoints" / "qa_mem.npz"
docs = ROOT / "data" / "documents.example.jsonl"
ckpt.parent.mkdir(parents=True, exist_ok=True)
save_checkpoint(ckpt, MultiTaskProjectionAdapter(768), 1)
n = ingest_main(
    [
        "--checkpoint",
        str(ckpt),
        "--documents",
        str(docs),
        "--encoder",
        "hash",
        "--backend",
        "memory",
        "--store",
        str(store),
    ]
)
print("upserted", n)
hits = search_main(
    [
        "--checkpoint",
        str(ckpt),
        "--query",
        "VectorPrism uses a 1024-dimensional tensor",
        "--encoder",
        "hash",
        "--backend",
        "memory",
        "--store",
        str(store),
        "--top-k",
        "3",
    ]
)
assert hits, "expected ranked hits"
print("ok", len(hits), "hits")
