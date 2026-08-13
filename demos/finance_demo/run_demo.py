"""
Finance client demo — near-real Phase-1 walkthrough.

Steps:
  1) Ensure corpus JSONL exists (generate if missing)
  2) Train dense channel
  3) Evaluate vs dense-cosine baseline
  4) Run sample enterprise queries (KYC / wire fraud / NAV / margin)

Usage:
  python demos/finance_demo/run_demo.py
  python demos/finance_demo/run_demo.py --encoder sentence-transformers/all-mpnet-base-v2
  python demos/finance_demo/run_demo.py --encoder hash --epochs 2   # offline plumbing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


SAMPLE_QUERIES = [
    "Why are international wires delayed after callback voids?",
    "When must a maintenance margin call be met before liquidation?",
    "What time is official fund NAV struck and who approves large corrections?",
    "Do large international wires require callback verification?",
    "What happens if weekly fund outflows exceed fifteen percent of AUM?",
]


def _run(cmd: list) -> int:
    print("\n+", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def ensure_corpus() -> None:
    needed = ["documents.jsonl", "dense_pairs.jsonl", "eval.jsonl"]
    if not all((DEMO / n).exists() for n in needed):
        print("[finance-demo] Generating corpus...")
        _run([sys.executable, str(DEMO / "generate_corpus.py")])


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(description="VectorPrism finance client demo")
    p.add_argument("--encoder", default="hash",
                   help="hash (offline) or sentence-transformers/all-mpnet-base-v2 (near-real)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default=str(ROOT / "checkpoints" / "finance_demo.pt"))
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args(argv)

    ensure_corpus()
    pairs = DEMO / "dense_pairs.jsonl"
    docs = DEMO / "documents.jsonl"
    ev = DEMO / "eval.jsonl"
    report = DEMO / "reports" / "phase1_eval.json"
    report.parent.mkdir(parents=True, exist_ok=True)

    n_pairs = sum(1 for _ in pairs.open(encoding="utf-8") if _.strip())
    n_docs = sum(1 for _ in docs.open(encoding="utf-8") if _.strip())
    n_eval = sum(1 for _ in ev.open(encoding="utf-8") if _.strip())
    print(f"[finance-demo] corpus: {n_docs} docs | {n_pairs} dense pairs | {n_eval} eval queries")
    print(f"[finance-demo] encoder={args.encoder}")

    if not args.skip_train:
        code = _run([
            sys.executable, "train.py",
            "--channel", "dense",
            "--data", str(pairs),
            "--out", args.out,
            "--encoder", args.encoder,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
        ])
        if code != 0:
            return code

    code = _run([
        sys.executable, "vectorprism.py", "eval",
        "--checkpoint", args.out,
        "--documents", str(docs),
        "--eval", str(ev),
        "--encoder", args.encoder,
        "--out", str(report),
    ])
    # exit 2 = DoD gate miss — still continue demo search
    if code not in (0, 2):
        return code

    if report.exists():
        print("\n[finance-demo] Phase-1 eval report:")
        print(report.read_text(encoding="utf-8"))

    # Interactive-style sample searches via in-process engine
    from base_encoder import HashingEncoder, SentenceTransformerEncoder
    from checkpointing import load_checkpoint
    from eval_runner import InMemoryCorpusDB
    from ingest_pipeline import VectorPrismIngestPipeline, IngestDocument
    from channel_datasets import load_documents_jsonl
    from retrieval_engine import PSMRetrievalEngine

    if args.encoder in {"hash", "hashing", "test"}:
        encoder = HashingEncoder(768)
    else:
        encoder = SentenceTransformerEncoder(args.encoder)

    ckpt = load_checkpoint(args.out)
    db = InMemoryCorpusDB()
    pipe = VectorPrismIngestPipeline(
        encoder, ckpt["adapter"], db,
        model_version=ckpt["model_version"],
        enabled_channels=ckpt.get("enabled_channels"),
    )
    pipe.upsert_documents([
        IngestDocument(str(d["document_id"]), str(d["chunk_text"]))
        for d in load_documents_jsonl(docs)
    ])
    engine = PSMRetrievalEngine(db, causal_matrix=ckpt["causal_matrix"])

    print("\n[finance-demo] Sample client queries:")
    for q in SAMPLE_QUERIES:
        qv = pipe.encode_query(q)
        hits = engine.search(qv, q, top_k=3)
        print(f"\nQ: {q}")
        for i, h in enumerate(hits, 1):
            print(f"  {i}. {h['document_id']}  score={h['final_score']:.3f}")
            print(f"     {h.get('chunk_text', '')[:140]}")

    print("\n[finance-demo] Done.")
    if args.encoder in {"hash", "hashing", "test"}:
        print("NOTE: hash encoder is for offline plumbing. For a client-facing demo, re-run with:")
        print("  python demos/finance_demo/run_demo.py --encoder sentence-transformers/all-mpnet-base-v2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
