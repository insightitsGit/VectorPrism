"""
vectorprism.py - Master CLI for Phases 0-6.

Subcommands:
  phase0          install-check + pytest
  train           train one channel (Phases 1-3)
  eval            Phase-1 retrieval eval vs dense baseline
  ablation        system-level channel ablation
  intrinsic       per-channel intrinsic DoD
  train-truth     Phase-4 epistemic truth classifier
  ingest          Phase-5 upsert
  search          Phase-5 query
  live-benchmark  Phase-5 e2e latency
  reingest        Phase-6 versioned re-encode
  run-all-smoke   end-to-end plumbing on example JSONL (not a quality claim)
  finance-demo    near-real finance client Phase-1 demo
  finance-pg      finance demo ingest+search on Postgres/pgvector
  hard-eval       isolated/mixed dense hard-eval baseline
  causal-recovery train causal + recovery scorecard on dense misses
  pilot-check     validate partner install (imports + ingest smoke)
  version         print package version
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    candidate = here.parent
    if (candidate / "pyproject.toml").exists() or (candidate / "demos").exists():
        return candidate
    return Path.cwd()


def _example_data_dir() -> Path:
    from vectorprism.paths import example_data_dir

    return example_data_dir()


ROOT = _repo_root()


def _run(cmd: list) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def cmd_phase0(_args) -> int:
    code = _run([sys.executable, "-m", "pytest", "test_psm.py", "test_phases.py", "-v"])
    return code


def cmd_version(_args) -> int:
    try:
        from importlib.metadata import version

        ver = version("vectorprism")
    except Exception:
        ver = "0.1.3 (editable / source tree)"
    print(f"vectorprism {ver}")
    print(f"root={ROOT}")
    return 0


def cmd_pilot_check(_args) -> int:
    """Validate that a partner install can import core modules and run a tiny encode."""
    errors: list[str] = []
    print("VectorPrism pilot-check", flush=True)
    try:
        from vectorprism.tensor_contract import PSMTensorContract
        from vectorprism.causal_graph import CausalDocGraph
        from vectorprism.structure_index import TaxonomyGraph, RelationalAttrIndex

        _ = (PSMTensorContract, CausalDocGraph, TaxonomyGraph, RelationalAttrIndex)
        print("  [ok] core imports", flush=True)
    except Exception as e:
        errors.append(f"core imports: {e}")
        print(f"  [fail] core imports: {e}", flush=True)

    try:
        import torch
        import numpy as np

        _ = torch.zeros(1) + float(np.zeros(1)[0])
        print(f"  [ok] torch={torch.__version__} numpy={np.__version__}", flush=True)
    except Exception as e:
        errors.append(f"torch/numpy: {e}")
        print(f"  [fail] torch/numpy: {e}", flush=True)

    try:
        from vectorprism.base_encoder import HashingEncoder
        from vectorprism.ingestion_adapter import MultiTaskProjectionAdapter
        from vectorprism.ingest_pipeline import VectorPrismIngestPipeline, IngestDocument
        from vectorprism.db_client import VectorDBClient
        from vectorprism.paths import schema_sql_path, example_jsonl

        class _Mem(VectorDBClient):
            def __init__(self):
                self.rows = []

            def upsert(self, doc_id, chunk_text, tensor_1024d, meta):
                self.rows.append({"document_id": doc_id})

            def query_dense_slice(
                self, vector_slice, min_truth, max_anchor_dist, limit, model_version=None
            ):
                return []

            def get_by_ids(self, doc_ids):
                return []

        enc = HashingEncoder(768)
        adapter = MultiTaskProjectionAdapter(768)
        db = _Mem()
        pipe = VectorPrismIngestPipeline(enc, adapter, db, model_version=1)
        pipe.upsert_documents(
            [IngestDocument(document_id="pilot_doc", chunk_text="pilot smoke document")]
        )
        assert len(db.rows) == 1
        _ = schema_sql_path().read_text(encoding="utf-8")[:40]
        _ = example_jsonl("documents.example.jsonl")
        print("  [ok] ingest smoke (hash encoder) + packaged schema/examples", flush=True)
    except Exception as e:
        errors.append(f"ingest smoke: {e}")
        print(f"  [fail] ingest smoke: {e}", flush=True)

    try:
        import sentence_transformers  # noqa: F401

        print("  [ok] sentence-transformers installed", flush=True)
    except Exception:
        print(
            "  [warn] sentence-transformers missing — pip install 'vectorprism[encoder]' or '.[all]'",
            flush=True,
        )

    if errors:
        print('\nFix with: pip install -e ".[all]"', flush=True)
        print("Docs: PILOT.md", flush=True)
        return 1
    print(
        "\nPilot install looks good. Next: drop partner JSONL and run train/ingest/search.",
        flush=True,
    )
    print("See PILOT.md for the full external-partner runbook.", flush=True)
    return 0


def cmd_train(args) -> int:
    from vectorprism.train import main as train_main
    forwarded = [
        "--channel", args.channel,
        "--data", args.data,
        "--out", args.out,
        "--encoder", args.encoder,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--device", args.device,
    ]
    if args.init:
        forwarded += ["--init", args.init]
    if args.identity_corpus:
        forwarded += ["--identity-corpus", args.identity_corpus]
    if getattr(args, "num_disentangled_classes", None) is not None:
        forwarded += ["--num-disentangled-classes", str(args.num_disentangled_classes)]
    if getattr(args, "unsafe_pickle", False):
        forwarded += ["--unsafe-pickle"]
    train_main(forwarded)
    return 0


def cmd_eval(args) -> int:
    from vectorprism.eval_runner import run_phase1_eval, write_report
    from vectorprism.train import build_encoder
    encoder = build_encoder(args.encoder, args.device)
    report = run_phase1_eval(args.checkpoint, encoder, args.documents, args.eval)
    payload = {
        "vectorprism": report.vectorprism,
        "dense_cosine_baseline": report.dense_cosine_baseline,
        "beats_or_ties_baseline": report.beats_or_ties_baseline,
    }
    if args.out:
        write_report(args.out, payload)
    print(json.dumps(payload, indent=2))
    return 0 if report.beats_or_ties_baseline else 2


def cmd_ablation(args) -> int:
    from vectorprism.eval_runner import run_ablation_eval, write_report
    from vectorprism.train import build_encoder
    encoder = build_encoder(args.encoder, args.device)
    results = run_ablation_eval(args.checkpoint, encoder, args.documents, args.eval)
    if args.out:
        write_report(args.out, results)
    return 0


def cmd_intrinsic(args) -> int:
    from vectorprism.intrinsic_runner import run_intrinsic, write_json
    from vectorprism.train import build_encoder
    encoder = build_encoder(args.encoder, args.device)
    report = run_intrinsic(args.channel, args.checkpoint, encoder, args.data, ood_path=args.ood)
    if args.out:
        write_json(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("pass_intrinsic") else 2


def cmd_train_truth(args) -> int:
    from vectorprism.epistemic_truth import train_truth_classifier
    from vectorprism.train import build_encoder
    encoder = build_encoder(args.encoder, args.device)
    report = train_truth_classifier(
        args.data,
        encoder,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(report.__dict__, indent=2))
    return 0 if report.hard_filter_allowed else 2


def cmd_ingest(args) -> int:
    from vectorprism.ingest_cli import main as ingest_main
    forwarded = [
        "--checkpoint", args.checkpoint,
        "--documents", args.documents,
        "--encoder", args.encoder,
        "--backend", args.backend,
        "--device", args.device,
    ]
    if args.dsn:
        forwarded += ["--dsn", args.dsn]
    if args.qdrant_url:
        forwarded += ["--qdrant-url", args.qdrant_url]
    if args.truth_checkpoint:
        forwarded += ["--truth-checkpoint", args.truth_checkpoint]
    if getattr(args, "store", None):
        forwarded += ["--store", args.store]
    if getattr(args, "unsafe_pickle", False):
        forwarded += ["--unsafe-pickle"]
    ingest_main(forwarded)
    return 0


def cmd_search(args) -> int:
    from vectorprism.search_cli import main as search_main
    forwarded = [
        "--checkpoint", args.checkpoint,
        "--query", args.query,
        "--encoder", args.encoder,
        "--backend", args.backend,
        "--top-k", str(args.top_k),
        "--device", args.device,
    ]
    if args.dsn:
        forwarded += ["--dsn", args.dsn]
    if args.qdrant_url:
        forwarded += ["--qdrant-url", args.qdrant_url]
    if args.hard_truth_filter:
        forwarded += ["--hard-truth-filter"]
    if args.truth_checkpoint:
        forwarded += ["--truth-checkpoint", args.truth_checkpoint]
    if getattr(args, "store", None):
        forwarded += ["--store", args.store]
    if getattr(args, "unsafe_pickle", False):
        forwarded += ["--unsafe-pickle"]
    if getattr(args, "model_version", None) is not None:
        forwarded += ["--model-version", str(args.model_version)]
    hits = search_main(forwarded)
    return 0 if hits else 1


def cmd_live_benchmark(args) -> int:
    from vectorprism.live_benchmark import main as bench_main
    report = bench_main([
        "--checkpoint", args.checkpoint,
        "--documents", args.documents,
        "--encoder", args.encoder,
        "--backend", args.backend,
        "--n-trials", str(args.n_trials),
        "--p95-budget-ms", str(args.p95_budget_ms),
        "--device", args.device,
        *(["--dsn", args.dsn] if args.dsn else []),
        *(["--qdrant-url", args.qdrant_url] if args.qdrant_url else []),
    ])
    return 0 if report["pass_sla"] else 2


def cmd_reingest(args) -> int:
    from vectorprism.reingest import main as reingest_main
    reingest_main([
        "--checkpoint", args.checkpoint,
        "--documents", args.documents,
        "--encoder", args.encoder,
        "--backend", args.backend,
        "--device", args.device,
        *(["--dsn", args.dsn] if args.dsn else []),
        *(["--qdrant-url", args.qdrant_url] if args.qdrant_url else []),
        *(["--truth-checkpoint", args.truth_checkpoint] if args.truth_checkpoint else []),
    ])
    return 0


def cmd_run_all_smoke(args) -> int:
    """Full plumbing smoke across Phases 1-6 on example JSONL. Not a quality DoD pass."""
    data = _example_data_dir()
    ckpt = ROOT / "checkpoints" / "vectorprism_smoke.pt"
    truth = ROOT / "checkpoints" / "truth_smoke.pt"
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)

    steps = [
        [sys.executable, "-m", "vectorprism", "train", "--channel", "dense",
         "--data", str(data / "dense_pairs.example.jsonl"),
         "--out", str(ckpt), "--encoder", "hash", "--epochs", "1", "--batch-size", "8"],
        [sys.executable, "-m", "vectorprism", "train", "--channel", "causal",
         "--data", str(data / "causal.example.jsonl"),
         "--init", str(ckpt), "--out", str(ckpt), "--encoder", "hash", "--epochs", "1"],
        [sys.executable, "-m", "vectorprism", "train", "--channel", "relational",
         "--data", str(data / "relational.example.jsonl"),
         "--init", str(ckpt), "--out", str(ckpt), "--encoder", "hash", "--epochs", "1"],
        [sys.executable, "-m", "vectorprism", "train", "--channel", "hyperbolic",
         "--data", str(data / "hyperbolic.example.jsonl"),
         "--init", str(ckpt), "--out", str(ckpt), "--encoder", "hash", "--epochs", "1"],
        [sys.executable, "-m", "vectorprism", "train", "--channel", "disentangled",
         "--data", str(data / "disentangled.example.jsonl"),
         "--init", str(ckpt), "--out", str(ckpt), "--encoder", "hash", "--epochs", "1",
         "--num-disentangled-classes", "8"],
        [sys.executable, "-m", "vectorprism", "train", "--channel", "identity",
         "--data", str(data / "identity.example.jsonl"),
         "--init", str(ckpt), "--out", str(ckpt), "--encoder", "hash", "--epochs", "1"],
        [sys.executable, "-m", "vectorprism", "train-truth",
         "--data", str(data / "truth.example.jsonl"),
         "--out", str(truth), "--encoder", "hash", "--epochs", "3"],
        [sys.executable, "-m", "vectorprism", "eval",
         "--checkpoint", str(ckpt),
         "--documents", str(data / "documents.example.jsonl"),
         "--eval", str(data / "eval.example.jsonl"),
         "--encoder", "hash",
         "--out", str(reports / "phase1_eval.json")],
        [sys.executable, "-m", "vectorprism", "ablation",
         "--checkpoint", str(ckpt),
         "--documents", str(data / "documents.example.jsonl"),
         "--eval", str(data / "eval.example.jsonl"),
         "--encoder", "hash",
         "--out", str(reports / "ablation.json")],
        [sys.executable, "-m", "vectorprism", "intrinsic",
         "--channel", "causal", "--checkpoint", str(ckpt),
         "--data", str(data / "causal.example.jsonl"), "--encoder", "hash",
         "--out", str(reports / "intrinsic_causal.json")],
        [sys.executable, "-m", "vectorprism", "ingest",
         "--checkpoint", str(ckpt),
         "--documents", str(data / "documents.example.jsonl"),
         "--encoder", "hash", "--backend", "memory",
         "--store", str(ROOT / "checkpoints" / "smoke_memory_corpus.npz"),
         "--truth-checkpoint", str(truth)],
        [sys.executable, "-m", "vectorprism", "search",
         "--checkpoint", str(ckpt),
         "--query", "VectorPrism uses a 1024-dimensional tensor",
         "--encoder", "hash", "--backend", "memory",
         "--store", str(ROOT / "checkpoints" / "smoke_memory_corpus.npz"),
         "--top-k", "3"],
        [sys.executable, "-m", "vectorprism", "live-benchmark",
         "--checkpoint", str(ckpt),
         "--documents", str(data / "documents.example.jsonl"),
         "--encoder", "hash", "--backend", "memory",
         "--n-trials", "10", "--p95-budget-ms", "100"],
        [sys.executable, "-m", "vectorprism", "reingest",
         "--checkpoint", str(ckpt),
         "--documents", str(data / "documents.example.jsonl"),
         "--encoder", "hash", "--backend", "memory"],
    ]

    # train-truth may exit 2 if ECE gate fails on tiny example data — allow that.
    allow_nonzero = {
        "vectorprism.py train-truth",
        "vectorprism.py eval",
        "vectorprism.py intrinsic",
    }
    for cmd in steps:
        key = " ".join(cmd[1:3]) if len(cmd) > 2 else cmd[-1]
        code = _run(cmd)
        if code != 0 and key not in allow_nonzero and "train-truth" not in " ".join(cmd) \
           and "eval" not in cmd and "intrinsic" not in cmd:
            # Allow DoD gate exits (2) from eval/intrinsic/train-truth
            if code == 2 and any(x in cmd for x in ["eval", "intrinsic", "train-truth"]):
                print(f"[smoke] DoD gate not met (exit {code}) — expected on tiny example data")
                continue
            return code
        if code not in (0, 2):
            return code
        if code == 2:
            print(f"[smoke] DoD gate not met (exit 2) — expected on tiny example data: {' '.join(cmd[1:4])}")
    print("[smoke] Full Phase 0-6 plumbing completed on example JSONL.")
    print("[smoke] Quality DoDs still require YOUR labeled datasets + real encoder.")
    return 0


def cmd_finance_demo(args) -> int:
    cmd = [
        sys.executable, str(ROOT / "demos" / "finance_demo" / "run_demo.py"),
        "--encoder", args.encoder,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--out", args.out,
    ]
    if args.skip_train:
        cmd.append("--skip-train")
    return _run(cmd)


def cmd_finance_pg(args) -> int:
    cmd = [
        sys.executable, str(ROOT / "demos" / "finance_demo" / "run_pgvector_demo.py"),
        "--dsn", args.dsn,
        "--checkpoint", args.checkpoint,
        "--encoder", args.encoder,
        "--epochs", str(args.epochs),
    ]
    if args.skip_train:
        cmd.append("--skip-train")
    return _run(cmd)


def cmd_hard_eval(args) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "demos" / "finance_demo" / "run_hard_eval.py"),
        "--mode",
        args.mode,
        "--encoder",
        args.encoder,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--packs",
        *list(args.packs),
    ]
    if args.skip_train:
        cmd.append("--skip-train")
    return _run(cmd)


def cmd_causal_recovery(args) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "demos" / "finance_demo" / "run_causal_recovery.py"),
        "--pack",
        args.pack,
        "--encoder",
        args.encoder,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--k",
        str(args.k),
    ]
    if args.skip_train:
        cmd.append("--skip-train")
    return _run(cmd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vectorprism", description="VectorPrism Phases 0-6 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("version", help="Print installed package version")
    s.set_defaults(func=cmd_version)

    s = sub.add_parser(
        "pilot-check",
        help="Validate partner install (imports + tiny ingest smoke)",
    )
    s.set_defaults(func=cmd_pilot_check)

    s = sub.add_parser("phase0", help="Run test suite")
    s.set_defaults(func=cmd_phase0)

    s = sub.add_parser("train", help="Train one channel")
    s.add_argument("--channel", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--out", default="checkpoints/vectorprism.pt")
    s.add_argument("--init", default=None)
    s.add_argument("--encoder", default="hash")
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--device", default="cpu")
    s.add_argument("--identity-corpus", default=None)
    s.add_argument("--num-disentangled-classes", type=int, default=None)
    s.add_argument("--unsafe-pickle", action="store_true")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("eval", help="Phase-1 eval vs dense baseline")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--documents", required=True)
    s.add_argument("--eval", required=True)
    s.add_argument("--encoder", default="hash")
    s.add_argument("--device", default="cpu")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("ablation", help="Channel ablation report")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--documents", required=True)
    s.add_argument("--eval", required=True)
    s.add_argument("--encoder", default="hash")
    s.add_argument("--device", default="cpu")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_ablation)

    s = sub.add_parser("intrinsic", help="Per-channel intrinsic DoD")
    s.add_argument("--channel", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--ood", default=None)
    s.add_argument("--encoder", default="hash")
    s.add_argument("--device", default="cpu")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_intrinsic)

    s = sub.add_parser("train-truth", help="Phase-4 truth classifier")
    s.add_argument("--data", required=True)
    s.add_argument("--out", default="checkpoints/truth.pt")
    s.add_argument("--encoder", default="hash")
    s.add_argument("--epochs", type=int, default=5)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--device", default="cpu")
    s.set_defaults(func=cmd_train_truth)

    for name, helper in [("ingest", cmd_ingest), ("reingest", cmd_reingest)]:
        s = sub.add_parser(name)
        s.add_argument("--checkpoint", required=True)
        s.add_argument("--documents", required=True)
        s.add_argument("--encoder", default="hash")
        s.add_argument("--backend", default="memory")
        s.add_argument("--dsn", default=None)
        s.add_argument("--qdrant-url", default=None)
        s.add_argument("--truth-checkpoint", default=None)
        s.add_argument("--device", default="cpu")
        s.add_argument("--store", default=None)
        s.add_argument("--unsafe-pickle", action="store_true")
        s.set_defaults(func=helper)

    s = sub.add_parser("search")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--query", required=True)
    s.add_argument("--encoder", default="hash")
    s.add_argument("--backend", default="memory")
    s.add_argument("--dsn", default=None)
    s.add_argument("--qdrant-url", default=None)
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--hard-truth-filter", action="store_true")
    s.add_argument("--truth-checkpoint", default=None)
    s.add_argument("--device", default="cpu")
    s.add_argument("--store", default=None)
    s.add_argument("--model-version", type=int, default=None)
    s.add_argument("--unsafe-pickle", action="store_true")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("live-benchmark")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--documents", required=True)
    s.add_argument("--encoder", default="hash")
    s.add_argument("--backend", default="memory")
    s.add_argument("--dsn", default=None)
    s.add_argument("--qdrant-url", default=None)
    s.add_argument("--n-trials", type=int, default=20)
    s.add_argument("--p95-budget-ms", type=float, default=50.0)
    s.add_argument("--device", default="cpu")
    s.set_defaults(func=cmd_live_benchmark)

    s = sub.add_parser("run-all-smoke", help="Full plumbing smoke on example data")
    s.set_defaults(func=cmd_run_all_smoke)

    s = sub.add_parser("finance-demo", help="Near-real finance client Phase-1 demo")
    s.add_argument("--encoder", default="hash")
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--out", default="checkpoints/finance_demo.pt")
    s.add_argument("--skip-train", action="store_true")
    s.set_defaults(func=cmd_finance_demo)

    s = sub.add_parser("finance-pg", help="Finance demo on Postgres/pgvector")
    s.add_argument("--dsn", default=os.environ.get(
        "VECTORPRISM_PG_DSN",
        "postgresql://vectorprism:vectorprism@localhost:5433/vectorprism",
    ))
    s.add_argument("--checkpoint", default="checkpoints/finance_demo.pt")
    s.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--skip-train", action="store_true")
    s.set_defaults(func=cmd_finance_pg)

    s = sub.add_parser("hard-eval", help="Dense hard-eval baseline (isolated or mixed packs)")
    s.add_argument("--mode", choices=["isolated", "mixed"], default="isolated")
    s.add_argument(
        "--packs",
        nargs="+",
        choices=["gemini", "gpt", "adversarial"],
        default=["adversarial"],
    )
    s.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--skip-train", action="store_true")
    s.set_defaults(func=cmd_hard_eval)

    s = sub.add_parser(
        "causal-recovery",
        help="Train causal from dense ckpt; score recovery on confirmed dense misses",
    )
    s.add_argument(
        "--pack",
        choices=["gemini", "gpt", "adversarial", "both", "all"],
        default="adversarial",
    )
    s.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    s.add_argument("--epochs", type=int, default=5)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--skip-train", action="store_true")
    s.add_argument("--k", type=int, default=10)
    s.set_defaults(func=cmd_causal_recovery)
    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
