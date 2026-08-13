"""
Step 1 — Causal channel recovery scorecard.

For an isolated hard pack:
  1) Train causal on pack causal.jsonl from a dense checkpoint
  2) Mine confirmed dense misses@k
  3) Score dense_only vs dense+causal vs intent_router on those misses
  4) Report Stage-1 recall@100 (if gold ∉ top-100 dense, Stage-2 cannot recover)

Usage:
  docker compose run --rm vectorprism python demos/finance_demo/run_causal_recovery.py --pack gemini
  docker compose run --rm vectorprism python demos/finance_demo/run_causal_recovery.py --pack gpt
  docker compose run --rm vectorprism python demos/finance_demo/run_causal_recovery.py --pack both
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PACKS = {
    "gemini": {
        "dir": DEMO / "hard_gemini",
        "dense_ckpt": ROOT / "checkpoints" / "finance_hard_gemini.pt",
        "out_ckpt": ROOT / "checkpoints" / "finance_hard_gemini_causal.pt",
    },
    "gpt": {
        "dir": DEMO / "hard_gpt",
        "dense_ckpt": ROOT / "checkpoints" / "finance_hard_gpt.pt",
        "out_ckpt": ROOT / "checkpoints" / "finance_hard_gpt_causal.pt",
    },
    "adversarial": {
        "dir": DEMO / "hard_adversarial",
        "dense_ckpt": ROOT / "checkpoints" / "finance_hard_adversarial.pt",
        "out_ckpt": ROOT / "checkpoints" / "finance_hard_adversarial_causal.pt",
    },
}
RESULTS = DEMO / "results" / "hard_eval"


def _run(cmd: list[str]) -> int:
    print("\n+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def _build_encoder(name: str):
    from base_encoder import HashingEncoder, SentenceTransformerEncoder

    if name in {"hash", "hashing", "test"}:
        return HashingEncoder(768)
    return SentenceTransformerEncoder(name)


def _ingest(checkpoint: str, encoder_name: str, documents_jsonl: Path):
    from channel_datasets import load_documents_jsonl
    from checkpointing import load_checkpoint
    from eval_runner import InMemoryCorpusDB
    from ingest_pipeline import VectorPrismIngestPipeline, IngestDocument

    encoder = _build_encoder(encoder_name)
    ckpt = load_checkpoint(checkpoint)
    db = InMemoryCorpusDB()
    pipe = VectorPrismIngestPipeline(
        encoder,
        ckpt["adapter"],
        db,
        model_version=ckpt["model_version"],
        enabled_channels=ckpt.get("enabled_channels"),
    )
    docs = load_documents_jsonl(str(documents_jsonl))
    pipe.upsert_documents(
        [
            IngestDocument(
                document_id=str(d["document_id"]),
                chunk_text=str(d["chunk_text"]),
                epistemic_truth=float(d.get("epistemic_truth", 1.0)),
            )
            for d in docs
        ]
    )
    return encoder, ckpt, db, pipe


def _eval_set(eval_jsonl: Path, encoder, pipe):
    from eval_runner import build_eval_set

    return build_eval_set(str(eval_jsonl), encoder, pipe)


def _dense_rank_stats(eval_set, db, k_miss: int = 10, stage1_limit: int = 100):
    """Return per-query dense ranks and confirmed miss indices."""
    from tensor_contract import PSMTensorContract as C

    doc_ids = [r["document_id"] for r in db.rows]
    mat = np.stack([r["tensor_1024d"][C.DENSE_CORE.start : C.DENSE_CORE.end] for r in db.rows])
    rows = []
    miss_idxs = []
    for i, ex in enumerate(eval_set):
        q = ex.query_1024d[C.DENSE_CORE.start : C.DENSE_CORE.end]
        scores = mat @ q
        order = np.argsort(scores)[::-1]
        ranked = [doc_ids[j] for j in order]
        gold = ex.relevant_doc_ids
        rank = None
        for rnk, did in enumerate(ranked, start=1):
            if did in gold:
                rank = rnk
                break
        in_stage1 = rank is not None and rank <= stage1_limit
        hit_k = rank is not None and rank <= k_miss
        if not hit_k:
            miss_idxs.append(i)
        rows.append(
            {
                "query": ex.query_text,
                "gold": sorted(gold),
                "dense_rank": rank,
                "in_stage1_top100": in_stage1,
                "dense_hit_at_k": hit_k,
            }
        )
    return rows, miss_idxs


def _search_ids(engine, ex, top_k: int) -> list[str]:
    results = engine.search(ex.query_1024d, ex.query_text, top_k=top_k)
    return [r["document_id"] for r in results]


def _hit(retrieved: list[str], gold: set[str], k: int) -> bool:
    return bool(set(retrieved[:k]) & gold)


def score_recovery(
    *,
    checkpoint: str,
    encoder_name: str,
    documents_jsonl: Path,
    eval_jsonl: Path,
    k: int = 10,
) -> dict[str, Any]:
    from ablation_harness import _fixed_weight_classifier
    from retrieval_engine import PSMRetrievalEngine

    encoder, ckpt, db, pipe = _ingest(checkpoint, encoder_name, documents_jsonl)
    eval_set = _eval_set(eval_jsonl, encoder, pipe)
    dense_stats, miss_idxs = _dense_rank_stats(eval_set, db, k_miss=k)

    engine = PSMRetrievalEngine(
        db,
        causal_matrix=ckpt["causal_matrix"],
        hard_truth_filter=False,
    )
    original = engine.classifier.classify

    configs = {
        "dense_only": np.array([1.0, 0, 0, 0, 0], dtype=np.float32),
        "dense+causal": np.array([0.7, 0, 0, 0, 0.3], dtype=np.float32),
        "causal_heavy": np.array([0.35, 0.1, 0.1, 0.0, 0.45], dtype=np.float32),
    }

    # Full-set metrics
    full_metrics: dict[str, dict[str, float]] = {}
    for name, w in configs.items():
        engine.classifier.classify = _fixed_weight_classifier(w)
        hits = {1: 0, 5: 0, 10: 0}
        rr = []
        for ex in eval_set:
            ids = _search_ids(engine, ex, top_k=10)
            for kk in hits:
                if _hit(ids, ex.relevant_doc_ids, kk):
                    hits[kk] += 1
            rank = 0.0
            for i, did in enumerate(ids, start=1):
                if did in ex.relevant_doc_ids:
                    rank = 1.0 / i
                    break
            rr.append(rank)
        n = max(len(eval_set), 1)
        full_metrics[name] = {
            "recall@1": hits[1] / n,
            "recall@5": hits[5] / n,
            "recall@10": hits[10] / n,
            "MRR": float(np.mean(rr)) if rr else 0.0,
            "n": len(eval_set),
        }

    # Intent router (real classifier)
    engine.classifier.classify = original
    hits_i = {1: 0, 5: 0, 10: 0}
    rr_i = []
    for ex in eval_set:
        ids = _search_ids(engine, ex, top_k=10)
        for kk in hits_i:
            if _hit(ids, ex.relevant_doc_ids, kk):
                hits_i[kk] += 1
        rank = 0.0
        for i, did in enumerate(ids, start=1):
            if did in ex.relevant_doc_ids:
                rank = 1.0 / i
                break
        rr_i.append(rank)
    n = max(len(eval_set), 1)
    full_metrics["intent_router"] = {
        "recall@1": hits_i[1] / n,
        "recall@5": hits_i[5] / n,
        "recall@10": hits_i[10] / n,
        "MRR": float(np.mean(rr_i)) if rr_i else 0.0,
        "n": len(eval_set),
    }

    # Miss-set recovery
    miss_examples = []
    recover_counts = {name: {"@1": 0, "@5": 0, "@10": 0} for name in list(configs) + ["intent_router"]}
    stage1_in = 0
    for idx in miss_idxs:
        ex = eval_set[idx]
        st = dense_stats[idx]
        if st["in_stage1_top100"]:
            stage1_in += 1
        entry: dict[str, Any] = {
            "query": ex.query_text,
            "gold": st["gold"],
            "dense_rank": st["dense_rank"],
            "in_stage1_top100": st["in_stage1_top100"],
            "configs": {},
        }
        for name, w in configs.items():
            engine.classifier.classify = _fixed_weight_classifier(w)
            ids = _search_ids(engine, ex, top_k=10)
            h1, h5, h10 = _hit(ids, ex.relevant_doc_ids, 1), _hit(ids, ex.relevant_doc_ids, 5), _hit(
                ids, ex.relevant_doc_ids, 10
            )
            if h1:
                recover_counts[name]["@1"] += 1
            if h5:
                recover_counts[name]["@5"] += 1
            if h10:
                recover_counts[name]["@10"] += 1
            entry["configs"][name] = {"hit@1": h1, "hit@5": h5, "hit@10": h10, "top5": ids[:5]}

        engine.classifier.classify = original
        ids = _search_ids(engine, ex, top_k=10)
        h1, h5, h10 = _hit(ids, ex.relevant_doc_ids, 1), _hit(ids, ex.relevant_doc_ids, 5), _hit(
            ids, ex.relevant_doc_ids, 10
        )
        if h1:
            recover_counts["intent_router"]["@1"] += 1
        if h5:
            recover_counts["intent_router"]["@5"] += 1
        if h10:
            recover_counts["intent_router"]["@10"] += 1
        entry["configs"]["intent_router"] = {
            "hit@1": h1,
            "hit@5": h5,
            "hit@10": h10,
            "top5": ids[:5],
        }
        miss_examples.append(entry)

    n_miss = max(len(miss_idxs), 1)
    recovery_rates = {
        name: {
            "recovery@1": c["@1"] / n_miss,
            "recovery@5": c["@5"] / n_miss,
            "recovery@10": c["@10"] / n_miss,
            "recovered@1": c["@1"],
            "recovered@5": c["@5"],
            "recovered@10": c["@10"],
        }
        for name, c in recover_counts.items()
    }

    # Net new recoveries vs dense_only on the miss set (dense_only should be ~0 @10 by definition)
    dense_rec10 = recover_counts["dense_only"]["@10"]
    causal_lift = {
        name: recovery_rates[name]["recovered@10"] - dense_rec10
        for name in recovery_rates
        if name != "dense_only"
    }

    return {
        "n_eval": len(eval_set),
        "n_dense_miss_at_k": len(miss_idxs),
        "k": k,
        "stage1_top100_coverage_on_misses": stage1_in / n_miss if miss_idxs else 1.0,
        "stage1_top100_in_count": stage1_in,
        "full_set_metrics": full_metrics,
        "recovery_on_dense_misses": recovery_rates,
        "causal_lift_recovered@10_vs_dense_only": causal_lift,
        "enabled_channels": ckpt.get("enabled_channels"),
        "miss_examples": miss_examples[:30],
        "all_miss_queries": [dense_stats[i]["query"] for i in miss_idxs],
    }


def run_pack(
    pack_name: str,
    *,
    encoder: str,
    epochs: int,
    batch_size: int,
    skip_train: bool,
    k: int,
) -> dict[str, Any]:
    meta = PACKS[pack_name]
    pack_dir = meta["dir"]
    dense_ckpt = meta["dense_ckpt"]
    out_ckpt = meta["out_ckpt"]
    pairs = pack_dir / "causal.jsonl"
    docs = pack_dir / "documents.jsonl"
    ev = pack_dir / "eval.jsonl"

    for p in (pairs, docs, ev, dense_ckpt):
        if not p.exists():
            raise SystemExit(
                f"missing {p} — run isolated dense hard-eval first "
                f"(demos/finance_demo/run_hard_eval.py --mode isolated)"
            )

    n_causal = sum(1 for line in pairs.open(encoding="utf-8") if line.strip())
    print(f"\n===== CAUSAL RECOVERY: {pack_name} =====", flush=True)
    print(f"dense_ckpt={dense_ckpt} causal_pairs={n_causal}", flush=True)

    if not skip_train:
        code = _run(
            [
                sys.executable,
                "train.py",
                "--channel",
                "causal",
                "--data",
                str(pairs),
                "--init",
                str(dense_ckpt),
                "--out",
                str(out_ckpt),
                "--encoder",
                encoder,
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
            ]
        )
        if code != 0:
            raise SystemExit(code)
    elif not out_ckpt.exists():
        raise SystemExit(f"--skip-train but missing {out_ckpt}")

    print(f"[recovery] scoring {pack_name} with {out_ckpt.name}", flush=True)
    scored = score_recovery(
        checkpoint=str(out_ckpt),
        encoder_name=encoder,
        documents_jsonl=docs,
        eval_jsonl=ev,
        k=k,
    )
    scored["pack"] = pack_name
    scored["dense_checkpoint"] = str(dense_ckpt)
    scored["causal_checkpoint"] = str(out_ckpt)
    scored["n_causal_pairs"] = n_causal
    scored["encoder"] = encoder
    return scored


def write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# Causal recovery scorecard (Step 1)",
        "",
        "Trained causal from dense checkpoint; scored recovery on confirmed dense misses.",
        "",
    ]
    for pack_name, r in summary["packs"].items():
        full = r["full_set_metrics"]
        rec = r["recovery_on_dense_misses"]
        lines += [
            f"## Pack: {pack_name}",
            "",
            f"- Causal pairs: {r['n_causal_pairs']}",
            f"- Checkpoint: `{r['causal_checkpoint']}`",
            f"- Dense misses@{r['k']}: **{r['n_dense_miss_at_k']}** / {r['n_eval']}",
            f"- Stage-1 top-100 coverage on those misses: "
            f"**{r['stage1_top100_in_count']}** / {r['n_dense_miss_at_k']} "
            f"({r['stage1_top100_coverage_on_misses']:.1%})",
            "",
            "### Full-set recall",
            "",
            "| Config | R@1 | R@5 | R@10 | MRR |",
            "|--------|-----|-----|------|-----|",
        ]
        for cfg, m in full.items():
            lines.append(
                f"| {cfg} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | "
                f"{m['recall@10']:.3f} | {m['MRR']:.3f} |"
            )
        lines += [
            "",
            "### Recovery on dense misses@10",
            "",
            "| Config | recovered@1 | recovered@5 | recovered@10 |",
            "|--------|-------------|-------------|--------------|",
        ]
        for cfg, m in rec.items():
            lines.append(
                f"| {cfg} | {m['recovered@1']} ({m['recovery@1']:.1%}) | "
                f"{m['recovered@5']} ({m['recovery@5']:.1%}) | "
                f"{m['recovered@10']} ({m['recovery@10']:.1%}) |"
            )
        lift = r["causal_lift_recovered@10_vs_dense_only"]
        lines += [
            "",
            f"- Lift@10 vs dense_only on miss set: `{json.dumps(lift)}`",
            "",
        ]

    lines += [
        "## How to read this",
        "",
        "- **Recovery@10 > 0 on dense+causal** while dense_only stays ~0 ⇒ channel earns its keep.",
        "- **Stage-1 coverage < 100%** ⇒ some golds never enter the rescoring pool; Stage-2 cannot fix those.",
        "- Full-set R@10 moving up with dense+causal is secondary; the miss-set lift is the product claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Causal recovery scorecard")
    ap.add_argument(
        "--pack",
        choices=["gemini", "gpt", "adversarial", "both", "all"],
        default="adversarial",
    )
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--k", type=int, default=10, help="Dense miss threshold (default 10)")
    args = ap.parse_args(argv)

    if args.pack == "both":
        names = ["gemini", "gpt"]
    elif args.pack == "all":
        names = ["gemini", "gpt", "adversarial"]
    else:
        names = [args.pack]
    RESULTS.mkdir(parents=True, exist_ok=True)

    packs_out = {}
    for name in names:
        packs_out[name] = run_pack(
            name,
            encoder=args.encoder,
            epochs=args.epochs,
            batch_size=args.batch_size,
            skip_train=args.skip_train,
            k=args.k,
        )

    summary = {"step": 1, "encoder": args.encoder, "packs": packs_out}
    json_path = RESULTS / "causal_recovery.json"
    md_path = RESULTS / "CAUSAL_RECOVERY_RESULTS.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, md_path)

    print("\n========== CAUSAL RECOVERY SUMMARY ==========")
    for name, r in packs_out.items():
        rec = r["recovery_on_dense_misses"]
        print(
            f"{name}: dense_misses@{r['k']}={r['n_dense_miss_at_k']}/{r['n_eval']}  "
            f"stage1_cov={r['stage1_top100_coverage_on_misses']:.0%}  "
            f"dense+causal recovered@10={rec['dense+causal']['recovered@10']}  "
            f"intent recovered@10={rec['intent_router']['recovered@10']}  "
            f"full R@10 dense={r['full_set_metrics']['dense_only']['recall@10']:.3f} "
            f"dense+causal={r['full_set_metrics']['dense+causal']['recall@10']:.3f}"
        )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
