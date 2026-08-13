"""
Run automated structure extraction on a pack and compare to curated graphs.

Optionally score multi-channel recovery using the *auto* graphs only.

Usage:
  python demos/finance_demo/extract_structure_auto.py --backend heuristic
  docker compose run --rm vectorprism python demos/finance_demo/extract_structure_auto.py --backend heuristic --score
  # with LLM:
  OPENAI_API_KEY=... docker compose run --rm -e OPENAI_API_KEY vectorprism \\
    python demos/finance_demo/extract_structure_auto.py --backend llm --score
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ADV = DEMO / "hard_adversarial"
OUT = DEMO / "hard_adversarial_auto"
RESULTS = DEMO / "results" / "hard_eval"
MULTI_CKPT = ROOT / "checkpoints" / "finance_hard_adversarial_multi.pt"
CAUSAL_CKPT = ROOT / "checkpoints" / "finance_hard_adversarial_causal.pt"


def _load_cr():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_causal_recovery", DEMO / "run_causal_recovery.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_with_graphs(
    *,
    checkpoint: str,
    encoder: str,
    docs_path: Path,
    eval_path: Path,
    causal_path: Path,
    tax_path: Path,
    rel_path: Path,
    label: str,
) -> dict[str, Any]:
    from ablation_harness import _fixed_weight_classifier
    from causal_graph import CausalDocGraph
    from retrieval_engine import PSMRetrievalEngine
    from structure_index import RelationalAttrIndex, TaxonomyGraph

    cr = _load_cr()
    encoder_obj, ckpt, db, pipe = cr._ingest(checkpoint, encoder, docs_path)
    eval_set = cr._eval_set(eval_path, encoder_obj, pipe)
    _, miss_idxs = cr._dense_rank_stats(eval_set, db, k_miss=10)
    causal_g = CausalDocGraph.from_jsonl(causal_path)
    tax_g = TaxonomyGraph.from_jsonl(tax_path)
    rel_idx = RelationalAttrIndex.from_jsonl(rel_path)
    text_lookup = {r["document_id"]: str(r.get("chunk_text", "")) for r in db.rows}
    w = np.array([0.25, 0.25, 0.05, 0.20, 0.25], dtype=np.float32)

    eng = PSMRetrievalEngine(
        db,
        causal_matrix=ckpt["causal_matrix"],
        causal_graph=causal_g,
        taxonomy_graph=tax_g,
        relational_index=rel_idx,
        stage1_dense_limit=50,
        stage1_graph_hops=2,
        graph_struct_bonus=4.0,
        fusion="zscore",
        doc_text_lookup=text_lookup,
        query_conditioned_expansion=True,
    )
    eng.classifier.classify = _fixed_weight_classifier(w)

    rec = 0
    hits = 0
    mrr = []
    for i, ex in enumerate(eval_set):
        ids = [r["document_id"] for r in eng.search(ex.query_1024d, ex.query_text, top_k=10)]
        gold = ex.relevant_doc_ids
        if set(ids[:10]) & gold:
            hits += 1
            if i in miss_idxs:
                rec += 1
        rr = 0.0
        for rank, did in enumerate(ids, start=1):
            if did in gold:
                rr = 1.0 / rank
                break
        mrr.append(rr)

    return {
        "label": label,
        "n_causal_edges": causal_g.n_edges,
        "n_tax_edges": tax_g.n_edges,
        "n_rel_attrs": len(rel_idx.attrs),
        "n_dense_miss": len(miss_idxs),
        "recovered@10": rec,
        "recovery@10": rec / max(len(miss_idxs), 1),
        "recall@10": hits / max(len(eval_set), 1),
        "MRR": float(np.mean(mrr)) if mrr else 0.0,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    cmp_ = report["comparison"]
    lines = [
        "# Automated structure extraction vs curated graphs",
        "",
        f"- Backend: `{report['backend']}`",
        f"- Output: `{report['out_dir']}`",
        "",
        "## Edge / attribute agreement vs curated",
        "",
        "| Structure | Precision | Recall | F1 | TP | FP | FN |",
        "|-----------|----------:|-------:|---:|---:|---:|---:|",
    ]
    for name, m in cmp_.items():
        lines.append(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
            f"{m['tp']} | {m['fp']} | {m['fn']} |"
        )
    if report.get("recovery"):
        lines += [
            "",
            "## Recovery with auto graphs (multi z-score)",
            "",
            "| Graph source | Causal edges | Tax edges | Rel attrs | Recovered@10 | R@10 | MRR |",
            "|--------------|-------------:|----------:|----------:|-------------:|-----:|----:|",
        ]
        for row in report["recovery"]:
            lines.append(
                f"| {row['label']} | {row['n_causal_edges']} | {row['n_tax_edges']} | "
                f"{row['n_rel_attrs']} | {row['recovered@10']}/{row['n_dense_miss']} | "
                f"{row['recall@10']:.3f} | {row['MRR']:.3f} |"
            )
    lines += [
        "",
        "## Notes",
        "",
        "- Extraction uses **document text only** (no eval gold id shortcuts).",
        "- Heuristic backend is offline/deterministic; LLM backend needs `OPENAI_API_KEY` "
        "or `VECTORPRISM_LLM_API_KEY` (+ optional `VECTORPRISM_LLM_BASE_URL` / `VECTORPRISM_LLM_MODEL`).",
        "- Low edge F1 with high recovery is possible when auto edges are *different but still useful*.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Automated structure extraction")
    ap.add_argument("--pack", default=str(ADV), help="Pack directory with documents.jsonl")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--backend", choices=["auto", "heuristic", "llm"], default="auto")
    ap.add_argument("--score", action="store_true", help="Score recovery with auto vs curated graphs")
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument(
        "--checkpoint",
        default=str(MULTI_CKPT if MULTI_CKPT.exists() else CAUSAL_CKPT),
    )
    args = ap.parse_args(argv)

    from structure_extract import (
        compare_extractions,
        extract_structure,
        load_documents_jsonl,
        write_extraction,
    )

    pack = Path(args.pack)
    docs = load_documents_jsonl(pack / "documents.jsonl")
    print(f"[extract] n_docs={len(docs)} backend={args.backend}", flush=True)
    result = extract_structure(docs, backend=args.backend)
    out_dir = Path(args.out)
    paths = write_extraction(result, out_dir)
    print(
        f"[extract] wrote causal={len(result.causal_edges)} tax={len(result.taxonomy_edges)} "
        f"rel={len(result.relational_attrs)} → {out_dir}",
        flush=True,
    )

    cmp_ = compare_extractions(out_dir, pack)
    report: dict[str, Any] = {
        "backend": result.backend,
        "out_dir": str(out_dir),
        "meta": result.meta,
        "comparison": cmp_,
        "paths": {k: str(v) for k, v in paths.items()},
    }

    if args.score:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            raise SystemExit(f"--score requires checkpoint {ckpt}")
        recovery = []
        recovery.append(
            score_with_graphs(
                checkpoint=str(ckpt),
                encoder=args.encoder,
                docs_path=pack / "documents.jsonl",
                eval_path=pack / "eval.jsonl",
                causal_path=out_dir / "causal_graph.jsonl",
                tax_path=out_dir / "hyperbolic_graph.jsonl",
                rel_path=out_dir / "relational_attrs.jsonl",
                label="auto",
            )
        )
        recovery.append(
            score_with_graphs(
                checkpoint=str(ckpt),
                encoder=args.encoder,
                docs_path=pack / "documents.jsonl",
                eval_path=pack / "eval.jsonl",
                causal_path=pack / "causal_graph.jsonl",
                tax_path=pack / "hyperbolic_graph.jsonl",
                rel_path=pack / "relational_attrs.jsonl",
                label="curated",
            )
        )
        report["recovery"] = recovery
        for row in recovery:
            print(
                f"[score] {row['label']}: recovered@10={row['recovered@10']}/{row['n_dense_miss']} "
                f"R@10={row['recall@10']:.3f} MRR={row['MRR']:.3f}",
                flush=True,
            )

    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "auto_extraction.json"
    md_path = RESULTS / "AUTO_EXTRACTION.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
