"""
Corpus recovery audit — partner / public documents + query log.

Unlike demos/finance_demo/run_multichannel_recovery.py (hardcoded adversarial pack),
this CLI takes *any* VectorPrism-shaped JSONL:

  documents.jsonl  {document_id, chunk_text, epistemic_truth?}
  eval.jsonl       {query, relevant_doc_ids}
  optional:
    causal_graph.jsonl
    hyperbolic_graph.jsonl
    relational_attrs.jsonl

Produces JSON + Markdown audit report under --out.

Usage:
  docker compose run --rm --no-deps vectorprism \\
    python scripts/corpus_recovery_audit.py \\
      --documents demos/external_audit/packs/incident_bench/documents.jsonl \\
      --eval demos/external_audit/packs/incident_bench/eval.jsonl \\
      --checkpoint checkpoints/finance_hard_adversarial_multi.pt \\
      --encoder sentence-transformers/all-mpnet-base-v2 \\
      --out demos/external_audit/results/incident_bench \\
      --vertical sre_incident --pack-meta demos/external_audit/packs/incident_bench/meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
    db = InMemoryCorpusDB(store_path=None, autoload=False)
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


def _dense_rank_stats(eval_set, db, k_miss: int = 10):
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
        hit_k = rank is not None and rank <= k_miss
        if not hit_k:
            miss_idxs.append(i)
        rows.append(
            {
                "query": ex.query_text,
                "gold": sorted(gold),
                "dense_rank": rank,
                "dense_hit_at_k": hit_k,
                "dense_top5": ranked[:5],
            }
        )
    return rows, miss_idxs


def _search_ids(engine, ex, top_k: int) -> list[str]:
    results = engine.search(ex.query_1024d, ex.query_text, top_k=top_k)
    return [r["document_id"] for r in results]


def _hit(retrieved: list[str], gold: set[str], k: int) -> bool:
    return bool(set(retrieved[:k]) & gold)


def _liability_note(vertical: str, n_silent: int, n_eval: int) -> dict[str, Any]:
    """Scenario framing — not actuarial. For regulated verticals."""
    rate = (n_silent / max(n_eval, 1)) * 100.0
    catalogs = {
        "sre_incident": {
            "failure_mode": "Wrong runbook / root-cause incident cited → longer MTTR, repeat outage",
            "unit_cost_assumption_usd": 15_000,
            "unit_label": "per major incident with wrong root-cause retrieval",
            "note": "Industry SRE cost-of-downtime proxies vary widely; use as conversation starter only.",
        },
        "finance_compliance": {
            "failure_mode": "Wrong policy / threshold retrieved → incorrect control applied",
            "unit_cost_assumption_usd": 75_000,
            "unit_label": "per material control miss or exam finding driven by wrong SOP",
            "note": "Illustrative; real exam/fine exposure is firm-specific.",
        },
        "healthcare": {
            "failure_mode": "Wrong protocol retrieved → clinical / privacy risk",
            "unit_cost_assumption_usd": 50_000,
            "unit_label": "per serious protocol miss attributed to retrieval error",
            "note": "Illustrative liability framing only — not medical advice.",
        },
        "generic": {
            "failure_mode": "Silent wrong neighbor in top-k → hallucinated answer grounded on wrong doc",
            "unit_cost_assumption_usd": 5_000,
            "unit_label": "per high-severity wrong-answer incident",
            "note": "Placeholder unit cost until partner provides loss model.",
        },
    }
    cat = catalogs.get(vertical, catalogs["generic"])
    expected = rate / 100.0 * cat["unit_cost_assumption_usd"]
    return {
        **cat,
        "silent_fail_rate_pct": round(rate, 1),
        "n_silent_fails_at_k": n_silent,
        "n_eval": n_eval,
        "illustrative_expected_loss_per_100_queries_usd": round(expected * 100.0 / max(n_eval, 1) * n_eval, 0)
        if n_eval
        else 0,
        "disclaimer": "Not a promise of savings. Scenario math for pilot scoping.",
    }


def run_audit(
    *,
    checkpoint: str,
    encoder_name: str,
    documents_jsonl: Path,
    eval_jsonl: Path,
    k: int = 10,
    causal_graph: Optional[Path] = None,
    hyp_graph: Optional[Path] = None,
    rel_attrs: Optional[Path] = None,
    vertical: str = "generic",
    pack_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    from ablation_harness import _fixed_weight_classifier
    from causal_graph import CausalDocGraph
    from retrieval_engine import PSMRetrievalEngine
    from structure_index import RelationalAttrIndex, TaxonomyGraph

    encoder, ckpt, db, pipe = _ingest(checkpoint, encoder_name, documents_jsonl)
    eval_set = _eval_set(eval_jsonl, encoder, pipe)
    dense_stats, miss_idxs = _dense_rank_stats(eval_set, db, k_miss=k)

    causal_g = CausalDocGraph.from_jsonl(causal_graph) if causal_graph and causal_graph.exists() else None
    tax_g = TaxonomyGraph.from_jsonl(hyp_graph) if hyp_graph and hyp_graph.exists() else None
    rel_idx = RelationalAttrIndex.from_jsonl(rel_attrs) if rel_attrs and rel_attrs.exists() else None
    text_lookup = {r["document_id"]: str(r.get("chunk_text", "")) for r in db.rows}
    use_struct = any(x is not None for x in (causal_g, tax_g, rel_idx))

    def _make(use_struct_flag: bool, fusion: str = "zscore") -> PSMRetrievalEngine:
        return PSMRetrievalEngine(
            db,
            causal_matrix=ckpt["causal_matrix"],
            hard_truth_filter=False,
            causal_graph=causal_g if use_struct_flag else None,
            taxonomy_graph=tax_g if use_struct_flag else None,
            relational_index=rel_idx if use_struct_flag else None,
            stage1_dense_limit=50 if use_struct_flag else 100,
            stage1_graph_hops=2,
            graph_struct_bonus=4.0 if use_struct_flag else 0.0,
            fusion=fusion,
            rrf_k=20,
            doc_text_lookup=dict(text_lookup),
            query_conditioned_expansion=True,
            model_version=int(ckpt.get("model_version", 0)),
        )

    configs = [
        ("dense_only", False, np.array([1, 0, 0, 0, 0], dtype=np.float32), "zscore"),
        ("multi_channels", False, np.array([0.25, 0.25, 0.05, 0.20, 0.25], dtype=np.float32), "zscore"),
        ("multi+structure", True, np.array([0.25, 0.25, 0.05, 0.20, 0.25], dtype=np.float32), "zscore"),
    ]
    if not use_struct:
        configs = [c for c in configs if c[0] != "multi+structure"]

    engines = {}
    for name, struct, w, fusion in configs:
        eng = _make(struct and use_struct, fusion=fusion)
        eng.classifier.classify = _fixed_weight_classifier(w)
        engines[name] = eng

    full: dict[str, Any] = {}
    recovery: dict[str, Any] = {}
    per_query: list[dict[str, Any]] = []

    for name, eng in engines.items():
        hits = {1: 0, 5: 0, 10: 0}
        rr = []
        rec = {1: 0, 5: 0, 10: 0}
        for i, ex in enumerate(eval_set):
            ids = _search_ids(eng, ex, top_k=10)
            for kk in hits:
                if _hit(ids, ex.relevant_doc_ids, kk):
                    hits[kk] += 1
            rank = 0.0
            for j, did in enumerate(ids, start=1):
                if did in ex.relevant_doc_ids:
                    rank = 1.0 / j
                    break
            rr.append(rank)
            if i in miss_idxs:
                for kk in rec:
                    if _hit(ids, ex.relevant_doc_ids, kk):
                        rec[kk] += 1
        n = max(len(eval_set), 1)
        n_miss = max(len(miss_idxs), 1)
        full[name] = {
            "recall@1": hits[1] / n,
            "recall@5": hits[5] / n,
            "recall@10": hits[10] / n,
            "MRR": float(np.mean(rr)) if rr else 0.0,
        }
        recovery[name] = {
            "recovered@1": rec[1],
            "recovered@5": rec[5],
            "recovered@10": rec[10],
            "recovery@10": rec[10] / n_miss if miss_idxs else 0.0,
            "n_dense_miss": len(miss_idxs),
        }

    # Per-query detail vs dense vs best available multi
    best_multi = "multi+structure" if "multi+structure" in engines else "multi_channels"
    for i, ex in enumerate(eval_set):
        d_ids = _search_ids(engines["dense_only"], ex, 10)
        m_ids = _search_ids(engines[best_multi], ex, 10)
        silent = not _hit(d_ids, ex.relevant_doc_ids, k)  # dense silent fail
        per_query.append(
            {
                "query": ex.query_text,
                "gold": sorted(ex.relevant_doc_ids),
                "dense_rank": dense_stats[i]["dense_rank"],
                "dense_top5": dense_stats[i]["dense_top5"],
                "dense_hit@10": _hit(d_ids, ex.relevant_doc_ids, 10),
                "multi_top5": m_ids[:5],
                "multi_hit@10": _hit(m_ids, ex.relevant_doc_ids, 10),
                "silent_dense_fail": silent,
                "multi_recovers_silent_fail": silent and _hit(m_ids, ex.relevant_doc_ids, 10),
            }
        )

    n_silent = sum(1 for r in per_query if r["silent_dense_fail"])
    liability = _liability_note(vertical, n_silent, len(eval_set))

    return {
        "checkpoint": checkpoint,
        "encoder": encoder_name,
        "k": k,
        "n_documents": len(db.rows),
        "n_eval": len(eval_set),
        "n_dense_miss_at_k": len(miss_idxs),
        "structure_files": {
            "causal_graph": str(causal_graph) if causal_graph and causal_graph.exists() else None,
            "hyperbolic_graph": str(hyp_graph) if hyp_graph and hyp_graph.exists() else None,
            "relational_attrs": str(rel_attrs) if rel_attrs and rel_attrs.exists() else None,
        },
        "enabled_channels": ckpt.get("enabled_channels"),
        "pack_meta": pack_meta or {},
        "full_set_metrics": full,
        "recovery_on_dense_misses": recovery,
        "per_query": per_query,
        "liability_framing": liability,
        "honesty": {
            "checkpoint_trained_on": "VectorPrism finance adversarial demo (transfer test if pack differs)",
            "graphs_provided": use_struct,
            "claim": (
                "This audit measures dense vs multi on *this* corpus. "
                "Finance adversarial 13/13 recovery does not automatically transfer."
            ),
        },
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    liab = result["liability_framing"]
    lines = [
        "# VectorPrism corpus recovery audit",
        "",
        f"- Documents: **{result['n_documents']}**",
        f"- Eval queries: **{result['n_eval']}**",
        f"- Dense misses@{result['k']}: **{result['n_dense_miss_at_k']}**",
        f"- Encoder: `{result['encoder']}`",
        f"- Checkpoint: `{result['checkpoint']}`",
        f"- Structure indexes provided: **{bool(result['structure_files']['causal_graph'] or result['structure_files']['relational_attrs'] or result['structure_files']['hyperbolic_graph'])}**",
        "",
        "## Honesty",
        "",
        result["honesty"]["claim"],
        "",
        f"Pack: {json.dumps(result.get('pack_meta') or {}, ensure_ascii=False)}",
        "",
        "## Full-set metrics",
        "",
        "| Config | R@1 | R@5 | R@10 | MRR |",
        "|--------|-----|-----|------|-----|",
    ]
    for name, m in result["full_set_metrics"].items():
        lines.append(
            f"| {name} | {m['recall@1']:.3f} | {m['recall@5']:.3f} | {m['recall@10']:.3f} | {m['MRR']:.3f} |"
        )
    lines += [
        "",
        "## Recovery on dense misses",
        "",
        "| Config | recovered@10 | rate |",
        "|--------|--------------|------|",
    ]
    for name, m in result["recovery_on_dense_misses"].items():
        lines.append(
            f"| {name} | {m['recovered@10']}/{m['n_dense_miss']} | {m['recovery@10']:.1%} |"
        )
    lines += [
        "",
        "## Silent dense failures (queries)",
        "",
    ]
    silents = [q for q in result["per_query"] if q["silent_dense_fail"]]
    if not silents:
        lines.append("_No dense Miss@10 on this pack._")
    for q in silents:
        rec = "MULTI RECOVERED" if q["multi_recovers_silent_fail"] else "STILL MISS"
        lines.append(
            f"- **{rec}** dense_rank={q['dense_rank']} — `{q['query']}`  "
            f"gold={q['gold']} dense_top5={q['dense_top5']} multi_top5={q['multi_top5']}"
        )
    lines += [
        "",
        "## Liability / risk framing (illustrative)",
        "",
        f"- Failure mode: {liab['failure_mode']}",
        f"- Silent dense fail rate: **{liab['silent_fail_rate_pct']}%** ({liab['n_silent_fails_at_k']}/{liab['n_eval']})",
        f"- Unit cost assumption: **${liab['unit_cost_assumption_usd']:,.0f}** ({liab['unit_label']})",
        f"- {liab['note']}",
        f"- Disclaimer: {liab['disclaimer']}",
        "",
        "## Next step",
        "",
        "Treat this as a **paid pilot audit** input: identify silent fails, add structure indexes "
        "(causal/rel/hyp), retrain channels on *their* labels, re-measure recovery.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="VectorPrism corpus recovery audit (partner/public packs)")
    ap.add_argument("--documents", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--causal-graph", type=Path, default=None)
    ap.add_argument("--hyperbolic-graph", type=Path, default=None)
    ap.add_argument("--relational-attrs", type=Path, default=None)
    ap.add_argument(
        "--vertical",
        default="generic",
        choices=["generic", "sre_incident", "finance_compliance", "healthcare"],
    )
    ap.add_argument("--pack-meta", type=Path, default=None)
    args = ap.parse_args(argv)

    meta = {}
    if args.pack_meta and args.pack_meta.exists():
        meta = json.loads(args.pack_meta.read_text(encoding="utf-8"))

    result = run_audit(
        checkpoint=args.checkpoint,
        encoder_name=args.encoder,
        documents_jsonl=args.documents,
        eval_jsonl=args.eval,
        k=args.k,
        causal_graph=args.causal_graph,
        hyp_graph=args.hyperbolic_graph,
        rel_attrs=args.relational_attrs,
        vertical=args.vertical,
        pack_meta=meta,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "corpus_recovery_audit.json"
    md_path = args.out / "CORPUS_RECOVERY_AUDIT.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(f"dense_miss@{args.k}={result['n_dense_miss_at_k']}/{result['n_eval']}")
    for name, m in result["full_set_metrics"].items():
        print(f"{name}: R@10={m['recall@10']:.3f} MRR={m['MRR']:.3f}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
