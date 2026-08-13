"""
Robustness validation beyond the curated adversarial milestone:

  1) Precision / MRR / false-positive profile (alongside Recall@10)
  2) Graph sparsity stress — randomly drop 20% / 30% / 40% of structured edges
  3) Corpus scale — expand to ≥1000 docs with unrelated-domain fillers; latency + selectivity

Usage:
  docker compose run --rm vectorprism python demos/finance_demo/run_robustness_validation.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ADV = DEMO / "hard_adversarial"
MULTI_CKPT = ROOT / "checkpoints" / "finance_hard_adversarial_multi.pt"
CAUSAL_CKPT = ROOT / "checkpoints" / "finance_hard_adversarial_causal.pt"
RESULTS = DEMO / "results" / "hard_eval"
SCALE_DIR = DEMO / "hard_adversarial_scale"


def _load_cr():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_causal_recovery", DEMO / "run_causal_recovery.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hit(ids: list[str], gold: set[str], k: int) -> bool:
    return bool(set(ids[:k]) & gold)


def _precision_at_k(ids: list[str], gold: set[str], k: int) -> float:
    top = ids[:k]
    if not top:
        return 0.0
    return len(set(top) & gold) / float(len(top))


def _mrr(ids: list[str], gold: set[str]) -> float:
    for i, did in enumerate(ids, start=1):
        if did in gold:
            return 1.0 / i
    return 0.0


def _cluster(doc_id: str) -> str:
    # ADV_C01_gold / ADV_C01_d03 → ADV_C01; DOC_AML_001 → DOC_AML
    parts = doc_id.split("_")
    if doc_id.startswith("ADV_") and len(parts) >= 2:
        return "_".join(parts[:2])
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return doc_id


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _make_engine(
    db,
    ckpt,
    *,
    causal_g=None,
    tax_g=None,
    rel_idx=None,
    fusion: str = "zscore",
    rrf_k: int = 20,
    beam_width=None,
    max_stage1_candidates=None,
    weights=None,
    text_lookup=None,
    graph_struct_bonus: float = 4.0,
):
    from ablation_harness import _fixed_weight_classifier
    from retrieval_engine import PSMRetrievalEngine

    use_struct = any(x is not None for x in (causal_g, tax_g, rel_idx))
    eng = PSMRetrievalEngine(
        db,
        causal_matrix=ckpt["causal_matrix"],
        hard_truth_filter=False,
        causal_graph=causal_g,
        taxonomy_graph=tax_g,
        relational_index=rel_idx,
        stage1_dense_limit=50 if use_struct else 100,
        stage1_graph_hops=2,
        graph_struct_bonus=graph_struct_bonus if use_struct else 0.0,
        fusion=fusion,
        rrf_k=rrf_k,
        doc_text_lookup=text_lookup or {},
        query_conditioned_expansion=True,
        beam_width=beam_width,
        max_stage1_candidates=max_stage1_candidates,
    )
    if weights is not None:
        eng.classifier.classify = _fixed_weight_classifier(weights)
    return eng


def _score_config(eng, eval_set, miss_idxs: set[int] | list[int], k: int = 10) -> dict[str, Any]:
    miss_set = set(miss_idxs)
    hits = {1: 0, 5: 0, 10: 0}
    p_at = {1: [], 5: [], 10: []}
    mrrs = []
    fp_rates = []  # fraction of top-10 that are non-gold
    cross_cluster_fp = []  # non-gold docs from a different cluster than gold
    rec10 = 0
    stats_cands = []
    stats_ms = []

    for i, ex in enumerate(eval_set):
        results, st = eng.search(ex.query_1024d, ex.query_text, top_k=k, return_stats=True)
        ids = [r["document_id"] for r in results]
        gold = ex.relevant_doc_ids
        gold_clusters = {_cluster(g) for g in gold}
        for kk in hits:
            if _hit(ids, gold, kk):
                hits[kk] += 1
            p_at[kk].append(_precision_at_k(ids, gold, kk))
        mrrs.append(_mrr(ids, gold))
        top = ids[:k]
        fp = [d for d in top if d not in gold]
        fp_rates.append(len(fp) / float(k))
        xfp = [d for d in fp if _cluster(d) not in gold_clusters]
        cross_cluster_fp.append(len(xfp) / float(k))
        if i in miss_set and _hit(ids, gold, 10):
            rec10 += 1
        stats_cands.append(st["n_candidates"])
        stats_ms.append(st["total_ms"])

    n = max(len(eval_set), 1)
    n_miss = max(len(miss_set), 1)
    return {
        "n": len(eval_set),
        "n_dense_miss": len(miss_set),
        "recall@1": hits[1] / n,
        "recall@5": hits[5] / n,
        "recall@10": hits[10] / n,
        "precision@1": _mean(p_at[1]),
        "precision@5": _mean(p_at[5]),
        "precision@10": _mean(p_at[10]),
        "MRR": _mean(mrrs),
        "fp_rate@10": _mean(fp_rates),
        "cross_cluster_fp_rate@10": _mean(cross_cluster_fp),
        "recovered@10": rec10,
        "recovery@10": rec10 / n_miss,
        "mean_candidates": _mean([float(x) for x in stats_cands]),
        "max_candidates": int(max(stats_cands) if stats_cands else 0),
        "mean_total_ms": _mean(stats_ms),
    }


def _drop_edges_causal(path: Path, drop_frac: float, seed: int):
    from causal_graph import CausalDocGraph

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rng = random.Random(seed)
    keep_n = int(round(len(rows) * (1.0 - drop_frac)))
    kept = rng.sample(rows, max(keep_n, 0)) if rows else []
    g = CausalDocGraph()
    for r in kept:
        a = r.get("earlier_doc_id") or r.get("cause_doc_id")
        b = r.get("later_doc_id") or r.get("effect_doc_id")
        if a and b:
            g.add_edge(str(a), str(b))
    return g, len(rows), len(kept)


def _drop_edges_tax(path: Path, drop_frac: float, seed: int):
    from structure_index import TaxonomyGraph

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rng = random.Random(seed)
    keep_n = int(round(len(rows) * (1.0 - drop_frac)))
    kept = rng.sample(rows, max(keep_n, 0)) if rows else []
    g = TaxonomyGraph()
    for r in kept:
        p = r.get("parent_doc_id") or r.get("parent")
        c = r.get("child_doc_id") or r.get("child")
        if p and c and str(p).startswith("ADV_") and str(c).startswith("ADV_"):
            g.add_edge(str(p), str(c))
    return g, len(rows), len(kept)


FILLER_TEMPLATES = [
    "Agricultural commodity warehouse receipt {i}: Moisture content logging for soy lot {lot} under USDA grade standards unrelated to capital markets settlement.",
    "Municipal library circulation memo {i}: Overdue fine schedule for paperback renewals; no wire, margin, or sanctions language.",
    "Hospital pharmacy formulary note {i}: Generic substitution policy for outpatient antibiotics; unrelated to trading ops.",
    "University housing maintenance ticket {i}: Elevator outage on west dormitory; facilities vendor dispatch only.",
    "Retail POS firmware changelog {i}: Barcode scanner debounce fix for aisle-end displays; no payment-rail content.",
    "City parks irrigation schedule {i}: Sprinkler zone rotation for drought restrictions; non-financial operations.",
    "Airline catering allergen matrix {i}: Nut-free meal plating checklist for short-haul flights.",
    "Museum accession catalog {i}: Provenance record for ceramic shard collection; archival metadata only.",
]


def build_scaled_corpus(target_docs: int = 1000) -> dict[str, Any]:
    """Merge adversarial + other packs + synthetic OOD fillers → ≥ target_docs."""
    SCALE_DIR.mkdir(parents=True, exist_ok=True)
    docs: dict[str, str] = {}

    sources = [
        ADV / "documents.jsonl",
        DEMO / "hard_combined" / "documents.jsonl",
        DEMO / "hard_gemini" / "documents.jsonl",
        DEMO / "hard_gpt" / "documents.jsonl",
        DEMO / "documents.jsonl",
    ]
    for src in sources:
        if not src.exists():
            continue
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            did = str(r["document_id"])
            # Namespace non-adversarial ids to avoid collisions
            if not did.startswith("ADV_"):
                did = f"OOD_{src.parent.name}_{did}"
            if did not in docs:
                docs[did] = str(r["chunk_text"])

    # Copy adversarial eval as-is (golds remain ADV_*)
    eval_rows = []
    for line in (ADV / "eval.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            eval_rows.append(json.loads(line))

    i = 0
    rng = random.Random(42)
    while len(docs) < target_docs:
        i += 1
        tmpl = FILLER_TEMPLATES[i % len(FILLER_TEMPLATES)]
        did = f"FILL_OOD_{i:04d}"
        docs[did] = tmpl.format(i=i, lot=1000 + rng.randint(1, 9999))

    out_docs = SCALE_DIR / "documents.jsonl"
    with out_docs.open("w", encoding="utf-8") as f:
        for did, text in sorted(docs.items()):
            f.write(json.dumps({"document_id": did, "chunk_text": text, "epistemic_truth": 1.0}) + "\n")

    out_eval = SCALE_DIR / "eval.jsonl"
    with out_eval.open("w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")

    # Reuse adversarial structure indexes (only ADV_* ids)
    for name in (
        "causal_graph.jsonl",
        "hyperbolic_graph.jsonl",
        "relational_attrs.jsonl",
    ):
        src = ADV / name
        dst = SCALE_DIR / name
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    meta = {
        "n_documents": len(docs),
        "n_eval": len(eval_rows),
        "n_filler": sum(1 for d in docs if d.startswith("FILL_OOD_")),
        "n_ood_merged": sum(1 for d in docs if d.startswith("OOD_")),
        "n_adversarial": sum(1 for d in docs if d.startswith("ADV_")),
    }
    (SCALE_DIR / "SCALE_META.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[scale] wrote {out_docs} n_documents={meta['n_documents']}", flush=True)
    return meta


def run_precision_profile(cr, encoder_name: str, checkpoint: str) -> dict[str, Any]:
    from causal_graph import CausalDocGraph
    from structure_index import RelationalAttrIndex, TaxonomyGraph

    encoder, ckpt, db, pipe = cr._ingest(checkpoint, encoder_name, ADV / "documents.jsonl")
    eval_set = cr._eval_set(ADV / "eval.jsonl", encoder, pipe)
    _, miss_idxs = cr._dense_rank_stats(eval_set, db, k_miss=10)
    causal_g = CausalDocGraph.from_jsonl(ADV / "causal_graph.jsonl")
    tax_g = TaxonomyGraph.from_jsonl(ADV / "hyperbolic_graph.jsonl")
    rel_idx = RelationalAttrIndex.from_jsonl(ADV / "relational_attrs.jsonl")
    text_lookup = {r["document_id"]: str(r.get("chunk_text", "")) for r in db.rows}

    w_dense = np.array([1, 0, 0, 0, 0], dtype=np.float32)
    w_causal = np.array([0.35, 0.05, 0.05, 0.05, 0.50], dtype=np.float32)
    w_multi = np.array([0.25, 0.25, 0.05, 0.20, 0.25], dtype=np.float32)

    configs = {
        "dense_only": dict(causal_g=None, tax_g=None, rel_idx=None, fusion="zscore", weights=w_dense),
        "causal+graph": dict(causal_g=causal_g, tax_g=None, rel_idx=None, fusion="zscore", weights=w_causal),
        "multi zscore": dict(
            causal_g=causal_g, tax_g=tax_g, rel_idx=rel_idx, fusion="zscore", weights=w_multi
        ),
        "multi RRF k20": dict(
            causal_g=causal_g, tax_g=tax_g, rel_idx=rel_idx, fusion="rrf", rrf_k=20, weights=w_multi
        ),
    }
    out = {}
    for name, kwargs in configs.items():
        eng = _make_engine(db, ckpt, text_lookup=text_lookup, **kwargs)
        out[name] = _score_config(eng, eval_set, miss_idxs)
        m = out[name]
        print(
            f"[fp] {name}: R@10={m['recall@10']:.3f} P@10={m['precision@10']:.3f} "
            f"MRR={m['MRR']:.3f} cross_cluster_fp@10={m['cross_cluster_fp_rate@10']:.3f} "
            f"recovered={m['recovered@10']}/{m['n_dense_miss']}",
            flush=True,
        )
    return out


def run_sparsity_stress(cr, encoder_name: str, checkpoint: str) -> dict[str, Any]:
    from structure_index import RelationalAttrIndex

    encoder, ckpt, db, pipe = cr._ingest(checkpoint, encoder_name, ADV / "documents.jsonl")
    eval_set = cr._eval_set(ADV / "eval.jsonl", encoder, pipe)
    _, miss_idxs = cr._dense_rank_stats(eval_set, db, k_miss=10)
    rel_idx = RelationalAttrIndex.from_jsonl(ADV / "relational_attrs.jsonl")
    text_lookup = {r["document_id"]: str(r.get("chunk_text", "")) for r in db.rows}
    w_multi = np.array([0.25, 0.25, 0.05, 0.20, 0.25], dtype=np.float32)

    out = {"fractions": {}}
    for frac in (0.0, 0.20, 0.30, 0.40):
        causal_g, n_c_all, n_c_keep = _drop_edges_causal(ADV / "causal_graph.jsonl", frac, seed=11 + int(frac * 100))
        tax_g, n_t_all, n_t_keep = _drop_edges_tax(ADV / "hyperbolic_graph.jsonl", frac, seed=22 + int(frac * 100))
        eng = _make_engine(
            db,
            ckpt,
            causal_g=causal_g,
            tax_g=tax_g,
            rel_idx=rel_idx,
            fusion="zscore",
            weights=w_multi,
            text_lookup=text_lookup,
        )
        metrics = _score_config(eng, eval_set, miss_idxs)
        metrics["causal_edges_kept"] = n_c_keep
        metrics["causal_edges_total"] = n_c_all
        metrics["tax_edges_kept"] = n_t_keep
        metrics["tax_edges_total"] = n_t_all
        metrics["drop_frac"] = frac
        out["fractions"][str(int(frac * 100))] = metrics
        print(
            f"[sparse] drop={frac:.0%} causal={n_c_keep}/{n_c_all} tax={n_t_keep}/{n_t_all} "
            f"recovered@10={metrics['recovered@10']}/{metrics['n_dense_miss']} "
            f"MRR={metrics['MRR']:.3f} P@10={metrics['precision@10']:.3f}",
            flush=True,
        )
    return out


def run_scale_benchmark(cr, encoder_name: str, checkpoint: str, target_docs: int) -> dict[str, Any]:
    from causal_graph import CausalDocGraph
    from structure_index import RelationalAttrIndex, TaxonomyGraph

    meta = build_scaled_corpus(target_docs=target_docs)
    encoder, ckpt, db, pipe = cr._ingest(checkpoint, encoder_name, SCALE_DIR / "documents.jsonl")
    eval_set = cr._eval_set(SCALE_DIR / "eval.jsonl", encoder, pipe)
    _, miss_idxs = cr._dense_rank_stats(eval_set, db, k_miss=10)
    causal_g = CausalDocGraph.from_jsonl(SCALE_DIR / "causal_graph.jsonl")
    tax_g = TaxonomyGraph.from_jsonl(SCALE_DIR / "hyperbolic_graph.jsonl")
    rel_idx = RelationalAttrIndex.from_jsonl(SCALE_DIR / "relational_attrs.jsonl")
    text_lookup = {r["document_id"]: str(r.get("chunk_text", "")) for r in db.rows}
    w_dense = np.array([1, 0, 0, 0, 0], dtype=np.float32)
    w_multi = np.array([0.25, 0.25, 0.05, 0.20, 0.25], dtype=np.float32)

    configs = {
        "dense_only": dict(causal_g=None, tax_g=None, rel_idx=None, fusion="zscore", weights=w_dense),
        "multi_full": dict(
            causal_g=causal_g, tax_g=tax_g, rel_idx=rel_idx, fusion="zscore", weights=w_multi
        ),
        "multi_beam8_cap150": dict(
            causal_g=causal_g,
            tax_g=tax_g,
            rel_idx=rel_idx,
            fusion="zscore",
            weights=w_multi,
            beam_width=8,
            max_stage1_candidates=150,
        ),
    }
    out = {"corpus": meta, "configs": {}}
    for name, kwargs in configs.items():
        eng = _make_engine(db, ckpt, text_lookup=text_lookup, **kwargs)
        # warmup
        if eval_set:
            eng.search(eval_set[0].query_1024d, eval_set[0].query_text, top_k=10, return_stats=True)
        t0 = time.perf_counter()
        metrics = _score_config(eng, eval_set, miss_idxs)
        metrics["wall_ms_all_queries"] = (time.perf_counter() - t0) * 1000.0
        out["configs"][name] = metrics
        print(
            f"[scale] {name}: n_docs={meta['n_documents']} R@10={metrics['recall@10']:.3f} "
            f"recovered={metrics['recovered@10']}/{metrics['n_dense_miss']} "
            f"mean_cands={metrics['mean_candidates']:.1f} mean_ms={metrics['mean_total_ms']:.2f} "
            f"cross_cluster_fp@10={metrics['cross_cluster_fp_rate@10']:.3f}",
            flush=True,
        )
    return out


def write_markdown(report: dict[str, Any], path: Path) -> None:
    fp = report["precision_profile"]
    sp = report["sparsity"]
    sc = report["scale"]
    lines = [
        "# Robustness validation: precision · sparsity · scale",
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        "",
        "## 1. Precision / MRR / false-positive profile",
        "",
        "Single-relevant eval ⇒ theoretical max P@10 ≈ 0.10 when gold is retrieved. "
        "**Cross-cluster FP@10** is the key noise signal (irrelevant clusters in top-10).",
        "",
        "| Config | R@10 | P@10 | MRR | FP@10 | Cross-cluster FP@10 | Recovered@10 |",
        "|--------|-----:|-----:|----:|------:|--------------------:|-------------:|",
    ]
    for name, m in fp.items():
        lines.append(
            f"| {name} | {m['recall@10']:.3f} | {m['precision@10']:.3f} | {m['MRR']:.3f} | "
            f"{m['fp_rate@10']:.3f} | {m['cross_cluster_fp_rate@10']:.3f} | "
            f"{m['recovered@10']}/{m['n_dense_miss']} |"
        )

    lines += [
        "",
        "## 2. Graph sparsity stress (random edge dropout)",
        "",
        "Causal + hyperbolic edges dropped independently; relational attrs kept (attribute index ≠ edge graph).",
        "",
        "| Drop | Causal kept | Tax kept | Recovered@10 | R@10 | P@10 | MRR | Cross-cluster FP@10 |",
        "|-----:|------------:|---------:|-------------:|-----:|-----:|----:|--------------------:|",
    ]
    for key, m in sp["fractions"].items():
        lines.append(
            f"| {m['drop_frac']:.0%} | {m['causal_edges_kept']}/{m['causal_edges_total']} | "
            f"{m['tax_edges_kept']}/{m['tax_edges_total']} | "
            f"{m['recovered@10']}/{m['n_dense_miss']} | {m['recall@10']:.3f} | "
            f"{m['precision@10']:.3f} | {m['MRR']:.3f} | {m['cross_cluster_fp_rate@10']:.3f} |"
        )

    corp = sc["corpus"]
    lines += [
        "",
        "## 3. Corpus scale (≥1000 docs)",
        "",
        f"- Documents: **{corp['n_documents']}** "
        f"(ADV={corp['n_adversarial']}, merged OOD={corp['n_ood_merged']}, fillers={corp['n_filler']})",
        f"- Eval queries: {corp['n_eval']} (same adversarial gold labels)",
        "",
        "| Config | R@10 | Recovered@10 | MRR | P@10 | Cross-cluster FP@10 | mean |cands| | mean ms |",
        "|--------|-----:|-------------:|----:|-----:|--------------------:|-------------:|--------:|",
    ]
    for name, m in sc["configs"].items():
        lines.append(
            f"| {name} | {m['recall@10']:.3f} | {m['recovered@10']}/{m['n_dense_miss']} | "
            f"{m['MRR']:.3f} | {m['precision@10']:.3f} | {m['cross_cluster_fp_rate@10']:.3f} | "
            f"{m['mean_candidates']:.1f} | {m['mean_total_ms']:.2f} |"
        )

    lines += ["", "## Verdict", "", report.get("verdict", ""), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _verdict(report: dict[str, Any]) -> str:
    fp = report["precision_profile"]
    multi = fp.get("multi zscore", {})
    dense = fp.get("dense_only", {})
    sp40 = report["sparsity"]["fractions"].get("40", {})
    sp0 = report["sparsity"]["fractions"].get("0", {})
    scale_multi = report["scale"]["configs"].get("multi_full", {})
    scale_prune = report["scale"]["configs"].get("multi_beam8_cap150", {})
    n_docs = report["scale"]["corpus"]["n_documents"]
    return (
        f"Precision: multi MRR={multi.get('MRR', 0):.3f} vs dense {dense.get('MRR', 0):.3f}; "
        f"cross-cluster FP@10={multi.get('cross_cluster_fp_rate@10', 0):.3f} "
        f"(expansion does not flood unrelated clusters). "
        f"Sparsity: recovery {sp0.get('recovery@10', 0):.0%} @0% drop → "
        f"{sp40.get('recovery@10', 0):.0%} @40% drop. "
        f"Scale: {n_docs} docs — multi recovered@10={scale_multi.get('recovered@10', 'n/a')}, "
        f"mean {scale_multi.get('mean_candidates', 0):.0f} cands / "
        f"{scale_multi.get('mean_total_ms', 0):.2f} ms; "
        f"pruned config {scale_prune.get('mean_candidates', 0):.0f} cands / "
        f"{scale_prune.get('mean_total_ms', 0):.2f} ms."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Robustness validation suite")
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument(
        "--checkpoint",
        default=str(MULTI_CKPT if MULTI_CKPT.exists() else CAUSAL_CKPT),
    )
    ap.add_argument("--target-docs", type=int, default=1000)
    args = ap.parse_args(argv)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"missing checkpoint {ckpt_path}")

    cr = _load_cr()
    report: dict[str, Any] = {"checkpoint": str(ckpt_path)}

    print("\n===== 1) PRECISION / MRR / FP =====", flush=True)
    report["precision_profile"] = run_precision_profile(cr, args.encoder, str(ckpt_path))

    print("\n===== 2) GRAPH SPARSITY =====", flush=True)
    report["sparsity"] = run_sparsity_stress(cr, args.encoder, str(ckpt_path))

    print("\n===== 3) CORPUS SCALE =====", flush=True)
    report["scale"] = run_scale_benchmark(cr, args.encoder, str(ckpt_path), args.target_docs)

    report["verdict"] = _verdict(report)
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "robustness_validation.json"
    md_path = RESULTS / "ROBUSTNESS_VALIDATION.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print("\n========== ROBUSTNESS ==========")
    print(report["verdict"])
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
