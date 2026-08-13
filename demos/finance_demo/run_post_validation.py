"""
Post-validation suite after multi-channel recovery ≥70%:

  1) OOD transfer — score adversarial multi ckpt on non-adversarial packs
  2) Graph-density stress — inject cross-cluster noise edges; measure P@10 / R@10
  3) Latency & pruning — Stage-1 candidate sizes + ms under beam / max_candidates
  4) RRF k sweep — k ∈ {20,30,40,50,60} vs z-score on adversarial dense misses

Usage:
  docker compose run --rm vectorprism python demos/finance_demo/run_post_validation.py
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

OOD_PACKS = {
    "easy_finance": DEMO,
    "hard_gemini": DEMO / "hard_gemini",
    "hard_gpt": DEMO / "hard_gpt",
    "hard_combined": DEMO / "hard_combined",
}


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


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _build_noisy_causal_graph(base_path: Path, noise_edges: int, seed: int = 0):
    from causal_graph import CausalDocGraph

    g = CausalDocGraph.from_jsonl(base_path)
    rng = random.Random(seed)
    nodes = sorted(set(g.downstream) | set(g.upstream) | {n for s in g.downstream.values() for n in s})
    if len(nodes) < 4 or noise_edges <= 0:
        return g, g.n_edges
    added = 0
    attempts = 0
    while added < noise_edges and attempts < noise_edges * 20:
        attempts += 1
        a, b = rng.sample(nodes, 2)
        # Cross-cluster only (different ADV_* prefix)
        pa = a.rsplit("_", 1)[0]
        pb = b.rsplit("_", 1)[0]
        if pa == pb:
            continue
        before = g.n_edges
        g.add_edge(a, b)
        if g.n_edges > before:
            added += 1
    return g, g.n_edges


def _make_engine(
    db,
    ckpt,
    *,
    causal_g=None,
    tax_g=None,
    rel_idx=None,
    fusion: str = "zscore",
    rrf_k: int = 60,
    beam_width=None,
    max_stage1_candidates=None,
    weights=None,
    text_lookup=None,
):
    from ablation_harness import _fixed_weight_classifier
    from retrieval_engine import PSMRetrievalEngine

    eng = PSMRetrievalEngine(
        db,
        causal_matrix=ckpt["causal_matrix"],
        hard_truth_filter=False,
        causal_graph=causal_g,
        taxonomy_graph=tax_g,
        relational_index=rel_idx,
        stage1_dense_limit=50 if any(x is not None for x in (causal_g, tax_g, rel_idx)) else 100,
        stage1_graph_hops=2,
        graph_struct_bonus=4.0 if causal_g is not None or tax_g is not None else 0.0,
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


def run_ood_transfer(cr, encoder_name: str, checkpoint: str) -> dict[str, Any]:
    """Score adversarial-trained multi ckpt on OOD packs (no pack-specific graphs)."""
    out = {}
    # dense, rel, dis, hyp, causal
    w_multi = np.array([0.25, 0.25, 0.05, 0.20, 0.25], dtype=np.float32)
    w_dense = np.array([1, 0, 0, 0, 0], dtype=np.float32)

    for name, pack_dir in OOD_PACKS.items():
        docs = pack_dir / "documents.jsonl"
        ev = pack_dir / "eval.jsonl"
        if not docs.exists() or not ev.exists():
            out[name] = {"error": "missing documents/eval"}
            continue
        encoder, ckpt, db, pipe = cr._ingest(checkpoint, encoder_name, docs)
        eval_set = cr._eval_set(ev, encoder, pipe)
        eng_d = _make_engine(db, ckpt, fusion="zscore", weights=w_dense)
        eng_m = _make_engine(db, ckpt, fusion="zscore", weights=w_multi)
        # restore intent for router baseline
        eng_i = _make_engine(db, ckpt, fusion="zscore", weights=None)

        rows = {}
        for label, eng in [("dense_only", eng_d), ("multi_zscore_no_graph", eng_m), ("intent_no_graph", eng_i)]:
            hits10 = 0
            p10 = []
            for ex in eval_set:
                ids = [r["document_id"] for r in eng.search(ex.query_1024d, ex.query_text, top_k=10)]
                if _hit(ids, ex.relevant_doc_ids, 10):
                    hits10 += 1
                p10.append(_precision_at_k(ids, ex.relevant_doc_ids, 10))
            n = max(len(eval_set), 1)
            rows[label] = {
                "n": len(eval_set),
                "recall@10": hits10 / n,
                "precision@10": _mean(p10),
            }
        # delta: does multi fusion hurt precision vs dense on soft packs?
        rows["delta_R@10_multi_minus_dense"] = (
            rows["multi_zscore_no_graph"]["recall@10"] - rows["dense_only"]["recall@10"]
        )
        rows["delta_P@10_multi_minus_dense"] = (
            rows["multi_zscore_no_graph"]["precision@10"] - rows["dense_only"]["precision@10"]
        )
        out[name] = rows
        print(
            f"[ood] {name}: dense R@10={rows['dense_only']['recall@10']:.3f} "
            f"multi R@10={rows['multi_zscore_no_graph']['recall@10']:.3f} "
            f"ΔP@10={rows['delta_P@10_multi_minus_dense']:+.3f}",
            flush=True,
        )
    return out


def run_graph_density_stress(cr, encoder_name: str, checkpoint: str) -> dict[str, Any]:
    """Inject cross-cluster causal noise; measure recovery + precision on adversarial."""
    from causal_graph import CausalDocGraph
    from structure_index import RelationalAttrIndex, TaxonomyGraph

    encoder, ckpt, db, pipe = cr._ingest(checkpoint, encoder_name, ADV / "documents.jsonl")
    eval_set = cr._eval_set(ADV / "eval.jsonl", encoder, pipe)
    _, miss_idxs = cr._dense_rank_stats(eval_set, db, k_miss=10)
    tax_g = TaxonomyGraph.from_jsonl(ADV / "hyperbolic_graph.jsonl")
    rel_idx = RelationalAttrIndex.from_jsonl(ADV / "relational_attrs.jsonl")
    text_lookup = {r["document_id"]: str(r.get("chunk_text", "")) for r in db.rows}
    w = np.array([0.25, 0.25, 0.05, 0.20, 0.25], dtype=np.float32)

    base_edges = CausalDocGraph.from_jsonl(ADV / "causal_graph.jsonl").n_edges
    noise_levels = [0, 50, 150, 400]
    levels = {}

    for noise in noise_levels:
        g, n_edges = _build_noisy_causal_graph(ADV / "causal_graph.jsonl", noise, seed=noise + 7)
        eng = _make_engine(
            db,
            ckpt,
            causal_g=g,
            tax_g=tax_g,
            rel_idx=rel_idx,
            fusion="zscore",
            weights=w,
            text_lookup=text_lookup,
        )
        hits = 0
        rec = 0
        p_all = []
        for i, ex in enumerate(eval_set):
            ids = [r["document_id"] for r in eng.search(ex.query_1024d, ex.query_text, top_k=10)]
            p_all.append(_precision_at_k(ids, ex.relevant_doc_ids, 10))
            h = _hit(ids, ex.relevant_doc_ids, 10)
            if h:
                hits += 1
            if i in miss_idxs and h:
                rec += 1
        levels[str(noise)] = {
            "noise_edges_requested": noise,
            "total_causal_edges": n_edges,
            "recovered@10": rec,
            "recovery@10": rec / max(len(miss_idxs), 1),
            "precision@10_full": _mean(p_all),
            "recall@10_full": hits / max(len(eval_set), 1),
        }
        print(
            f"[density] noise={noise} edges={n_edges} recovered@10={rec}/{len(miss_idxs)} "
            f"P@10={levels[str(noise)]['precision@10_full']:.3f}",
            flush=True,
        )
    return {"base_causal_edges": base_edges, "levels": levels}


def run_latency_pruning(cr, encoder_name: str, checkpoint: str) -> dict[str, Any]:
    from causal_graph import CausalDocGraph
    from structure_index import RelationalAttrIndex, TaxonomyGraph

    encoder, ckpt, db, pipe = cr._ingest(checkpoint, encoder_name, ADV / "documents.jsonl")
    eval_set = cr._eval_set(ADV / "eval.jsonl", encoder, pipe)
    # denser graph for latency stress
    g_dense, n_edges = _build_noisy_causal_graph(ADV / "causal_graph.jsonl", 400, seed=99)
    tax_g = TaxonomyGraph.from_jsonl(ADV / "hyperbolic_graph.jsonl")
    rel_idx = RelationalAttrIndex.from_jsonl(ADV / "relational_attrs.jsonl")
    text_lookup = {r["document_id"]: str(r.get("chunk_text", "")) for r in db.rows}
    w = np.array([0.25, 0.25, 0.05, 0.20, 0.25], dtype=np.float32)
    _, miss_idxs = cr._dense_rank_stats(eval_set, db, k_miss=10)

    configs = [
        {"name": "baseline_no_cap", "beam_width": None, "max_stage1_candidates": None},
        {"name": "beam8", "beam_width": 8, "max_stage1_candidates": None},
        {"name": "beam4", "beam_width": 4, "max_stage1_candidates": None},
        {"name": "cap200", "beam_width": None, "max_stage1_candidates": 200},
        {"name": "cap100", "beam_width": None, "max_stage1_candidates": 100},
        {"name": "beam8_cap150", "beam_width": 8, "max_stage1_candidates": 150},
        {"name": "clean_graph_baseline", "beam_width": None, "max_stage1_candidates": None, "clean": True},
    ]
    out = {"noisy_graph_edges": n_edges, "configs": {}}

    for cfg in configs:
        causal = (
            CausalDocGraph.from_jsonl(ADV / "causal_graph.jsonl")
            if cfg.get("clean")
            else g_dense
        )
        eng = _make_engine(
            db,
            ckpt,
            causal_g=causal,
            tax_g=tax_g,
            rel_idx=rel_idx,
            fusion="zscore",
            weights=w,
            text_lookup=text_lookup,
            beam_width=cfg["beam_width"],
            max_stage1_candidates=cfg["max_stage1_candidates"],
        )
        cands = []
        s1 = []
        s2 = []
        tot = []
        rec = 0
        # warmup
        if eval_set:
            eng.search(eval_set[0].query_1024d, eval_set[0].query_text, top_k=10, return_stats=True)
        t_wall0 = time.perf_counter()
        for i, ex in enumerate(eval_set):
            ids_stats = eng.search(ex.query_1024d, ex.query_text, top_k=10, return_stats=True)
            ids, st = ids_stats
            ids = [r["document_id"] for r in ids]
            cands.append(st["n_candidates"])
            s1.append(st["stage1_ms"])
            s2.append(st["stage2_ms"])
            tot.append(st["total_ms"])
            if i in miss_idxs and _hit(ids, ex.relevant_doc_ids, 10):
                rec += 1
        wall = (time.perf_counter() - t_wall0) * 1000.0
        out["configs"][cfg["name"]] = {
            "beam_width": cfg["beam_width"],
            "max_stage1_candidates": cfg["max_stage1_candidates"],
            "mean_candidates": _mean([float(x) for x in cands]),
            "max_candidates": int(max(cands) if cands else 0),
            "mean_stage1_ms": _mean(s1),
            "mean_stage2_ms": _mean(s2),
            "mean_total_ms": _mean(tot),
            "wall_ms_all_queries": wall,
            "recovered@10": rec,
            "recovery@10": rec / max(len(miss_idxs), 1),
        }
        print(
            f"[latency] {cfg['name']}: mean_cands={out['configs'][cfg['name']]['mean_candidates']:.1f} "
            f"mean_total_ms={out['configs'][cfg['name']]['mean_total_ms']:.2f} "
            f"recovered@10={rec}/{len(miss_idxs)}",
            flush=True,
        )
    return out


def run_rrf_k_sweep(cr, encoder_name: str, checkpoint: str) -> dict[str, Any]:
    from causal_graph import CausalDocGraph
    from structure_index import RelationalAttrIndex, TaxonomyGraph

    encoder, ckpt, db, pipe = cr._ingest(checkpoint, encoder_name, ADV / "documents.jsonl")
    eval_set = cr._eval_set(ADV / "eval.jsonl", encoder, pipe)
    _, miss_idxs = cr._dense_rank_stats(eval_set, db, k_miss=10)
    causal_g = CausalDocGraph.from_jsonl(ADV / "causal_graph.jsonl")
    tax_g = TaxonomyGraph.from_jsonl(ADV / "hyperbolic_graph.jsonl")
    rel_idx = RelationalAttrIndex.from_jsonl(ADV / "relational_attrs.jsonl")
    text_lookup = {r["document_id"]: str(r.get("chunk_text", "")) for r in db.rows}
    w = np.array([0.25, 0.25, 0.05, 0.20, 0.25], dtype=np.float32)

    results = {}
    # z-score baseline
    eng_z = _make_engine(
        db,
        ckpt,
        causal_g=causal_g,
        tax_g=tax_g,
        rel_idx=rel_idx,
        fusion="zscore",
        weights=w,
        text_lookup=text_lookup,
    )
    rec = 0
    hits = 0
    for i, ex in enumerate(eval_set):
        ids = [r["document_id"] for r in eng_z.search(ex.query_1024d, ex.query_text, top_k=10)]
        if _hit(ids, ex.relevant_doc_ids, 10):
            hits += 1
            if i in miss_idxs:
                rec += 1
    results["zscore"] = {
        "recovered@10": rec,
        "recovery@10": rec / max(len(miss_idxs), 1),
        "recall@10_full": hits / max(len(eval_set), 1),
    }

    for k in [20, 30, 40, 50, 60]:
        eng = _make_engine(
            db,
            ckpt,
            causal_g=causal_g,
            tax_g=tax_g,
            rel_idx=rel_idx,
            fusion="rrf",
            rrf_k=k,
            weights=w,
            text_lookup=text_lookup,
        )
        rec = 0
        hits = 0
        mrr = []
        for i, ex in enumerate(eval_set):
            ids = [r["document_id"] for r in eng.search(ex.query_1024d, ex.query_text, top_k=10)]
            if _hit(ids, ex.relevant_doc_ids, 10):
                hits += 1
                if i in miss_idxs:
                    rec += 1
            rr = 0.0
            for rank, did in enumerate(ids, start=1):
                if did in ex.relevant_doc_ids:
                    rr = 1.0 / rank
                    break
            mrr.append(rr)
        results[f"rrf_k{k}"] = {
            "rrf_k": k,
            "recovered@10": rec,
            "recovery@10": rec / max(len(miss_idxs), 1),
            "recall@10_full": hits / max(len(eval_set), 1),
            "MRR": _mean(mrr),
        }
        print(
            f"[rrf] k={k}: recovered@10={rec}/{len(miss_idxs)} "
            f"R@10={results[f'rrf_k{k}']['recall@10_full']:.3f} MRR={results[f'rrf_k{k}']['MRR']:.3f}",
            flush=True,
        )

    best_rrf = max(
        (v for k, v in results.items() if k.startswith("rrf_")),
        key=lambda v: (v["recovered@10"], v["MRR"]),
    )
    results["best_rrf"] = best_rrf
    results["gap_to_zscore_recovered"] = results["zscore"]["recovered@10"] - best_rrf["recovered@10"]
    return results


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Post-validation: OOD · density · latency · RRF-k",
        "",
        f"Checkpoint: `{report['checkpoint']}`",
        "",
        "## 1. OOD transfer (no pack graphs)",
        "",
        "Adversarial-trained multi checkpoint scored on other finance corpora with channel fusion only (no Stage-1 graph).",
        "",
        "| Pack | dense R@10 | multi R@10 | ΔR@10 | dense P@10 | multi P@10 | ΔP@10 |",
        "|------|-----------:|-----------:|------:|-----------:|-----------:|------:|",
    ]
    for name, row in report["ood"].items():
        if "error" in row:
            lines.append(f"| {name} | — | — | — | — | — | error |")
            continue
        lines.append(
            f"| {name} | {row['dense_only']['recall@10']:.3f} | "
            f"{row['multi_zscore_no_graph']['recall@10']:.3f} | "
            f"{row['delta_R@10_multi_minus_dense']:+.3f} | "
            f"{row['dense_only']['precision@10']:.3f} | "
            f"{row['multi_zscore_no_graph']['precision@10']:.3f} | "
            f"{row['delta_P@10_multi_minus_dense']:+.3f} |"
        )

    dens = report["density"]
    lines += [
        "",
        "## 2. Graph-density stress (cross-cluster noise edges)",
        "",
        f"Base causal edges: **{dens['base_causal_edges']}**. Noise edges are random cross-cluster cause→effect links.",
        "",
        "| Noise edges | Total edges | Recovered@10 | Recovery | Full R@10 | Full P@10 |",
        "|------------:|------------:|-------------:|---------:|----------:|----------:|",
    ]
    for noise, row in dens["levels"].items():
        lines.append(
            f"| {row['noise_edges_requested']} | {row['total_causal_edges']} | "
            f"{row['recovered@10']} | {row['recovery@10']:.1%} | "
            f"{row['recall@10_full']:.3f} | {row['precision@10_full']:.3f} |"
        )

    lat = report["latency"]
    lines += [
        "",
        "## 3. Latency & candidate pruning",
        "",
        f"Stress graph edges: **{lat['noisy_graph_edges']}** (base + 400 noise).",
        "",
        "| Config | mean |cands| | max |cands| | mean Stage-1 ms | mean total ms | Recovered@10 |",
        "|--------|-------------:|------------:|----------------:|--------------:|-------------:|",
    ]
    for name, row in lat["configs"].items():
        lines.append(
            f"| {name} | {row['mean_candidates']:.1f} | {row['max_candidates']} | "
            f"{row['mean_stage1_ms']:.2f} | {row['mean_total_ms']:.2f} | "
            f"{row['recovered@10']} ({row['recovery@10']:.0%}) |"
        )

    rrf = report["rrf_sweep"]
    lines += [
        "",
        "## 4. RRF \(k\) sweep vs z-score",
        "",
        f"Z-score recovered@10: **{rrf['zscore']['recovered@10']}** "
        f"({rrf['zscore']['recovery@10']:.1%})",
        "",
        "| Config | Recovered@10 | Full R@10 | MRR |",
        "|--------|-------------:|----------:|----:|",
        f"| zscore | {rrf['zscore']['recovered@10']} | {rrf['zscore']['recall@10_full']:.3f} | — |",
    ]
    for key, row in rrf.items():
        if not key.startswith("rrf_k"):
            continue
        lines.append(
            f"| rrf k={row['rrf_k']} | {row['recovered@10']} | "
            f"{row['recall@10_full']:.3f} | {row['MRR']:.3f} |"
        )
    lines += [
        "",
        f"- Best RRF: `k={rrf['best_rrf']['rrf_k']}` recovered@10={rrf['best_rrf']['recovered@10']}",
        f"- Gap to z-score (recovered@10): **{rrf['gap_to_zscore_recovered']}**",
        "",
        "## Verdict",
        "",
        report.get("verdict", ""),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _verdict(report: dict[str, Any]) -> str:
    ood = report["ood"]
    soft = []
    for name in ("easy_finance", "hard_combined", "hard_gemini", "hard_gpt"):
        row = ood.get(name) or {}
        if "delta_P@10_multi_minus_dense" in row:
            soft.append((name, row["delta_P@10_multi_minus_dense"], row["delta_R@10_multi_minus_dense"]))
    dens0 = report["density"]["levels"].get("0", {})
    dens400 = report["density"]["levels"].get("400", {})
    best_rrf = report["rrf_sweep"]["best_rrf"]
    z = report["rrf_sweep"]["zscore"]
    lat = report["latency"]["configs"]
    prune = lat.get("beam8_cap150") or lat.get("cap200") or {}
    parts = [
        f"OOD: multi fusion without graphs "
        + (
            "does not collapse precision on soft packs"
            if all(d[1] >= -0.05 for d in soft)
            else "shows precision drift on some soft packs — monitor channel weights in prod"
        )
        + ".",
        f"Density: recovery {dens0.get('recovery@10', 0):.0%} @0 noise → "
        f"{dens400.get('recovery@10', 0):.0%} @400 noise edges "
        f"(P@10 {dens0.get('precision@10_full', 0):.3f} → {dens400.get('precision@10_full', 0):.3f}).",
        f"Latency: pruning config recovered@10={prune.get('recovered@10', 'n/a')} "
        f"at mean {prune.get('mean_candidates', 0):.0f} candidates / "
        f"{prune.get('mean_total_ms', 0):.2f} ms.",
        f"RRF: best k={best_rrf['rrf_k']} closes "
        f"{max(0, z['recovered@10'] - best_rrf['recovered@10'])} recovered gap vs z-score "
        f"({best_rrf['recovered@10']} vs {z['recovered@10']}).",
    ]
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Post-validation suite")
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument(
        "--checkpoint",
        default=str(MULTI_CKPT if MULTI_CKPT.exists() else CAUSAL_CKPT),
    )
    args = ap.parse_args(argv)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise SystemExit(f"missing checkpoint {ckpt_path}")

    cr = _load_cr()
    print(f"[post] checkpoint={ckpt_path}", flush=True)

    report: dict[str, Any] = {"checkpoint": str(ckpt_path)}
    print("\n===== 1) OOD TRANSFER =====", flush=True)
    report["ood"] = run_ood_transfer(cr, args.encoder, str(ckpt_path))

    print("\n===== 2) GRAPH DENSITY STRESS =====", flush=True)
    report["density"] = run_graph_density_stress(cr, args.encoder, str(ckpt_path))

    print("\n===== 3) LATENCY & PRUNING =====", flush=True)
    report["latency"] = run_latency_pruning(cr, args.encoder, str(ckpt_path))

    print("\n===== 4) RRF k SWEEP =====", flush=True)
    report["rrf_sweep"] = run_rrf_k_sweep(cr, args.encoder, str(ckpt_path))

    report["verdict"] = _verdict(report)

    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "post_validation.json"
    md_path = RESULTS / "POST_VALIDATION.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print("\n========== POST-VALIDATION ==========")
    print(report["verdict"])
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
