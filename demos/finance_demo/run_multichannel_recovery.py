"""
Multi-channel recovery scorecard (causal + hyperbolic + relational + RRF).

Builds Stage-1 structure indexes, optionally trains hyp/rel on top of the causal
adversarial checkpoint, then scores recovery on confirmed dense misses@10.

Usage:
  docker compose run --rm vectorprism python demos/finance_demo/run_multichannel_recovery.py
  docker compose run --rm vectorprism python demos/finance_demo/run_multichannel_recovery.py --skip-train
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

PACK = DEMO / "hard_adversarial"
DENSE_CKPT = ROOT / "checkpoints" / "finance_hard_adversarial.pt"
CAUSAL_CKPT = ROOT / "checkpoints" / "finance_hard_adversarial_causal.pt"
MULTI_CKPT = ROOT / "checkpoints" / "finance_hard_adversarial_multi.pt"
RESULTS = DEMO / "results" / "hard_eval"


def _run(cmd: list[str]) -> int:
    print("\n+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def _ensure_structure_files() -> None:
    hyp = PACK / "hyperbolic.jsonl"
    if not hyp.exists() or not (PACK / "relational_attrs.jsonl").exists():
        code = _run([sys.executable, str(DEMO / "build_adversarial_multichannel.py")])
        if code != 0:
            raise SystemExit(code)


def _train_channels(encoder: str, epochs: int, batch_size: int) -> None:
    if not CAUSAL_CKPT.exists():
        raise SystemExit(
            f"missing {CAUSAL_CKPT} — run demos/finance_demo/run_causal_recovery.py first"
        )
    # hyperbolic then relational, both writing MULTI_CKPT
    for channel, data in [
        ("hyperbolic", PACK / "hyperbolic.jsonl"),
        ("relational", PACK / "relational.jsonl"),
    ]:
        init = str(CAUSAL_CKPT if channel == "hyperbolic" else MULTI_CKPT)
        code = _run(
            [
                sys.executable,
                "train.py",
                "--channel",
                channel,
                "--data",
                str(data),
                "--init",
                init,
                "--out",
                str(MULTI_CKPT),
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


def score_multichannel(
    *,
    checkpoint: str,
    encoder_name: str,
    k: int = 10,
) -> dict[str, Any]:
    from ablation_harness import _fixed_weight_classifier
    from causal_graph import CausalDocGraph
    from retrieval_engine import PSMRetrievalEngine
    from structure_index import RelationalAttrIndex, TaxonomyGraph

    # Import helpers from sibling scorecard without requiring demos as a package
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "run_causal_recovery", DEMO / "run_causal_recovery.py"
    )
    assert _spec and _spec.loader
    _cr = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_cr)
    _dense_rank_stats = _cr._dense_rank_stats
    _eval_set = _cr._eval_set
    _hit = _cr._hit
    _ingest = _cr._ingest
    _search_ids = _cr._search_ids

    docs = PACK / "documents.jsonl"
    ev = PACK / "eval.jsonl"
    encoder, ckpt, db, pipe = _ingest(checkpoint, encoder_name, docs)
    eval_set = _eval_set(ev, encoder, pipe)
    dense_stats, miss_idxs = _dense_rank_stats(eval_set, db, k_miss=k)

    causal_g = CausalDocGraph.from_jsonl(PACK / "causal_graph.jsonl")
    tax_g = TaxonomyGraph.from_jsonl(PACK / "hyperbolic_graph.jsonl")
    rel_idx = RelationalAttrIndex.from_jsonl(PACK / "relational_attrs.jsonl")
    text_lookup = {r["document_id"]: str(r.get("chunk_text", "")) for r in db.rows}

    print(
        f"[multi] causal_edges={causal_g.n_edges} tax_edges={tax_g.n_edges} "
        f"rel_attrs={len(rel_idx.attrs)} enabled={ckpt.get('enabled_channels')}",
        flush=True,
    )

    def _make(
        *,
        use_causal: bool,
        use_hyp: bool,
        use_rel: bool,
        fusion: str,
        bonus: float = 4.0,
    ) -> PSMRetrievalEngine:
        return PSMRetrievalEngine(
            db,
            causal_matrix=ckpt["causal_matrix"],
            hard_truth_filter=False,
            causal_graph=causal_g if use_causal else None,
            taxonomy_graph=tax_g if use_hyp else None,
            relational_index=rel_idx if use_rel else None,
            stage1_dense_limit=50 if (use_causal or use_hyp or use_rel) else 100,
            stage1_graph_hops=2,
            graph_struct_bonus=bonus,
            fusion=fusion,
            rrf_k=60,
            doc_text_lookup=dict(text_lookup),
            query_conditioned_expansion=True,
        )

    # weight vector: dense, relational, disentangled, hyperbolic, causal
    run_cfgs: list[tuple[str, np.ndarray | None, dict[str, Any]]] = [
        ("dense_only", np.array([1, 0, 0, 0, 0], dtype=np.float32), dict(use_causal=False, use_hyp=False, use_rel=False, fusion="zscore", bonus=0.0)),
        ("causal+graph", np.array([0.35, 0.05, 0.05, 0.05, 0.50], dtype=np.float32), dict(use_causal=True, use_hyp=False, use_rel=False, fusion="zscore")),
        ("+hyp+rel zscore", np.array([0.25, 0.25, 0.05, 0.20, 0.25], dtype=np.float32), dict(use_causal=True, use_hyp=True, use_rel=True, fusion="zscore")),
        ("+hyp+rel RRF", np.array([0.25, 0.25, 0.05, 0.20, 0.25], dtype=np.float32), dict(use_causal=True, use_hyp=True, use_rel=True, fusion="rrf")),
        ("balanced RRF", np.array([0.20, 0.25, 0.05, 0.25, 0.25], dtype=np.float32), dict(use_causal=True, use_hyp=True, use_rel=True, fusion="rrf")),
        ("intent_router RRF", None, dict(use_causal=True, use_hyp=True, use_rel=True, fusion="rrf")),
    ]

    engines: dict[str, PSMRetrievalEngine] = {}
    for name, _w, kwargs in run_cfgs:
        engines[name] = _make(**kwargs)

    original = {
        name: eng.classifier.classify for name, eng in engines.items()
    }

    def _eval_config(name: str, w: np.ndarray | None) -> dict[str, float]:
        eng = engines[name]
        if w is None:
            eng.classifier.classify = original[name]
        else:
            eng.classifier.classify = _fixed_weight_classifier(w)
        hits = {1: 0, 5: 0, 10: 0}
        rr = []
        for ex in eval_set:
            ids = _search_ids(eng, ex, top_k=10)
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
        return {
            "recall@1": hits[1] / n,
            "recall@5": hits[5] / n,
            "recall@10": hits[10] / n,
            "MRR": float(np.mean(rr)) if rr else 0.0,
            "n": len(eval_set),
        }

    full_metrics = {name: _eval_config(name, w) for name, w, _ in run_cfgs}

    recover_counts = {name: {"@1": 0, "@5": 0, "@10": 0} for name, _, _ in run_cfgs}
    miss_examples = []
    for idx in miss_idxs:
        ex = eval_set[idx]
        st = dense_stats[idx]
        entry: dict[str, Any] = {
            "query": ex.query_text,
            "gold": st["gold"],
            "dense_rank": st["dense_rank"],
            "in_stage1_top100": st["in_stage1_top100"],
            "configs": {},
        }
        for name, w, _ in run_cfgs:
            eng = engines[name]
            if w is None:
                eng.classifier.classify = original[name]
            else:
                eng.classifier.classify = _fixed_weight_classifier(w)
            ids = _search_ids(eng, ex, top_k=10)
            h1 = _hit(ids, ex.relevant_doc_ids, 1)
            h5 = _hit(ids, ex.relevant_doc_ids, 5)
            h10 = _hit(ids, ex.relevant_doc_ids, 10)
            if h1:
                recover_counts[name]["@1"] += 1
            if h5:
                recover_counts[name]["@5"] += 1
            if h10:
                recover_counts[name]["@10"] += 1
            entry["configs"][name] = {
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

    return {
        "n_eval": len(eval_set),
        "n_dense_miss_at_k": len(miss_idxs),
        "k": k,
        "full_set_metrics": full_metrics,
        "recovery_on_dense_misses": recovery_rates,
        "enabled_channels": ckpt.get("enabled_channels"),
        "miss_examples": miss_examples,
        "checkpoint": checkpoint,
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    full = result["full_set_metrics"]
    rec = result["recovery_on_dense_misses"]
    lines = [
        "# Multi-channel recovery scorecard",
        "",
        "Stage-1: dense ∪ causal ancestors ∪ taxonomy lineage ∪ relational predicates.",
        "Stage-2: z-score fusion or Reciprocal Rank Fusion (RRF).",
        "",
        f"- Checkpoint: `{result['checkpoint']}`",
        f"- Dense misses@{result['k']}: **{result['n_dense_miss_at_k']}** / {result['n_eval']}",
        f"- Enabled channels: `{json.dumps(result.get('enabled_channels'))}`",
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

    best = max(
        rec.items(),
        key=lambda kv: (kv[1]["recovered@10"], kv[1]["recovered@1"]),
    )[0]
    lines += ["", f"### Per-miss detail (`{best}`)", ""]
    for m in result["miss_examples"]:
        cfg = m["configs"].get(best, {})
        mark = "RECOVERED" if cfg.get("hit@10") else "MISS"
        lines.append(
            f"- `{m['gold'][0]}` dense_rank={m['dense_rank']} → **{mark}** "
            f"top5={cfg.get('top5')}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-channel recovery scorecard")
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--checkpoint",
        default="",
        help="Override checkpoint (default: multi if exists else causal)",
    )
    args = ap.parse_args(argv)

    if not DENSE_CKPT.exists():
        raise SystemExit(f"missing dense checkpoint {DENSE_CKPT}")

    _ensure_structure_files()
    # refresh structure files every run so attrs stay aligned
    _run([sys.executable, str(DEMO / "build_adversarial_multichannel.py")])

    if not args.skip_train:
        _train_channels(args.encoder, args.epochs, args.batch_size)
        ckpt = str(MULTI_CKPT)
    else:
        ckpt = args.checkpoint or (
            str(MULTI_CKPT) if MULTI_CKPT.exists() else str(CAUSAL_CKPT)
        )
        if not Path(ckpt).exists():
            raise SystemExit(f"--skip-train but missing {ckpt}")

    print(f"[multi] scoring with {ckpt}", flush=True)
    result = score_multichannel(checkpoint=ckpt, encoder_name=args.encoder, k=args.k)
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "multichannel_recovery.json"
    md_path = RESULTS / "MULTICHANNEL_RECOVERY_RESULTS.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, md_path)

    rec = result["recovery_on_dense_misses"]
    print("\n========== MULTI-CHANNEL RECOVERY ==========")
    for name, m in rec.items():
        print(
            f"{name}: recovered@10={m['recovered@10']}/{result['n_dense_miss_at_k']} "
            f"({m['recovery@10']:.1%})  full_R@10={result['full_set_metrics'][name]['recall@10']:.3f}"
        )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
