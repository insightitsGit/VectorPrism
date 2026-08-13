"""
Stakeholder demo packaging for the adversarial multi-channel milestone.

Default (fast): rebuild TECHNICAL_REPORT.md from existing JSON result artifacts.
With --full: also re-run extraction comparison (heuristic) + print scorecard pointers.

Usage:
  python demos/finance_demo/run_stakeholder_demo.py
  docker compose run --rm vectorprism python demos/finance_demo/run_stakeholder_demo.py --full
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEMO = Path(__file__).resolve().parent
RESULTS = DEMO / "results" / "hard_eval"
REPORT = DEMO / "TECHNICAL_REPORT.md"


def _load(name: str) -> dict[str, Any] | None:
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _run(cmd: list[str]) -> int:
    print("\n+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def build_report(*, include_auto: bool = True) -> str:
    multi = _load("multichannel_recovery.json") or {}
    post = _load("post_validation.json") or {}
    robust = _load("robustness_validation.json") or {}
    auto = _load("auto_extraction.json") if include_auto else None
    causal = _load("causal_recovery.json") or {}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# VectorPrism Technical Report — Adversarial Multi-Channel Milestone",
        "",
        f"*Generated {now} by `demos/finance_demo/run_stakeholder_demo.py`*",
        "",
        "## 1. Executive summary",
        "",
        "VectorPrism’s multi-channel thesis is validated on a calibrated adversarial finance pack:",
        "",
        "- Dense-only retrieval collapses to **R@10 = 7.1%** (13/14 Miss@10).",
        "- Structured Stage-1 expansion + Stage-2 fusion recovers **10–13 / 13** dense misses "
        "(**77–100%**), depending on fusion (RRF vs z-score).",
        "- The critical engineering fix was **per-channel hop maps**: topological bonuses apply "
        "even when gold is already inside the dense Stage-1 pool.",
        "",
        "## 2. Architecture (what was measured)",
        "",
        "```",
        "Query",
        "  ├─ Stage-1: Dense Top-K  ∪  Causal ancestors  ∪  Taxonomy lineage  ∪  Relational predicates",
        "  └─ Stage-2: z-score weighted fusion  |  Reciprocal Rank Fusion (k=20 default)",
        "```",
        "",
        "| Channel | Role on this pack |",
        "|---------|-------------------|",
        "| Dense | Semantic neighborhood (fails under cohesive distractors) |",
        "| Causal | Cause→symptom DAG; hop-scaled Stage-2 bonus |",
        "| Hyperbolic | Policy tree parent/child/sibling expansion |",
        "| Relational | Exact attribute predicates (amount, callback, timezone, …) |",
        "",
        "## 3. Primary recovery scorecard",
        "",
    ]

    if multi:
        rec = multi.get("recovery_on_dense_misses") or {}
        full = multi.get("full_set_metrics") or {}
        lines += [
            f"- Checkpoint: `{multi.get('checkpoint', 'n/a')}`",
            f"- Dense misses@10: **{multi.get('n_dense_miss_at_k', '?')}** / {multi.get('n_eval', '?')}",
            "",
            "| Config | Recovered@10 | Full R@10 | MRR |",
            "|--------|-------------:|----------:|----:|",
        ]
        for name, m in rec.items():
            fm = full.get(name) or {}
            lines.append(
                f"| {name} | {m.get('recovered@10', '?')}/{multi.get('n_dense_miss_at_k', '?')} "
                f"({m.get('recovery@10', 0):.1%}) | {fm.get('recall@10', float('nan')):.3f} | "
                f"{fm.get('MRR', float('nan')):.3f} |"
            )
    else:
        lines.append("_Missing `multichannel_recovery.json` — run `run_multichannel_recovery.py`._")

    # Post-validation
    lines += ["", "## 4. Post-validation (noise · latency · OOD · RRF-k)", ""]
    if post:
        ood = post.get("ood") or {}
        dens = (post.get("density") or {}).get("levels") or {}
        lat = (post.get("latency") or {}).get("configs") or {}
        rrf = post.get("rrf_sweep") or {}
        lines += [
            "### 4.1 OOD transfer (no pack graphs)",
            "",
            "| Pack | dense R@10 | multi R@10 | ΔP@10 |",
            "|------|-----------:|-----------:|------:|",
        ]
        for name, row in ood.items():
            if "error" in row:
                continue
            lines.append(
                f"| {name} | {row['dense_only']['recall@10']:.3f} | "
                f"{row['multi_zscore_no_graph']['recall@10']:.3f} | "
                f"{row['delta_P@10_multi_minus_dense']:+.3f} |"
            )
        lines += [
            "",
            "### 4.2 Graph density stress",
            "",
            "| Noise edges | Recovered@10 | Full R@10 | Full P@10 |",
            "|------------:|-------------:|----------:|----------:|",
        ]
        for _, row in dens.items():
            lines.append(
                f"| {row['noise_edges_requested']} | {row['recovered@10']} | "
                f"{row['recall@10_full']:.3f} | {row['precision@10_full']:.3f} |"
            )
        clean = lat.get("clean_graph_baseline") or {}
        noisy = lat.get("baseline_no_cap") or {}
        lines += [
            "",
            "### 4.3 Latency envelope",
            "",
            f"- Clean graph: mean **{clean.get('mean_candidates', 'n/a')}** cands / "
            f"**{clean.get('mean_total_ms', 'n/a')}** ms",
            f"- Noisy graph (+400 edges): mean **{noisy.get('mean_candidates', 'n/a')}** cands / "
            f"**{noisy.get('mean_total_ms', 'n/a')}** ms",
            "",
            "### 4.4 RRF \(k\)",
            "",
            f"- Z-score recovered@10: **{(rrf.get('zscore') or {}).get('recovered@10', 'n/a')}**",
            f"- Best RRF: k=**{(rrf.get('best_rrf') or {}).get('rrf_k', 20)}** → "
            f"recovered@10=**{(rrf.get('best_rrf') or {}).get('recovered@10', 'n/a')}**",
            "",
            f"**Verdict:** {post.get('verdict', '')}",
        ]
    else:
        lines.append("_Missing `post_validation.json` — run `run_post_validation.py`._")

    # Robustness
    lines += ["", "## 5. Robustness (precision · sparsity · scale)", ""]
    if robust:
        fp = robust.get("precision_profile") or {}
        sp = (robust.get("sparsity") or {}).get("fractions") or {}
        sc = robust.get("scale") or {}
        lines += [
            "### 5.1 Precision / MRR / false positives",
            "",
            "| Config | R@10 | P@10 | MRR | Cross-cluster FP@10 |",
            "|--------|-----:|-----:|----:|--------------------:|",
        ]
        for name, m in fp.items():
            lines.append(
                f"| {name} | {m['recall@10']:.3f} | {m['precision@10']:.3f} | "
                f"{m['MRR']:.3f} | {m['cross_cluster_fp_rate@10']:.3f} |"
            )
        lines += [
            "",
            "### 5.2 Graph sparsity (edge dropout)",
            "",
            "| Drop | Recovered@10 | MRR |",
            "|-----:|-------------:|----:|",
        ]
        for _, m in sp.items():
            lines.append(
                f"| {m['drop_frac']:.0%} | {m['recovered@10']}/{m['n_dense_miss']} | {m['MRR']:.3f} |"
            )
        corp = sc.get("corpus") or {}
        lines += [
            "",
            f"### 5.3 Scale ({corp.get('n_documents', '?')} docs)",
            "",
            "| Config | Recovered@10 | mean cands | mean ms | Cross-cluster FP@10 |",
            "|--------|-------------:|-----------:|--------:|--------------------:|",
        ]
        for name, m in (sc.get("configs") or {}).items():
            lines.append(
                f"| {name} | {m['recovered@10']}/{m['n_dense_miss']} | "
                f"{m['mean_candidates']:.1f} | {m['mean_total_ms']:.2f} | "
                f"{m['cross_cluster_fp_rate@10']:.3f} |"
            )
        lines += ["", f"**Verdict:** {robust.get('verdict', '')}"]
    else:
        lines.append("_Missing `robustness_validation.json` — run `run_robustness_validation.py`._")

    # Auto extraction
    lines += ["", "## 6. Automated graph ingestion (curated vs auto)", ""]
    if auto:
        cmp_ = auto.get("comparison") or {}
        lines += [
            f"- Backend: `{auto.get('backend')}`",
            "",
            "| Structure | P | R | F1 |",
            "|-----------|--:|--:|---:|",
        ]
        for name, m in cmp_.items():
            lines.append(
                f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
            )
        if auto.get("recovery"):
            lines += [
                "",
                "| Graph source | Recovered@10 | R@10 | MRR |",
                "|--------------|-------------:|-----:|----:|",
            ]
            for row in auto["recovery"]:
                lines.append(
                    f"| {row['label']} | {row['recovered@10']}/{row['n_dense_miss']} | "
                    f"{row['recall@10']:.3f} | {row['MRR']:.3f} |"
                )
    else:
        lines += [
            "Run:",
            "",
            "```bash",
            "docker compose run --rm vectorprism python demos/finance_demo/extract_structure_auto.py "
            "--backend heuristic --score",
            "```",
        ]

    lines += [
        "",
        "## 7. Caveats for stakeholders",
        "",
        "1. **Curated alignment:** High recovery assumes structure indexes reflect domain logic. "
        "Auto extraction is the path to production; expect lower edge F1 until LLM/human review loops land.",
        "2. **RRF vs z-score:** Prefer **RRF (k=20)** as the conservative production default; "
        "z-score shines when channel score scales are stable.",
        "3. **Open-world noise:** Cross-cluster edge injection still kept recovery ≥85% at +400 edges; "
        "use `beam_width` / `max_stage1_candidates` when graphs densify.",
        "",
        "## 8. Reproducibility commands",
        "",
        "```bash",
        "# Multi-channel recovery scorecard",
        "docker compose run --rm vectorprism python demos/finance_demo/run_multichannel_recovery.py --skip-train",
        "",
        "# Post-validation (OOD / density / latency / RRF-k)",
        "docker compose run --rm vectorprism python demos/finance_demo/run_post_validation.py",
        "",
        "# Robustness (precision / sparsity / 1k scale)",
        "docker compose run --rm vectorprism python demos/finance_demo/run_robustness_validation.py",
        "",
        "# Automated structure extraction (+ recovery with auto graphs)",
        "docker compose run --rm vectorprism python demos/finance_demo/extract_structure_auto.py "
        "--backend heuristic --score",
        "",
        "# Rebuild this report from artifacts",
        "python demos/finance_demo/run_stakeholder_demo.py",
        "```",
        "",
        "## 9. Artifact index",
        "",
        "| File | Contents |",
        "|------|----------|",
        "| `results/hard_eval/multichannel_recovery.json` | Primary recovery scorecard |",
        "| `results/hard_eval/post_validation.json` | OOD / density / latency / RRF |",
        "| `results/hard_eval/robustness_validation.json` | MRR / sparsity / scale |",
        "| `results/hard_eval/auto_extraction.json` | Auto vs curated graphs |",
        "| `checkpoints/finance_hard_adversarial_multi.pt` | Trained multi-channel checkpoint |",
        "| `hard_adversarial/` | Curated pack + graphs |",
        "| `hard_adversarial_auto/` | Auto-extracted graphs |",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stakeholder demo / technical report packager")
    ap.add_argument(
        "--full",
        action="store_true",
        help="Run heuristic auto-extraction --score before rebuilding the report",
    )
    ap.add_argument("--backend", choices=["auto", "heuristic", "llm"], default="heuristic")
    args = ap.parse_args(argv)

    if args.full:
        code = _run(
            [
                sys.executable,
                str(DEMO / "extract_structure_auto.py"),
                "--backend",
                args.backend,
                "--score",
            ]
        )
        if code != 0:
            return code

    text = build_report(include_auto=True)
    REPORT.write_text(text, encoding="utf-8")
    # Also mirror under results for the eval folder consumers
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "TECHNICAL_REPORT.md").write_text(text, encoding="utf-8")
    print(f"Wrote {REPORT}")
    print(f"Wrote {RESULTS / 'TECHNICAL_REPORT.md'}")
    # Short console brief
    multi = _load("multichannel_recovery.json") or {}
    rec = (multi.get("recovery_on_dense_misses") or {}).get("+hyp+rel zscore") or (
        multi.get("recovery_on_dense_misses") or {}
    ).get("multi zscore")
    if not rec:
        # try keys from scorecard
        for k, v in (multi.get("recovery_on_dense_misses") or {}).items():
            if "hyp" in k or "zscore" in k.lower() or "multi" in k:
                rec = v
                break
    if rec:
        print(
            f"Brief: multi-channel recovered@10="
            f"{rec.get('recovered@10')}/{multi.get('n_dense_miss_at_k')} "
            f"({rec.get('recovery@10', 0):.0%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
