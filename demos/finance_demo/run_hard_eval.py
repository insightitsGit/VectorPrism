"""
Hard-eval dense baseline.

Modes:
  isolated  Train+index+eval each pack alone (hard_gemini, hard_gpt) — preferred
  mixed     Train on hard_combined, score both evals on the mixed corpus

Usage:
  docker compose run --rm vectorprism python demos/finance_demo/run_hard_eval.py --mode isolated
  docker compose run --rm vectorprism python demos/finance_demo/run_hard_eval.py --mode mixed
  python demos/finance_demo/run_hard_eval.py --mode isolated --encoder hash --epochs 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

COMBINED = DEMO / "hard_combined"
GEMINI = DEMO / "hard_gemini"
GPT = DEMO / "hard_gpt"
ADVERSARIAL = DEMO / "hard_adversarial"
RESULTS = DEMO / "results" / "hard_eval"

ISOLATED_PACKS = {
    "gemini": GEMINI,
    "gpt": GPT,
    "adversarial": ADVERSARIAL,
}


def _run(cmd: list[str]) -> int:
    print("\n+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def mine_dense_misses(
    *,
    encoder_name: str,
    checkpoint: str,
    documents_jsonl: Path,
    eval_jsonl: Path,
    k: int = 10,
) -> dict:
    from base_encoder import HashingEncoder, SentenceTransformerEncoder
    from channel_datasets import load_documents_jsonl, load_eval_examples_raw
    from checkpointing import load_checkpoint
    from eval_runner import InMemoryCorpusDB, dense_cosine_baseline, build_eval_set
    from ingest_pipeline import VectorPrismIngestPipeline, IngestDocument
    from retrieval_engine import PSMRetrievalEngine
    from eval_harness import evaluate
    from tensor_contract import PSMTensorContract as C

    if encoder_name in {"hash", "hashing", "test"}:
        encoder = HashingEncoder(768)
    else:
        encoder = SentenceTransformerEncoder(encoder_name)

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
    eval_set = build_eval_set(str(eval_jsonl), encoder, pipe)
    engine = PSMRetrievalEngine(db, causal_matrix=ckpt["causal_matrix"], hard_truth_filter=False)
    engine.classifier.classify = lambda _q: (
        np.array([1.0, 0, 0, 0, 0], dtype=np.float32),
        {"min_truth": 0.0, "max_anchor_dist": 1.0},
    )
    vp = evaluate(engine, eval_set, k_values=[1, 5, 10])
    corpus = {r["document_id"]: r["tensor_1024d"] for r in db.rows}
    base = dense_cosine_baseline(eval_set, corpus, k_values=[1, 5, 10])

    raw = load_eval_examples_raw(str(eval_jsonl))
    doc_ids = list(corpus.keys())
    mat = np.stack([corpus[i][C.DENSE_CORE.start : C.DENSE_CORE.end] for i in doc_ids])

    confirmed_misses = []
    labeled_miss_hits = 0
    labeled_miss_total = 0
    miss_at_1 = 0
    for ex, row in zip(eval_set, raw):
        q = ex.query_1024d[C.DENSE_CORE.start : C.DENSE_CORE.end]
        scores = mat @ q
        order = np.argsort(scores)[::-1][:k]
        retrieved = [doc_ids[i] for i in order]
        hit10 = bool(set(retrieved) & ex.relevant_doc_ids)
        hit1 = bool(set(retrieved[:1]) & ex.relevant_doc_ids)
        if not hit1:
            miss_at_1 += 1
        expect_miss = bool(row.get("dense_should_miss"))
        if expect_miss:
            labeled_miss_total += 1
            if not hit10:
                labeled_miss_hits += 1
        if not hit10:
            confirmed_misses.append(
                {
                    "query": ex.query_text,
                    "gold": sorted(ex.relevant_doc_ids),
                    "top10": retrieved,
                    "dense_should_miss": expect_miss,
                    "channel_hint": row.get("channel_hint"),
                    "miss_reason": row.get("miss_reason"),
                }
            )

    return {
        "n_docs_indexed": len(docs),
        "n_eval": len(eval_set),
        "vectorprism_dense_only": vp,
        "dense_cosine_baseline": base,
        "confirmed_dense_misses_at_10": len(confirmed_misses),
        "confirmed_miss_rate_at_10": len(confirmed_misses) / max(len(eval_set), 1),
        "confirmed_dense_misses_at_1": miss_at_1,
        "confirmed_miss_rate_at_1": miss_at_1 / max(len(eval_set), 1),
        "labeled_dense_should_miss": labeled_miss_total,
        "labeled_miss_confirmed_at_10": labeled_miss_hits,
        "labeled_miss_confirmation_rate": labeled_miss_hits / max(labeled_miss_total, 1),
        "miss_examples": confirmed_misses[:25],
    }


def pack_result_block(name: str, mined: dict, *, pairs: Path, n_pairs: int, ckpt: str) -> dict:
    return {
        "pack": name,
        "train_pairs": str(pairs),
        "n_train_pairs": n_pairs,
        "n_docs_indexed": mined["n_docs_indexed"],
        "checkpoint": ckpt,
        "n_eval": mined["n_eval"],
        "metrics": {
            "vectorprism_dense_only": mined["vectorprism_dense_only"],
            "dense_cosine_baseline": mined["dense_cosine_baseline"],
        },
        "confirmed_dense_misses_at_1": mined["confirmed_dense_misses_at_1"],
        "confirmed_miss_rate_at_1": mined["confirmed_miss_rate_at_1"],
        "confirmed_dense_misses_at_10": mined["confirmed_dense_misses_at_10"],
        "confirmed_miss_rate_at_10": mined["confirmed_miss_rate_at_10"],
        "labeled_dense_should_miss": mined["labeled_dense_should_miss"],
        "labeled_miss_confirmed_at_10": mined["labeled_miss_confirmed_at_10"],
        "labeled_miss_confirmation_rate": mined["labeled_miss_confirmation_rate"],
        "miss_examples": mined["miss_examples"],
    }


def train_dense(
    *,
    pairs: Path,
    out: str,
    encoder: str,
    epochs: int,
    batch_size: int,
    skip_train: bool,
) -> int:
    if skip_train:
        if not Path(out).exists():
            raise SystemExit(f"--skip-train but checkpoint missing: {out}")
        return 0
    return _run(
        [
            sys.executable,
            "train.py",
            "--channel",
            "dense",
            "--data",
            str(pairs),
            "--out",
            out,
            "--encoder",
            encoder,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
        ]
    )


def run_isolated_pack(
    *,
    name: str,
    pack_dir: Path,
    encoder: str,
    epochs: int,
    batch_size: int,
    skip_train: bool,
    ckpt: str,
) -> dict:
    pairs = pack_dir / "dense_pairs.jsonl"
    docs = pack_dir / "documents.jsonl"
    ev = pack_dir / "eval.jsonl"
    for p in (pairs, docs, ev):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    n_pairs = sum(1 for line in pairs.open(encoding="utf-8") if line.strip())
    n_docs = sum(1 for line in docs.open(encoding="utf-8") if line.strip())
    print(f"\n===== ISOLATED {name} =====", flush=True)
    print(f"[hard-eval] pairs={n_pairs} docs={n_docs} eval={ev}", flush=True)
    code = train_dense(
        pairs=pairs,
        out=ckpt,
        encoder=encoder,
        epochs=epochs,
        batch_size=batch_size,
        skip_train=skip_train,
    )
    if code != 0:
        raise SystemExit(code)
    mined = mine_dense_misses(
        encoder_name=encoder,
        checkpoint=ckpt,
        documents_jsonl=docs,
        eval_jsonl=ev,
    )
    return pack_result_block(name, mined, pairs=pairs, n_pairs=n_pairs, ckpt=ckpt)


def write_isolated_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# Hard-eval dense baseline — ISOLATED packs",
        "",
        f"Encoder: `{summary['encoder']}`",
        "Each pack was trained, indexed, and evaluated **alone** (no cross-pack mixing).",
        "",
    ]
    titles = {"gemini": "Gemini", "gpt": "GPT", "adversarial": "Adversarial"}
    for key, title in titles.items():
        if key not in summary:
            continue
        r = summary[key]
        m = r["metrics"]["dense_cosine_baseline"]
        lines += [
            f"## {title}",
            "",
            f"- Train pairs: {r['n_train_pairs']} | Docs indexed: {r['n_docs_indexed']} | Eval: {r['n_eval']}",
            f"- Checkpoint: `{r['checkpoint']}`",
            f"- R@1 / R@5 / R@10: **{m['recall@1']:.3f}** / **{m['recall@5']:.3f}** / **{m['recall@10']:.3f}**",
            f"- MRR: **{m['MRR']:.3f}**",
            f"- Misses@1: **{r['confirmed_dense_misses_at_1']}** / {r['n_eval']} ({r['confirmed_miss_rate_at_1']:.1%})",
            f"- Misses@10: **{r['confirmed_dense_misses_at_10']}** / {r['n_eval']} ({r['confirmed_miss_rate_at_10']:.1%})",
            f"- Label confirm@10: **{r['labeled_miss_confirmed_at_10']}** / "
            f"{r['labeled_dense_should_miss']} ({r['labeled_miss_confirmation_rate']:.1%})",
            "",
        ]
    if summary.get("mixed_comparison"):
        lines += ["## vs previous MIXED run", "", "```json", json.dumps(summary["mixed_comparison"], indent=2), "```", ""]
    lines += [
        "## Interpretation",
        "",
        "- Isolated = fair pack hardness (no foreign distractors / no cross-pack pair leakage).",
        "- Higher miss@10 than mixed ⇒ mixing was making dense look stronger (or weaker) artificially.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hard-eval dense baseline")
    ap.add_argument("--mode", choices=["isolated", "mixed"], default="isolated")
    ap.add_argument(
        "--packs",
        nargs="+",
        choices=list(ISOLATED_PACKS.keys()),
        default=["adversarial"],
        help="Which isolated packs to run (default: adversarial)",
    )
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--out-gemini", default=str(ROOT / "checkpoints" / "finance_hard_gemini.pt"))
    ap.add_argument("--out-gpt", default=str(ROOT / "checkpoints" / "finance_hard_gpt.pt"))
    ap.add_argument(
        "--out-adversarial",
        default=str(ROOT / "checkpoints" / "finance_hard_adversarial.pt"),
    )
    ap.add_argument("--out-mixed", default=str(ROOT / "checkpoints" / "finance_hard.pt"))
    args = ap.parse_args(argv)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_map = {
        "gemini": args.out_gemini,
        "gpt": args.out_gpt,
        "adversarial": args.out_adversarial,
    }

    if args.mode == "isolated":
        summary: dict = {"mode": "isolated", "encoder": args.encoder}
        for name in args.packs:
            summary[name] = run_isolated_pack(
                name=name,
                pack_dir=ISOLATED_PACKS[name],
                encoder=args.encoder,
                epochs=args.epochs,
                batch_size=args.batch_size,
                skip_train=args.skip_train,
                ckpt=out_map[name],
            )

        json_path = RESULTS / "dense_baseline_isolated.json"
        if args.packs == ["adversarial"]:
            json_path = RESULTS / "dense_baseline_adversarial.json"
            md_path = RESULTS / "HARD_EVAL_RESULTS_ADVERSARIAL.md"
        else:
            md_path = RESULTS / "HARD_EVAL_RESULTS_ISOLATED.md"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_isolated_markdown(summary, md_path)

        print("\n========== ISOLATED HARD EVAL SUMMARY ==========")
        for name in args.packs:
            r = summary[name]
            m = r["metrics"]["dense_cosine_baseline"]
            print(
                f"{name}: R@1={m['recall@1']:.3f} R@10={m['recall@10']:.3f}  "
                f"miss@1={r['confirmed_dense_misses_at_1']}/{r['n_eval']}  "
                f"miss@10={r['confirmed_dense_misses_at_10']}/{r['n_eval']}  "
                f"label_confirm@10={r['labeled_miss_confirmation_rate']:.1%}"
            )
        print(f"Wrote {json_path}")
        print(f"Wrote {md_path}")
        return 0

    # mixed mode (legacy)
    pairs = COMBINED / "dense_pairs.jsonl"
    docs = COMBINED / "documents.jsonl"
    eval_a = COMBINED / "eval_gemini.jsonl"
    eval_b = COMBINED / "eval_gpt.jsonl"
    for p in (pairs, docs, eval_a, eval_b):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    n_pairs = sum(1 for line in pairs.open(encoding="utf-8") if line.strip())
    print(f"[hard-eval mixed] train pairs={n_pairs} encoder={args.encoder}")
    code = train_dense(
        pairs=pairs,
        out=args.out_mixed,
        encoder=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        skip_train=args.skip_train,
    )
    if code != 0:
        return code
    print("\n[hard-eval] Run A — Gemini on mixed corpus", flush=True)
    a = mine_dense_misses(
        encoder_name=args.encoder,
        checkpoint=args.out_mixed,
        documents_jsonl=docs,
        eval_jsonl=eval_a,
    )
    print("\n[hard-eval] Run B — GPT on mixed corpus", flush=True)
    b = mine_dense_misses(
        encoder_name=args.encoder,
        checkpoint=args.out_mixed,
        documents_jsonl=docs,
        eval_jsonl=eval_b,
    )
    summary = {
        "mode": "mixed",
        "encoder": args.encoder,
        "checkpoint": args.out_mixed,
        "n_train_pairs": n_pairs,
        "run_a_gemini": pack_result_block("gemini", a, pairs=pairs, n_pairs=n_pairs, ckpt=args.out_mixed),
        "run_b_gpt": pack_result_block("gpt", b, pairs=pairs, n_pairs=n_pairs, ckpt=args.out_mixed),
    }
    json_path = RESULTS / "dense_baseline.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n========== MIXED HARD EVAL SUMMARY ==========")
    print(
        f"Gemini R@10={a['dense_cosine_baseline']['recall@10']:.3f} "
        f"miss@10={a['confirmed_dense_misses_at_10']}/{a['n_eval']}"
    )
    print(
        f"GPT    R@10={b['dense_cosine_baseline']['recall@10']:.3f} "
        f"miss@10={b['confirmed_dense_misses_at_10']}/{b['n_eval']}"
    )
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
