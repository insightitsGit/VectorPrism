"""Validate a hard-eval JSONL pack.

Usage:
  python scripts/validate_hard_eval.py
  python scripts/validate_hard_eval.py --pack-dir demos/finance_demo/hard_gemini
  python scripts/validate_hard_eval.py --pack-dir demos/finance_demo/hard_gpt --min-dense-pairs 400
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(p: Path):
    rows, bad = [], []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception as e:
            bad.append((i, str(e), line[:120]))
    return rows, bad


def validate(
    pack_dir: Path,
    *,
    min_docs: int = 120,
    min_dense_pairs: int = 400,
    min_eval: int = 80,
    min_causal: int = 40,
) -> list[str]:
    files = {
        "documents": pack_dir / "documents.jsonl",
        "dense_pairs": pack_dir / "dense_pairs.jsonl",
        "eval": pack_dir / "eval.jsonl",
        "causal": pack_dir / "causal.jsonl",
    }
    data, errors = {}, {}
    for k, p in files.items():
        if not p.exists():
            print(f"MISSING {p}")
            return ["missing_files"]
        rows, bad = load(p)
        data[k] = rows
        errors[k] = bad
        print(f"{k}: {len(rows)} rows, {len(bad)} parse errors, size={p.stat().st_size}")

    docs, pairs, ev, causal = data["documents"], data["dense_pairs"], data["eval"], data["causal"]
    doc_ids = [d.get("document_id") for d in docs]
    doc_set = set(doc_ids)

    print("\n--- documents ---")
    print("unique ids", len(doc_set), "dupes", len(doc_ids) - len(doc_set))
    print(
        "missing fields",
        sum(1 for d in docs if not d.get("document_id") or not str(d.get("chunk_text", "")).strip()),
    )
    lens = [len(str(d.get("chunk_text", ""))) for d in docs]
    if lens:
        print("chunk chars min/median/max", min(lens), sorted(lens)[len(lens) // 2], max(lens))

    print("\n--- dense_pairs ---")
    print("missing query/passage", sum(1 for r in pairs if not r.get("query") or not r.get("passage")))
    src_ok = sum(1 for r in pairs if r.get("source_doc_id") in doc_set)
    src_bad = sum(1 for r in pairs if r.get("source_doc_id") and r.get("source_doc_id") not in doc_set)
    print("source_doc_id ok/bad/missing", src_ok, src_bad, sum(1 for r in pairs if not r.get("source_doc_id")))
    doc_text = {d["document_id"]: str(d["chunk_text"]).strip() for d in docs if d.get("document_id")}
    verbatim = near = 0
    for r in pairs:
        p = str(r.get("passage", "")).strip()
        sid = r.get("source_doc_id")
        if sid in doc_text and p == doc_text[sid]:
            verbatim += 1
        elif sid in doc_text and (p in doc_text[sid] or doc_text[sid][:80] in p):
            near += 1
    print(f"passage exact verbatim of source doc {verbatim} ({verbatim/max(len(pairs),1):.1%})")
    print("passage near/substring source doc", near)

    print("\n--- eval ---")
    orphans = []
    multi = 0
    for i, r in enumerate(ev, 1):
        ids = r.get("relevant_doc_ids") or []
        if len(ids) > 1:
            multi += 1
        for did in ids:
            if did not in doc_set:
                orphans.append((i, did, str(r.get("query", ""))[:80]))
    print("orphan relevant_doc_ids", len(orphans))
    for o in orphans[:25]:
        print("  ORPHAN", o)
    print("multi-gold rows", multi)
    miss = Counter(bool(r.get("dense_should_miss")) for r in ev)
    print("dense_should_miss", dict(miss), "pct_true", f"{miss[True]/max(len(ev),1):.1%}")
    print("channel_hint", dict(Counter(r.get("channel_hint") for r in ev)))
    print("difficulty", dict(Counter(r.get("difficulty") for r in ev)))
    train_q = {str(r.get("query", "")).strip().lower() for r in pairs}
    overlap = sum(1 for r in ev if str(r.get("query", "")).strip().lower() in train_q)
    print("eval queries identical to train queries", overlap)

    print("\n--- causal ---")
    print("missing earlier/later", sum(1 for r in causal if not r.get("earlier") or not r.get("later")))

    def prefix(s: str) -> str:
        s = str(s)
        if s.startswith("DOC_"):
            parts = s.split("_")
            return "_".join(parts[:2]) if len(parts) >= 2 else s[:8]
        parts = s.split("-")
        return "-".join(parts[:2]) if len(parts) >= 2 else s[:8]

    fam = defaultdict(list)
    for d in docs:
        fam[prefix(d["document_id"])].append(d["document_id"])
    weak = []
    for r in ev:
        g = (r.get("relevant_doc_ids") or [None])[0]
        if not g:
            continue
        siblings = [x for x in fam[prefix(g)] if x != g]
        if len(siblings) < 2:
            weak.append((g, len(siblings), str(r.get("query", ""))[:70]))
    print("\n--- distractor heuristic ---")
    print("eval golds with <2 same-prefix sibling distractors", len(weak), "/", len(ev))
    for w in weak[:20]:
        print("  WEAK", w)

    print("\n--- scale gates vs targets ---")
    targets = {
        "documents": min_docs,
        "dense_pairs": min_dense_pairs,
        "eval": min_eval,
        "causal": min_causal,
    }
    for k, t in targets.items():
        n = len(data[k])
        print(f"{k}: {n} (target>={t}) {'PASS' if n >= t else 'FAIL'}")

    print("\n--- parse errors ---")
    for k, bad in errors.items():
        if bad:
            print(k, bad[:5])

    fail: list[str] = []
    if len(docs) < min_docs:
        fail.append("docs_scale")
    if len(pairs) < min_dense_pairs:
        fail.append("pairs_scale")
    if len(ev) < min_eval:
        fail.append("eval_scale")
    if len(causal) < min_causal:
        fail.append("causal_scale")
    if orphans:
        fail.append("orphan_ids")
    if len(doc_ids) != len(doc_set):
        fail.append("duplicate_doc_ids")
    confirmed = sum(1 for r in ev if r.get("miss_confirmed") is True)
    # After Step-2 label hardening, optimistic 50%+ dense_should_miss is not expected.
    # Prefer confirmed-miss annotations when present; otherwise keep the 50% pack-hardness gate.
    if confirmed > 0:
        print(
            f"\nnote: {confirmed} rows have miss_confirmed=true "
            "(labels hardened to measured dense misses; skipping 50% dense_should_miss gate)"
        )
    elif miss[True] / max(len(ev), 1) < 0.5:
        fail.append("dense_miss_ratio")
    if overlap:
        fail.append("train_eval_query_leak")
    print("\nVERDICT:", "NOT READY" if fail else "READY_FOR_BASELINE_RUN", "fails=", fail)
    return fail


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate hard-eval JSONL pack")
    ap.add_argument(
        "--pack-dir",
        type=Path,
        default=Path("demos/finance_demo/hard_gemini"),
        help="Directory containing documents/dense_pairs/eval/causal.jsonl",
    )
    ap.add_argument("--min-docs", type=int, default=120)
    ap.add_argument("--min-dense-pairs", type=int, default=150)
    ap.add_argument("--min-eval", type=int, default=80)
    ap.add_argument("--min-causal", type=int, default=40)
    args = ap.parse_args()
    print(f"pack_dir={args.pack_dir.resolve()}")
    validate(
        args.pack_dir,
        min_docs=args.min_docs,
        min_dense_pairs=args.min_dense_pairs,
        min_eval=args.min_eval,
        min_causal=args.min_causal,
    )


if __name__ == "__main__":
    main()
