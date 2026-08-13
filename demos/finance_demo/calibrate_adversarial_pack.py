"""
Dense calibration loop for hard_adversarial pack.

For each eval query:
  - Rank all documents with a real dense encoder (cosine)
  - If gold rank <= K: harden distractors (inject query lexical anchors)
    and optionally abstract the gold text; re-rank
  - Keep only queries where gold rank > K (confirmed dense misses)

Target: Miss@10 >= 50% of generated eval rows (or as many as survive).

Usage:
  python demos/finance_demo/generate_adversarial_pack.py
  docker compose run --rm vectorprism python demos/finance_demo/calibrate_adversarial_pack.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACK = Path(__file__).resolve().parent / "hard_adversarial"
RESULTS = Path(__file__).resolve().parent / "results" / "hard_eval"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9$%]{3,}", text.lower())


def query_anchors(query: str, top_n: int = 12) -> list[str]:
    stop = {
        "why",
        "did",
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "when",
        "what",
        "which",
        "was",
        "were",
        "does",
        "have",
        "has",
        "than",
        "even",
        "though",
        "without",
        "still",
        "showed",
        "specifically",
        "currently",
        "effective",
        "required",
        "requirement",
    }
    counts: dict[str, int] = {}
    for t in _tokens(query):
        if t in stop or t.isdigit():
            continue
        counts[t] = counts.get(t, 0) + 1
    ranked = sorted(counts, key=lambda x: (-counts[x], -len(x), x))
    return ranked[:top_n]


def harden_distractor(text: str, anchors: list[str]) -> str:
    # Prefix dense-negative bait using query surface forms
    bait = " ".join(anchors[:8])
    prefix = (
        f"Topic keywords frequently searched with this incident include: {bait}. "
        f"Operators investigating those symptoms often open this procedure first. "
    )
    if text.startswith("Topic keywords frequently searched"):
        return text
    return prefix + text


def abstract_gold(text: str, anchors: list[str]) -> str:
    """Reduce direct lexical overlap with the query symptom tokens."""
    out = text
    for a in anchors:
        if len(a) < 5:
            continue
        # Avoid destroying policy IDs excessively; skip tokens with digits mixed oddly
        pattern = re.compile(rf"\b{re.escape(a)}\b", re.IGNORECASE)
        if pattern.search(out):
            out = pattern.sub("the referenced control condition", out, count=2)
    return out


def dense_ranks(
    encoder,
    docs: list[dict],
    queries: list[str],
) -> list[list[tuple[str, float]]]:
    import torch.nn.functional as F

    doc_ids = [d["document_id"] for d in docs]
    doc_texts = [d["chunk_text"] for d in docs]
    # Batch encode
    d_emb = encoder.encode(doc_texts)
    d_emb = F.normalize(d_emb, dim=-1).cpu().numpy()
    q_emb = encoder.encode(queries)
    q_emb = F.normalize(q_emb, dim=-1).cpu().numpy()
    sims = q_emb @ d_emb.T
    rankings = []
    for i in range(len(queries)):
        order = np.argsort(sims[i])[::-1]
        rankings.append([(doc_ids[j], float(sims[i, j])) for j in order])
    return rankings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--target-miss-rate", type=float, default=0.5)
    args = ap.parse_args()

    docs_path = PACK / "documents.jsonl"
    eval_path = PACK / "eval.jsonl"
    if not docs_path.exists() or not eval_path.exists():
        raise SystemExit("Run generate_adversarial_pack.py first")

    import sys

    sys.path.insert(0, str(ROOT))
    from base_encoder import HashingEncoder, SentenceTransformerEncoder

    if args.encoder in {"hash", "hashing", "test"}:
        encoder = HashingEncoder(768)
        print("WARNING: hash encoder is not valid for calibration quality")
    else:
        encoder = SentenceTransformerEncoder(args.encoder)

    docs = load_jsonl(docs_path)
    eval_rows = load_jsonl(eval_path)
    doc_by_id = {d["document_id"]: d for d in docs}

    history = []
    for rnd in range(1, args.max_rounds + 1):
        queries = [r["query"] for r in eval_rows]
        rankings = dense_ranks(encoder, docs, queries)
        miss_ids = []
        hit_ids = []
        detail = []
        for r, ranking in zip(eval_rows, rankings):
            gold = r["relevant_doc_ids"][0]
            rank = next((i + 1 for i, (did, _) in enumerate(ranking) if did == gold), None)
            top = ranking[: args.k]
            entry = {
                "query": r["query"],
                "gold": gold,
                "rank": rank,
                "hit_at_k": rank is not None and rank <= args.k,
                "top_ids": [did for did, _ in top],
                "top_scores": [sc for _, sc in top],
                "gold_score": next((sc for did, sc in ranking if did == gold), None),
            }
            detail.append(entry)
            if entry["hit_at_k"]:
                hit_ids.append(r)
            else:
                miss_ids.append(r)

        miss_rate = len(miss_ids) / max(len(eval_rows), 1)
        history.append({"round": rnd, "n": len(eval_rows), "misses": len(miss_ids), "miss_rate": miss_rate})
        print(f"[calibrate] round={rnd} miss@{args.k}={len(miss_ids)}/{len(eval_rows)} ({miss_rate:.1%})")

        if miss_rate >= args.target_miss_rate:
            print("[calibrate] target miss rate reached")
            break

        # Harden remaining hits
        changed = 0
        for r, entry in zip(eval_rows, detail):
            if not entry["hit_at_k"]:
                continue
            anchors = query_anchors(r["query"])
            gold_id = r["relevant_doc_ids"][0]
            prefix = gold_id.rsplit("_gold", 1)[0]
            # Harden same-cluster distractors
            for did, d in list(doc_by_id.items()):
                if did.startswith(prefix + "_d") or did.startswith(prefix + "_f"):
                    new_text = harden_distractor(d["chunk_text"], anchors)
                    if new_text != d["chunk_text"]:
                        d["chunk_text"] = new_text
                        changed += 1
            # Abstract gold slightly
            g = doc_by_id[gold_id]
            new_g = abstract_gold(g["chunk_text"], anchors)
            if new_g != g["chunk_text"]:
                g["chunk_text"] = new_g
                changed += 1
        docs = [doc_by_id[d["document_id"]] for d in docs]
        print(f"[calibrate] hardened texts touched≈{changed}")
        if changed == 0:
            print("[calibrate] no further hardening possible; stopping")
            break

    # Final filter: keep only confirmed misses in eval; rewrite labels
    queries = [r["query"] for r in eval_rows]
    rankings = dense_ranks(encoder, docs, queries)
    kept = []
    dropped = []
    confirmed_detail = []
    for r, ranking in zip(eval_rows, rankings):
        gold = r["relevant_doc_ids"][0]
        rank = next((i + 1 for i, (did, _) in enumerate(ranking) if did == gold), None)
        hit = rank is not None and rank <= args.k
        r = dict(r)
        r["dense_should_miss"] = not hit
        r["miss_confirmed"] = not hit
        r["dense_rank"] = rank
        if hit:
            r["label_note"] = f"dropped from hard eval; dense rank={rank}"
            dropped.append(r)
        else:
            r["miss_confirm_source"] = f"calibrated_dense_cosine_rank>{args.k}"
            kept.append(r)
            confirmed_detail.append(
                {
                    "query": r["query"],
                    "gold": gold,
                    "dense_rank": rank,
                    "pattern": r.get("pattern"),
                    "channel_hint": r.get("channel_hint"),
                    "top10": [did for did, _ in ranking[:10]],
                }
            )

    # Refresh dense_pairs passages to match hardened gold/distractor texts
    pairs_path = PACK / "dense_pairs.jsonl"
    if pairs_path.exists():
        pairs = load_jsonl(pairs_path)
        for p in pairs:
            sid = p.get("source_doc_id")
            if sid in doc_by_id:
                p["passage"] = doc_by_id[sid]["chunk_text"]
        write_jsonl(pairs_path, pairs)

    # Persist calibrated corpus (hardened docs) + filtered eval
    write_jsonl(docs_path, docs)
    write_jsonl(eval_path, kept)
    # Keep full generated eval archive
    write_jsonl(PACK / "eval_generated_all.jsonl", eval_rows)
    write_jsonl(PACK / "eval_dropped_hits.jsonl", dropped)

    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {
        "encoder": args.encoder,
        "k": args.k,
        "target_miss_rate": args.target_miss_rate,
        "history": history,
        "n_docs": len(docs),
        "n_eval_kept_confirmed_misses": len(kept),
        "n_eval_dropped_dense_hits": len(dropped),
        "confirmed_miss_rate_among_generated": len(kept) / max(len(kept) + len(dropped), 1),
        "confirmed_misses": confirmed_detail,
        "pack_dir": str(PACK),
    }
    (PACK / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS / "adversarial_calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Adversarial pack calibration",
        "",
        f"Encoder: `{args.encoder}`  K={args.k}",
        f"Confirmed dense misses kept: **{len(kept)}** / {len(kept)+len(dropped)} generated "
        f"({report['confirmed_miss_rate_among_generated']:.1%})",
        "",
        "## Rounds",
        "",
    ]
    for h in history:
        md.append(f"- round {h['round']}: miss@{args.k}={h['misses']}/{h['n']} ({h['miss_rate']:.1%})")
    md += ["", "## Kept misses", ""]
    for c in confirmed_detail:
        md.append(f"- rank={c['dense_rank']} `{c['pattern']}` — {c['query'][:100]}...")
    (RESULTS / "ADVERSARIAL_CALIBRATION.md").write_text("\n".join(md), encoding="utf-8")

    print(
        f"[calibrate] DONE kept_misses={len(kept)} dropped_hits={len(dropped)} "
        f"rate={report['confirmed_miss_rate_among_generated']:.1%}"
    )
    print(f"wrote {PACK / 'calibration_report.json'}")
    if len(kept) / max(len(kept) + len(dropped), 1) < args.target_miss_rate:
        print(
            "[calibrate] WARNING: below target miss rate among generated queries; "
            "add more clusters / stronger distractors"
        )


if __name__ == "__main__":
    main()
