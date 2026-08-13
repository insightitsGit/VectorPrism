"""Convert a partner (or public) docs+queries pack into VectorPrism JSONL.

Supported presets:
  incident_bench  — HuggingFace trentdoney/synthetic-incident-search-benchmark
  generic         — already-VectorPrism-shaped documents.jsonl + eval.jsonl

Output layout (under --out):
  documents.jsonl
  eval.jsonl
  meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def convert_incident_bench(raw_dir: Path, out_dir: Path) -> dict[str, Any]:
    incidents = [json.loads(l) for l in (raw_dir / "incidents.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    queries = [json.loads(l) for l in (raw_dir / "queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    relevance = {
        json.loads(l)["query_id"]: json.loads(l)
        for l in (raw_dir / "relevance.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }

    docs = []
    for inc in incidents:
        did = str(inc["incident_id"])
        parts = [
            str(inc.get("title", "")),
            f"Severity: {inc.get('severity', 'unknown')}.",
            f"Systems: {', '.join(inc.get('systems') or [])}.",
            f"Services: {', '.join(inc.get('services') or [])}.",
            f"Symptoms: {'; '.join(inc.get('symptoms') or [])}.",
            f"Root cause: {inc.get('root_cause', '')}",
            f"Resolution: {inc.get('resolution', '')}",
            str(inc.get("body", "")),
        ]
        docs.append(
            {
                "document_id": did,
                "chunk_text": " ".join(p for p in parts if p).strip(),
                "epistemic_truth": 1.0,
                "severity": inc.get("severity"),
                "tags": inc.get("tags") or [],
            }
        )

    eval_rows = []
    for q in queries:
        qid = q["query_id"]
        rel = relevance.get(qid, {})
        gold = [str(x) for x in rel.get("relevant_incident_ids") or []]
        if not gold:
            continue
        eval_rows.append(
            {
                "query": str(q["query"]),
                "relevant_doc_ids": gold,
                "query_id": qid,
                "intent": q.get("intent"),
                "difficulty": q.get("difficulty"),
                "hard_negative_doc_ids": [str(x) for x in rel.get("hard_negative_incident_ids") or []],
                "rationale": rel.get("rationale"),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "documents.jsonl", docs)
    _write_jsonl(out_dir / "eval.jsonl", eval_rows)
    meta = {
        "source": "trentdoney/synthetic-incident-search-benchmark",
        "license_note": "Public Hugging Face dataset; synthetic incidents (not real customer data).",
        "n_documents": len(docs),
        "n_eval": len(eval_rows),
        "vertical": "SRE / incident-response retrieval",
        "why_this_pack": (
            "Public-safe operational incident notes with labeled queries and hard negatives. "
            "Not constructed by VectorPrism authors to force dense failure."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Import external corpus into VectorPrism audit pack")
    ap.add_argument("--preset", choices=["incident_bench"], default="incident_bench")
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.preset == "incident_bench":
        meta = convert_incident_bench(args.raw_dir, args.out)
    else:
        raise SystemExit(f"unknown preset {args.preset}")
    print(json.dumps(meta, indent=2))
    print(f"Wrote pack under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
