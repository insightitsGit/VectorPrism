"""Step 2 — Rewrite dense_should_miss labels from confirmed dense misses.

Reads causal_recovery.json (or dense_baseline_isolated.json) miss lists and
updates hard_gemini/hard_gpt eval.jsonl so labels match measured misses@10.

Also writes:
  demos/finance_demo/results/hard_eval/confirmed_misses_{pack}.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos" / "finance_demo"
RESULTS = DEMO / "results" / "hard_eval"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def miss_queries_from_recovery(pack: str) -> set[str]:
    path = RESULTS / "causal_recovery.json"
    if not path.exists():
        raise SystemExit(f"missing {path} — run run_causal_recovery.py first")
    data = json.loads(path.read_text(encoding="utf-8"))
    pack_data = data["packs"][pack]
    return {q.strip() for q in pack_data.get("all_miss_queries", [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="+", default=["gemini", "gpt"])
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {}

    for pack in args.packs:
        pack_dir = DEMO / f"hard_{pack}"
        eval_path = pack_dir / "eval.jsonl"
        if not eval_path.exists():
            raise SystemExit(f"missing {eval_path}")

        miss_q = miss_queries_from_recovery(pack)
        rows = load_jsonl(eval_path)
        before_true = sum(1 for r in rows if r.get("dense_should_miss"))
        confirmed_rows = []

        for r in rows:
            q = str(r.get("query", "")).strip()
            is_miss = q in miss_q
            r["dense_should_miss"] = bool(is_miss)
            if is_miss:
                r["miss_confirmed"] = True
                r["miss_confirm_source"] = "dense_rank>10_isolated_baseline"
                confirmed_rows.append(
                    {
                        "query": q,
                        "relevant_doc_ids": r.get("relevant_doc_ids"),
                        "channel_hint": r.get("channel_hint"),
                        "miss_reason": r.get("miss_reason"),
                    }
                )
            else:
                r["miss_confirmed"] = False
                # Keep original narrative reason but mark as optimistic label
                if r.get("miss_reason") and "UNCONFIRMED" not in str(r.get("miss_reason")):
                    # only annotate once
                    pass
                r["label_note"] = "dense_should_miss cleared; dense retrieved gold within top-10"

        after_true = sum(1 for r in rows if r.get("dense_should_miss"))
        write_jsonl(eval_path, rows)

        miss_out = RESULTS / f"confirmed_misses_{pack}.jsonl"
        write_jsonl(miss_out, confirmed_rows)

        summary[pack] = {
            "eval_path": str(eval_path),
            "n_eval": len(rows),
            "dense_should_miss_before": before_true,
            "dense_should_miss_after": after_true,
            "confirmed_miss_file": str(miss_out),
        }
        print(
            f"{pack}: dense_should_miss {before_true} -> {after_true} / {len(rows)}  "
            f"wrote {miss_out.name}"
        )

    # Refresh combined eval copies
    merge = ROOT / "scripts" / "merge_hard_eval_packs.py"
    if merge.exists():
        import subprocess
        import sys

        subprocess.check_call([sys.executable, str(merge)], cwd=str(ROOT))
        print("refreshed hard_combined via merge_hard_eval_packs.py")

    out = RESULTS / "label_hardening.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
