"""Merge GPT + Gemini hard packs for denser training while keeping evals separate.

Writes:
  demos/finance_demo/hard_combined/
    documents.jsonl   # union (IDs do not collide: vp_* vs DOC_*)
    dense_pairs.jsonl # union
    causal.jsonl      # union
    eval_gemini.jsonl # Gemini hard eval only
    eval_gpt.jsonl    # GPT hard eval only
    README.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("demos/finance_demo")
GPT = ROOT / "hard_gpt"
GEMINI = ROOT / "hard_gemini"
OUT = ROOT / "hard_combined"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    gpt_docs = load_jsonl(GPT / "documents.jsonl")
    gem_docs = load_jsonl(GEMINI / "documents.jsonl")
    gpt_ids = {d["document_id"] for d in gpt_docs}
    gem_ids = {d["document_id"] for d in gem_docs}
    overlap = gpt_ids & gem_ids
    if overlap:
        raise SystemExit(f"ID collision between packs: {sorted(overlap)[:10]}")

    docs = gpt_docs + gem_docs
    pairs = load_jsonl(GPT / "dense_pairs.jsonl") + load_jsonl(GEMINI / "dense_pairs.jsonl")
    causal = load_jsonl(GPT / "causal.jsonl") + load_jsonl(GEMINI / "causal.jsonl")
    eval_g = load_jsonl(GEMINI / "eval.jsonl")
    eval_p = load_jsonl(GPT / "eval.jsonl")

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "documents.jsonl", docs)
    write_jsonl(OUT / "dense_pairs.jsonl", pairs)
    write_jsonl(OUT / "causal.jsonl", causal)
    write_jsonl(OUT / "eval_gemini.jsonl", eval_g)
    write_jsonl(OUT / "eval_gpt.jsonl", eval_p)
    # Default eval.jsonl = Gemini (primary claim pack)
    write_jsonl(OUT / "eval.jsonl", eval_g)

    readme = f"""# Combined hard-eval training pack

Merged from `hard_gpt` + `hard_gemini` for denser Stage-1 training.

| File | Rows | Role |
|------|------|------|
| documents.jsonl | {len(docs)} | Union corpus (GPT {len(gpt_docs)} + Gemini {len(gem_docs)}) |
| dense_pairs.jsonl | {len(pairs)} | Union dense supervision |
| causal.jsonl | {len(causal)} | Union causal pairs |
| eval.jsonl | {len(eval_g)} | **Primary scorecard = Gemini** |
| eval_gemini.jsonl | {len(eval_g)} | Gemini hard eval |
| eval_gpt.jsonl | {len(eval_p)} | GPT hard eval |

## Recommended protocol (2 runs, 1 train)

1. Train dense / channels on `hard_combined` (documents + dense_pairs + causal).
2. **Run A (primary):** evaluate on `eval_gemini.jsonl` — realistic banking traps.
3. **Run B (secondary):** evaluate on `eval_gpt.jsonl` — templated cause/symptom traps.
4. Report both scorecards; do not average into one vanity number.

Do not naive-merge eval sets into one ranking without labeling which pack a query came from.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"docs={len(docs)} pairs={len(pairs)} causal={len(causal)}")
    print(f"eval_gemini={len(eval_g)} eval_gpt={len(eval_p)}")


if __name__ == "__main__":
    main()
