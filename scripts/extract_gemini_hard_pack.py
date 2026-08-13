"""Extract Gemini hard-eval pack from the agent transcript into demos/finance_demo/hard_gemini/."""
from __future__ import annotations

import json
import re
from pathlib import Path

TRANSCRIPT = Path(
    r"C:\Users\parva\.cursor\projects\c-code-VectorPrism\agent-transcripts"
    r"\ef0b93c5-3c7e-45a2-b15e-9f2ebd0b1e13\ef0b93c5-3c7e-45a2-b15e-9f2ebd0b1e13.jsonl"
)
OUT_DIR = Path("demos/finance_demo/hard_gemini")

SECTIONS = {
    "documents": r"### 1\) documents\.jsonl\n\n(.*?)(?=\n### 2\)|\Z)",
    "dense_pairs": r"### 2\) dense_pairs\.jsonl\n\n(.*?)(?=\n### 3\)|\Z)",
    "eval": r"### 3\) eval\.jsonl\n\n(.*?)(?=\n### 4\)|\Z)",
    "causal": r"### 4\) causal\.jsonl\n\n(.*?)(?=\n### Summary|\Z)",
}


def main() -> None:
    text = None
    for line in TRANSCRIPT.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("role") != "user":
            continue
        content = obj.get("message", {}).get("content", [])
        for part in content:
            if part.get("type") == "text" and "DOC_AML_001" in part.get("text", ""):
                text = part["text"]
                break
        if text:
            break
    if not text:
        raise SystemExit("Gemini pack not found in transcript")

    idx = text.find("### 1) documents.jsonl")
    if idx < 0:
        raise SystemExit("documents section not found")
    text = text[idx:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, pat in SECTIONS.items():
        m = re.search(pat, text, flags=re.S)
        if not m:
            raise SystemExit(f"section missing: {name}")
        body = m.group(1).strip()
        rows = []
        bad = 0
        for i, line in enumerate(body.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                bad += 1
                print(f"parse error {name}:{i}: {e}")
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {path} rows={len(rows)} bad={bad}")

    docs = [
        json.loads(l)
        for l in (OUT_DIR / "documents.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    doc_set = {d["document_id"] for d in docs}
    pairs = [
        json.loads(l)
        for l in (OUT_DIR / "dense_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    ev = [
        json.loads(l)
        for l in (OUT_DIR / "eval.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    causal = [
        json.loads(l)
        for l in (OUT_DIR / "causal.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    orph = [did for r in ev for did in r.get("relevant_doc_ids", []) if did not in doc_set]
    pair_bad = [r.get("source_doc_id") for r in pairs if r.get("source_doc_id") not in doc_set]
    print("integrity docs/pairs/eval/causal", len(docs), len(pairs), len(ev), len(causal))
    print("eval orphans", len(orph), "pair bad", len(pair_bad))
    print(
        "dense_should_miss",
        sum(1 for r in ev if r.get("dense_should_miss")),
        "/",
        len(ev),
    )


if __name__ == "__main__":
    main()
