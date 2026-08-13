"""
Production smoke: capture a single readiness report for VectorPrism.

Runs (inside Docker recommended):
  - pytest (optional --full)
  - finance pgvector live search (optional --full / default if DB up)
  - Stage-2 latency benchmark
  - Writes production_readiness.json + PRODUCTION_RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO = Path(__file__).resolve().parent
RESULTS = DEMO / "results"
sys.path.insert(0, str(ROOT))


def _run(cmd: list, cwd: Path = ROOT) -> tuple[int, str]:
    print("+", " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if out.strip():
        print(out[-4000:] if len(out) > 4000 else out)
    return p.returncode, out


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="Also run pytest + pgvector demo")
    ap.add_argument("--dsn", default=os.environ.get(
        "VECTORPRISM_PG_DSN",
        "postgresql://vectorprism:vectorprism@db:5432/vectorprism",
    ))
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "finance_demo.pt"))
    ap.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    args = ap.parse_args(argv)

    RESULTS.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    report: dict = {
        "started_at_utc": started,
        "encoder": args.encoder,
        "checkpoint": args.checkpoint,
        "dsn": args.dsn,
        "steps": {},
        "artifacts": {},
        "production_ready_pre_client": True,
        "notes": [
            "Synthetic finance corpus — replace with client data before claiming client production quality.",
            "Hard epistemic truth filter remains opt-in until ECE-calibrated.",
        ],
    }

    # 1) Tests
    if args.full:
        code, out = _run([sys.executable, "-m", "pytest", "test_psm.py", "test_phases.py", "-q"])
        report["steps"]["pytest"] = {"exit_code": code, "pass": code == 0}
        if code != 0:
            report["production_ready_pre_client"] = False

    # 2) pgvector live path
    if args.full or os.environ.get("VECTORPRISM_PG_DSN"):
        code, out = _run([
            sys.executable, str(DEMO / "run_pgvector_demo.py"),
            "--dsn", args.dsn,
            "--checkpoint", args.checkpoint,
            "--encoder", args.encoder,
            "--skip-train",
        ])
        report["steps"]["pgvector_demo"] = {"exit_code": code, "pass": code == 0}
        if code != 0:
            report["production_ready_pre_client"] = False

    # 3) Latency benchmark (encode_query + search; real encoder is slower than Stage-2-only SLA)
    code, out = _run([
        sys.executable, str(ROOT / "live_benchmark.py"),
        "--checkpoint", args.checkpoint,
        "--documents", str(DEMO / "documents.jsonl"),
        "--encoder", args.encoder,
        "--backend", "memory",
        "--n-trials", "12",
        "--p95-budget-ms", "5000",
    ])
    report["steps"]["live_benchmark"] = {
        "exit_code": code,
        "pass": code == 0,
        "tail": out[-1500:],
    }
    if code != 0:
        report["production_ready_pre_client"] = False

    # 4) Collect existing artifacts
    for name in [
        "phase1_eval.json",
        "pgvector_live_search.json",
        "finance_demo.checkpoint.json",
    ]:
        path = RESULTS / name
        if path.exists():
            report["artifacts"][name] = json.loads(path.read_text(encoding="utf-8"))

    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["git"] = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(ROOT), capture_output=True, text=True,
    ).stdout.strip()

    out_json = RESULTS / "production_readiness.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Markdown summary
    phase1 = report["artifacts"].get("phase1_eval.json", {})
    pg = report["artifacts"].get("pgvector_live_search.json", {})
    md = f"""# VectorPrism Production Results

Generated (UTC): `{report['finished_at_utc']}`  
Git: `{report.get('git', 'n/a')}`  
Encoder: `{args.encoder}`  
Checkpoint: `{args.checkpoint}`

## Gate summary

| Step | Pass |
|---|---|
| pytest | {report['steps'].get('pytest', {}).get('pass', 'skipped')} |
| pgvector live demo | {report['steps'].get('pgvector_demo', {}).get('pass', 'skipped')} |
| live benchmark | {report['steps'].get('live_benchmark', {}).get('pass', False)} |
| **Pre-client production ready** | **{report['production_ready_pre_client']}** |

## Phase-1 eval (finance demo)

```json
{json.dumps(phase1, indent=2)}
```

## pgvector live search

- Document count: `{pg.get('document_count', 'n/a')}`
- Model version: `{pg.get('model_version', 'n/a')}`
- Queries exercised: `{len(pg.get('queries', []))}`

See `pgvector_live_search.json` for full hit lists.

## Notes

{chr(10).join('- ' + n for n in report['notes'])}

## Reproduce

```bash
docker compose up -d db
docker compose run --rm test
docker compose run --rm finance-pg
docker compose run --rm vectorprism python demos/finance_demo/run_production_smoke.py --full
```
"""
    out_md = RESULTS / "PRODUCTION_RESULTS.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"[production-smoke] Wrote {out_json}")
    print(f"[production-smoke] Wrote {out_md}")
    print(f"[production-smoke] production_ready_pre_client={report['production_ready_pre_client']}")
    return 0 if report["production_ready_pre_client"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
