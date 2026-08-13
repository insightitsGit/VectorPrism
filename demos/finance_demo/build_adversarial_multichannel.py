"""
Build hyperbolic + relational supervision and Stage-1 indexes for hard_adversarial.

Outputs:
  hyperbolic.jsonl       — train format {parent, child, negatives} (text)
  hyperbolic_graph.jsonl — {parent_doc_id, child_doc_id} for Stage-1 lineage
  relational.jsonl       — train format TransE triples
  relational_attrs.jsonl — {doc_id, attributes} for predicate matching
"""

from __future__ import annotations

import json
from pathlib import Path

PACK = Path(__file__).resolve().parent / "hard_adversarial"


def load_docs() -> dict[str, str]:
    out = {}
    for line in (PACK / "documents.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["document_id"]] = r["chunk_text"]
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    docs = load_docs()
    prefixes = sorted({d.rsplit("_", 1)[0] for d in docs if d.endswith("_gold")})

    hyp_train: list[dict] = []
    hyp_graph: list[dict] = []
    rel_train: list[dict] = []
    rel_attrs: list[dict] = []

    # --- Taxonomy / hyperbolic clusters ---
    hyp_focus = ["ADV_H01", "ADV_H02", "ADV_H03", "ADV_E01", "ADV_E02", "ADV_E03"]
    for p in prefixes:
        gold = f"{p}_gold"
        if gold not in docs:
            continue
        root = f"{p}_f01" if f"{p}_f01" in docs else f"{p}_d15"
        if root not in docs:
            continue
        children = [gold] + [f"{p}_d{i:02d}" for i in range(1, 16) if f"{p}_d{i:02d}" in docs]
        for ch in children:
            hyp_graph.append({"parent_doc_id": root, "child_doc_id": ch, "relation": "policy_child"})
        # Train pairs: parent text / gold child vs sibling negatives
        if p in hyp_focus or p.startswith("ADV_H") or p.startswith("ADV_E"):
            negs = [docs[f"{p}_d{i:02d}"][:240] for i in range(1, 6) if f"{p}_d{i:02d}" in docs]
            if negs:
                hyp_train.append(
                    {
                        "parent": docs[root][:300],
                        "child": docs[gold][:300],
                        "negatives": negs,
                        "parent_doc_id": root,
                        "child_doc_id": gold,
                    }
                )
                # deeper child emphasis for H01-style
                hyp_train.append(
                    {
                        "parent": f"Parent policy node for {p} discretionary / product taxonomy",
                        "child": docs[gold][:300],
                        "negatives": negs,
                        "parent_doc_id": root,
                        "child_doc_id": gold,
                    }
                )

    # --- Relational attributes + TransE triples ---
    attr_specs = {
        "ADV_R01_gold": {
            "amount_min": 5_000_000,
            "is_repetitive": True,
            "requires_callback": False,
            "transfer_type": "commercial_fedwire",
        },
        "ADV_R02_gold": {
            "disposition": "reject",
            "sanctions_list": "geography_only",
            "sdn_property_interest": False,
        },
        "ADV_R03_gold": {
            "timezone": "london",
            "cutoff_hour": 10,
            "same_day_if_before_cutoff": True,
            "agreement": "gmra",
        },
        "ADV_R04_gold": {
            "sanctions_list": "ssi",
            "allow_secondary_grandfathered": True,
            "disposition": "allow_settle",
        },
        "ADV_E01_gold": {
            "umr_phase": 6,
            "im_segregation": "third_party_aca",
            "doc_status": "active",
        },
        "ADV_E02_gold": {
            "nav_error_bps_min": 50,
            "shareholder_reprocessing": True,
            "doc_status": "active",
        },
        "ADV_H01_gold": {
            "trust_tier": 2,
            "entity_listing": "unlisted",
            "requires_protector_ack": True,
        },
    }
    # Distractor contrasting attrs (near-misses that fail ≥1 predicate)
    distractor_specs = {
        "ADV_R01": {
            "amount_min": 5_000_000,
            "is_repetitive": False,
            "requires_callback": True,
            "transfer_type": "commercial_fedwire",
        },
        "ADV_R02": {
            "disposition": "block",
            "sanctions_list": "sdn",
            "sdn_property_interest": True,
        },
        "ADV_R03": {
            "timezone": "est",
            "cutoff_hour": 10,
            "same_day_if_before_cutoff": True,
            "agreement": "csa",
        },
        "ADV_R04": {
            "sanctions_list": "sdn",
            "allow_secondary_grandfathered": False,
            "disposition": "block",
        },
        "ADV_E01": {
            "umr_phase": 6,
            "im_segregation": "omnibus",
            "doc_status": "superseded",
        },
        "ADV_H01": {
            "trust_tier": 2,
            "entity_listing": "listed",
            "requires_protector_ack": False,
        },
        "ADV_E02": {
            "nav_error_bps_min": 100,
            "shareholder_reprocessing": False,
            "doc_status": "active",
        },
    }
    for p, attrs in distractor_specs.items():
        for i in range(1, 16):
            did = f"{p}_d{i:02d}"
            if did in docs:
                rel_attrs.append({"doc_id": did, "attributes": dict(attrs)})

    for did, attrs in attr_specs.items():
        if did in docs:
            rel_attrs.append({"doc_id": did, "attributes": attrs})

    # TransE-style text triples (trainer format)
    triples = [
        ("Repetitive SSI commercial Fedwire", "waives", "out-of-band voice callback", "mandatory voice callback for all wires over five million"),
        ("Seven million commercial Fedwire with authenticated SSI", "exempt_from", "phone callback", "dual authorization only without SSI match"),
        ("Comprehensively sanctioned geography without SDN interest", "requires", "Reject disposition", "Block into interest-bearing account"),
        ("GMRA margin notice after 10:00 AM London", "rolls_to", "next business day delivery", "same-day settlement under 10:00 AM EST CSA"),
        ("SSI-listed grandfathered secondary debt", "allows", "settlement without blocking funds", "immediate SDN asset freeze"),
        ("Phase 6 Initial Margin active standard", "requires", "third-party ACA segregation", "omnibus house Initial Margin"),
        ("NAV error at or above fifty basis points", "requires", "full shareholder reprocessing", "fund-level restatement only"),
        ("Unlisted holdco under Tier-2 discretionary trust", "requires", "Protector and Regional Fiduciary Counsel approval", "parent Tier-2 trust checklist only"),
    ]
    for subj, rel, obj, neg in triples:
        rel_train.append(
            {"subject": subj, "relation": rel, "object": obj, "negative_object": neg}
        )
        # paraphrase variants
        rel_train.append(
            {
                "subject": f"Policy case: {subj}",
                "relation": rel,
                "object": obj,
                "negative_object": neg,
            }
        )

    write_jsonl(PACK / "hyperbolic.jsonl", hyp_train)
    write_jsonl(PACK / "hyperbolic_graph.jsonl", hyp_graph)
    write_jsonl(PACK / "relational.jsonl", rel_train)
    write_jsonl(PACK / "relational_attrs.jsonl", rel_attrs)
    print(
        f"hyperbolic_train={len(hyp_train)} hyp_graph={len(hyp_graph)} "
        f"rel_train={len(rel_train)} rel_attrs={len(rel_attrs)}"
    )


if __name__ == "__main__":
    main()
