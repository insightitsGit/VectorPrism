"""
Expand hard_adversarial causal supervision + doc-id graph for Stage-1 expansion.

Writes:
  hard_adversarial/causal.jsonl       — earlier/later pairs for train.py (60–80+)
  hard_adversarial/causal_graph.jsonl — {earlier_doc_id, later_doc_id, hops_hint}

Graph convention: earlier_doc_id (cause/gold) → later_doc_id (symptom/distractor).
Stage-1 expansion walks UPSTREAM from dense seeds (symptom docs) to pull gold causes.
"""

from __future__ import annotations

import json
from pathlib import Path

PACK = Path(__file__).resolve().parent / "hard_adversarial"

# Multi-hop chains: (prefix, trigger, intermediate, symptom, train_variants)
CHAINS: list[dict] = [
    {
        "prefix": "ADV_C01",
        "trigger": "Scheduled sovereign bond haircut revaluation reduces unencumbered collateral value at the noon risk cycle",
        "mid": "Desk Intraday Credit Ceiling falls below open gross exposure after collateral revaluation",
        "symptom": "London trading desk aggressive order routing halt at 14:15 while FIX sessions still show connected",
    },
    {
        "prefix": "ADV_C02",
        "trigger": "Covered counterparty misses undisputed variation margin transfer deadline under the CSA",
        "mid": "Credit Operations imposes temporary financing freeze on the prime brokerage credit line",
        "symptom": "Friday morning DVP settlement lock despite matched trade economics",
    },
    {
        "prefix": "ADV_C03",
        "trigger": "Customer profile telephone number change or carrier SIM replacement is recorded",
        "mid": "Fraud engine classifies a high-risk account-takeover sequence within twenty-four hours",
        "symptom": "Outbound wire freeze on a wealth account despite valid wire request and matching MFA",
    },
    {
        "prefix": "ADV_C04",
        "trigger": "Morning large customer payment releases push consolidated daylight overdraft above ninety percent of the uncollateralized cap",
        "mid": "Uncollateralized net debit capacity is exhausted pending Fedwire Securities Service collateral pledge",
        "symptom": "Payment Queue Manager pauses large customer Fedwire releases despite sufficient customer ledger balances",
    },
    {
        "prefix": "ADV_C05",
        "trigger": "WAN partition severs Datacenter Alpha quorum contact with peer site and cloud Witness",
        "mid": "Active-active majority lock cannot be established for master write authority",
        "symptom": "Datacenter Alpha rejects settlement writes and flips database engines to read-only during the network event",
    },
    {
        "prefix": "ADV_C06",
        "trigger": "Central counterparty issues an electronic intraday variation margin call during a volatility spike",
        "mid": "Clearing Operations fails to pledge eligible cash collateral via Fedwire within sixty minutes",
        "symptom": "LCH declares clearing-member default even though overnight margin balances looked sufficient",
    },
    {
        "prefix": "ADV_C07",
        "trigger": "Primary consumer group lag on topic tx-ledger-events exceeds five hundred thousand uncommitted messages",
        "mid": "Core ledger balance updater risks missing market-close throughput targets",
        "symptom": "Non-critical audit consumers are disconnected from the Kafka cluster during market-close processing",
    },
    {
        "prefix": "ADV_R01",
        "trigger": "Commercial Fedwire instruction exactly matches Standing Settlement Instructions authenticated within 180 days",
        "mid": "Payments exception matrix classifies the wire as repetitive SSI-eligible",
        "symptom": "Seven-million-dollar commercial outbound Fedwire releases without out-of-band voice callback",
    },
    {
        "prefix": "ADV_R02",
        "trigger": "Inbound payment references a comprehensively sanctioned geography without SDN party or SDN property interest",
        "mid": "Sanctions disposition matrix selects Reject rather than Block",
        "symptom": "Incoming payment is returned across the clearing chain instead of landing in an interest-bearing blocked account",
    },
    {
        "prefix": "ADV_R03",
        "trigger": "GMRA margin call notice is served after 10:00 AM London time",
        "mid": "Repo timing matrix rolls collateral delivery to the next business day",
        "symptom": "Repo margin call notice is rejected for same-day settlement at 10:45 AM London time",
    },
    {
        "prefix": "ADV_R04",
        "trigger": "Secondary-market trade involves grandfathered debt of an SSI-listed name without extending prohibited credit",
        "mid": "Sectoral sanctions trading matrix permits settlement without full asset blocking",
        "symptom": "Screening allows short-tenor SSI-listed notes to settle without blocking funds",
    },
    {
        "prefix": "ADV_H01",
        "trigger": "Unlisted holding company is nested beneath a Tier-2 discretionary trust structure",
        "mid": "Fiduciary child-node rule 4.3.b requires Protector acknowledgment and Regional Fiduciary Counsel countersignature",
        "symptom": "Operations asks which approval applies for an unlisted holding company on the Tier-2 discretionary trust onboarding path",
    },
    {
        "prefix": "ADV_E01",
        "trigger": "Phase 6 counterparty is classified in-scope for Uncleared Margin Rules Initial Margin",
        "mid": "Active segregation standard mandates unaffiliated third-party custodian under an Account Control Agreement",
        "symptom": "Question about currently effective Initial Margin segregation requirement for Phase 6 covered counterparties",
    },
    {
        "prefix": "ADV_E02",
        "trigger": "Published fund NAV error is measured at or above fifty basis points of correct NAV",
        "mid": "Active fund accounting error standard requires investor-level remediation",
        "symptom": "Need for full retrospective shareholder transaction reprocessing and direct investor compensation",
    },
]


def _variants(a: str, b: str) -> list[tuple[str, str]]:
    return [
        (a, b),
        (f"Upstream event: {a}", f"Downstream control outcome: {b}"),
        (f"Root condition — {a}", f"Observable result — {b}"),
        (f"Because {a[0].lower() + a[1:]}", f"Therefore {b[0].lower() + b[1:]}"),
    ]


def main() -> None:
    docs = {
        json.loads(l)["document_id"]
        for l in (PACK / "documents.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }

    pairs: list[dict] = []
    edges: list[dict] = []

    for ch in CHAINS:
        p = ch["prefix"]
        gold = f"{p}_gold"
        if gold not in docs:
            print(f"skip missing gold {gold}")
            continue

        # Text hops for training density
        for a, b in _variants(ch["trigger"], ch["mid"]):
            pairs.append(
                {
                    "earlier": a,
                    "later": b,
                    "earlier_doc_id": gold,
                    "later_doc_id": None,
                    "hop": "trigger_to_mid",
                    "cluster": p,
                }
            )
        for a, b in _variants(ch["mid"], ch["symptom"]):
            pairs.append(
                {
                    "earlier": a,
                    "later": b,
                    "earlier_doc_id": gold,
                    "later_doc_id": None,
                    "hop": "mid_to_symptom",
                    "cluster": p,
                }
            )
        # Full span paraphrase
        pairs.append(
            {
                "earlier": ch["trigger"],
                "later": ch["symptom"],
                "earlier_doc_id": gold,
                "later_doc_id": None,
                "hop": "trigger_to_symptom",
                "cluster": p,
            }
        )

        # Doc graph: gold (cause) → symptom distractors (later)
        for i in range(1, 9):
            did = f"{p}_d{i:02d}"
            if did not in docs:
                continue
            edges.append(
                {
                    "earlier_doc_id": gold,
                    "later_doc_id": did,
                    "relation": "causes_symptom_neighborhood",
                    "cluster": p,
                }
            )
            # Also add a training pair tying gold policy language to distractor symptom language
            pairs.append(
                {
                    "earlier": ch["trigger"],
                    "later": ch["symptom"] if i <= 3 else f"Related operational symptom cluster for {p}: {ch['symptom']}",
                    "earlier_doc_id": gold,
                    "later_doc_id": did,
                    "hop": "gold_to_distractor",
                    "cluster": p,
                }
            )

    # Deduplicate training pairs by (earlier, later)
    seen = set()
    uniq_pairs = []
    for r in pairs:
        key = (r["earlier"], r["later"])
        if key in seen:
            continue
        seen.add(key)
        uniq_pairs.append(r)

    causal_path = PACK / "causal.jsonl"
    graph_path = PACK / "causal_graph.jsonl"
    with causal_path.open("w", encoding="utf-8") as f:
        for r in uniq_pairs:
            # Trainer only requires earlier/later; keep ids as optional metadata
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with graph_path.open("w", encoding="utf-8") as f:
        for r in edges:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {causal_path} n={len(uniq_pairs)}")
    print(f"wrote {graph_path} n={len(edges)}")
    print(f"clusters_linked={len({e['cluster'] for e in edges})}")


if __name__ == "__main__":
    main()
