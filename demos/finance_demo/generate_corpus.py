"""
Generate a near-realistic finance RAG demo corpus for VectorPrism Phase-1.

Produces:
  documents.jsonl   — searchable knowledge base chunks
  dense_pairs.jsonl — {query, passage} training pairs (hundreds)
  eval.jsonl        — held-out {query, relevant_doc_ids}
  causal.jsonl      — optional ordered event pairs (incident RCA)

This is a **demo corpus** for client walkthroughs — not a claim of production
quality on a live bank dataset. Swap these files for the client's real docs
when available.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Document catalog: (doc_id, title/topic, body)
# Bodies are written as standalone retrieval chunks (policy / runbook style).
# ---------------------------------------------------------------------------
DOCS = [
    ("pol_kyc_01", "KYC refresh must complete within 30 calendar days of trigger events including beneficial ownership changes above 25 percent."),
    ("pol_kyc_02", "Enhanced due diligence is mandatory for PEPs and high-risk jurisdictions before onboarding retail or corporate accounts."),
    ("pol_aml_01", "Suspicious Activity Reports must be filed within 30 days of detection when aggregated wire activity exceeds the internal SAR threshold."),
    ("pol_aml_02", "Transaction monitoring alerts aged beyond 5 business days require dual control review by AML Investigations and Compliance."),
    ("pol_wire_01", "International wires above USD 50,000 require callback verification to a registered phone number on file."),
    ("pol_wire_02", "Same-day wire release after 3 PM ET is blocked unless Treasury Operations override with dual approval."),
    ("pol_margin_01", "Maintenance margin calls must be met by 2 PM ET on T+1 or positions are subject to liquidation."),
    ("pol_margin_02", "House margin requirements for concentrated equity positions are 40 percent of market value."),
    ("pol_settle_01", "Failed equity settlements are auto-retried at 10 AM ET; persistent fails escalate to Settlement Ops after two retries."),
    ("pol_settle_02", "DTC partial settlements require allocation updates in the books-and-records system before end-of-day NAV lock."),
    ("pol_nav_01", "Official fund NAV is struck at 4 PM ET using final pricing; late-arriving prices use fair-value hierarchy Level 2 when available."),
    ("pol_nav_02", "NAV error corrections above 0.5 percent of prior NAV require Fund Accounting Director approval and investor notice."),
    ("pol_liq_01", "Liquidity Coverage Ratio breach warnings trigger Treasury liquidity contingency playbook within 2 hours."),
    ("pol_liq_02", "Redemption gates for the short-duration credit fund activate if weekly outflows exceed 15 percent of AUM."),
    ("pol_cpty_01", "Counterparty exposure limits for prime brokers are set at 10 percent of Tier 1 capital and reviewed quarterly."),
    ("pol_cpty_02", "ISDA CSA disputes above USD 1 million escalate to Collateral Management and Legal within one business day."),
    ("pol_sox_01", "SOX 404 key control FR-CTRL-17 requires monthly reconciliation of suspense accounts with dual sign-off."),
    ("pol_sox_02", "Privileged access to the general ledger must be recertified quarterly by Control Owners."),
    ("pol_mkt_01", "Interest-rate DV01 limits for the rates desk are USD 250,000 per 1bp; breaches page Risk Desk immediately."),
    ("pol_mkt_02", "VaR model exceptions above 4 in a rolling 20-day window require Model Risk Management review."),
    ("run_inc_01", "Incident runbook: when trade capture latency exceeds 2 seconds p95, disable non-critical enrichments and fail open to STP."),
    ("run_inc_02", "Incident runbook: matching engine freeze — switch to secondary matcher, notify Market Structure, freeze new order intake."),
    ("run_inc_03", "Root-cause pattern: overnight batch pricing job overrun delays NAV; mitigation is to pre-warm FX rates at 3:30 PM ET."),
    ("run_inc_04", "Root-cause pattern: wire callback queue causes release delay; verify CRM phone sync jobs completed before 9 AM."),
    ("run_inc_05", "Root-cause pattern: failed settlement cascade from short inventory; borrow desk must locate shares before 11 AM ET."),
    ("faq_tax_01", "Form 1099-B cost basis uses average cost for mutual funds unless the client elects specific identification in writing."),
    ("faq_tax_02", "Wash-sale losses are disallowed when substantially identical securities are purchased within 30 days before or after the sale."),
    ("faq_acct_01", "Clients can enable ACH pulls for margin deficits; ACH rejects re-open the margin call as unmet."),
    ("faq_acct_02", "Corporate action elections close at 2 PM ET on election day; default is DTC mandatory instruction."),
    ("faq_esg_01", "Article 8 funds disclose principal adverse impacts annually; Article 9 funds must evidence sustainable investment objectives."),
    ("faq_esg_02", "Green bond proceeds tracking uses earmarked ledgers; unallocated proceeds older than 24 months require Sustainability Committee review."),
    ("risk_credit_01", "Issuer downgrade below BB- forces automatic review of hold-to-maturity eligibility in the credit book."),
    ("risk_credit_02", "Single-name CDS notional cannot exceed USD 50 million without Credit Committee approval."),
    ("risk_ops_01", "Reconciliation breaks above USD 100,000 aged over 3 days are material and reported to the CFO dashboard."),
    ("risk_ops_02", "STP exception rate above 3 percent for a product line triggers Operations quality review."),
    ("legal_reg_01", "Reg BI suitability documentation must be retained for six years and available for SEC exam within 48 hours."),
    ("legal_reg_02", "MiFID II best-execution reports are published quarterly for EEA clients with systematic internaliser flow."),
    ("legal_reg_03", "GDPR data subject access requests for EU clients must be fulfilled within 30 days via the Privacy Office."),
    ("tech_data_01", "Golden source for legal entity identifiers is the GLEIF feed refreshed nightly into the MDM hub."),
    ("tech_data_02", "Trade blotter Kafka topic trades.raw.v3 is authoritative for intraday risk; JDBC warehouse is T+1 only."),
    ("tech_sec_01", "Privileged break-glass access to payment systems requires Cyber Security approval and expires in 4 hours."),
    ("tech_sec_02", "API keys for market-data vendors rotate every 90 days; expired keys cause silent quote stalls on the UI."),
    ("desk_eq_01", "Equity cash desk soft blocks orders that would exceed 5 percent of 20-day ADV without Sales Trader override."),
    ("desk_fi_01", "Corporate bond RFQ responses older than 90 seconds are considered stale and must be refreshed before hit."),
    ("desk_fx_01", "NDF fixings use the WM/R 4 PM London rate unless the ticket specifies an alternate fix source."),
    ("ops_rec_01", "Cash breaks between custodian and books are aged daily; aged over 5 days require Ops Manager comment."),
    ("ops_rec_02", "Position breaks on listed options after expiration must be cleared before next trading open."),
    ("client_svc_01", "Tier-1 institutional clients receive 15-minute response SLA for trade query tickets during market hours."),
    ("client_svc_02", "Complaint handling clock starts at first written notice; acknowledgment within 2 business days is mandatory."),
    ("audit_01", "Internal Audit tests wire dual-control annually; sample size is 25 high-value wires plus all overrides."),
    ("audit_02", "Model inventory attestation is due each January 31 for all production pricing and risk models."),
    ("treasury_01", "Intraday liquidity buffer target is USD 500 million in HQLA; usage above 70 percent pages Treasury."),
    ("treasury_02", "Fedwire cutoff for third-party USD wires is 5:45 PM ET; internal book transfers may settle later."),
    ("fraud_01", "New payee creation followed by a wire within 24 hours is auto-escalated to Fraud Operations."),
    ("fraud_02", "Voice-phish indicators include urgency language and requests to bypass callback; freeze the payment instruction."),
    ("hr_comp_01", "Material risk takers have deferral of at least 40 percent of variable compensation over three years."),
    ("hr_comp_02", "Personal trading pre-clearance is required for covered employees before any equity transaction."),
    ("vendor_01", "Critical payment vendors require annual SOC 2 Type II review by Third-Party Risk Management."),
    ("vendor_02", "Market-data redistribution outside licensed desktops is prohibited without vendor amendment."),
    ("bc_dr_01", "Disaster recovery RTO for trade capture is 2 hours; RPO is 5 minutes of durable log loss max."),
    ("bc_dr_02", "Annual market-close DR test includes failover of matching, middle office, and client reporting."),
    ("prod_fund_01", "The Global Credit Income Fund invests primarily in IG corporates with max 20 percent HY sleeve."),
    ("prod_fund_02", "The Ultra-Short Liquidity Fund targets WAM under 60 days and forbids subordinated bank paper."),
    ("prod_loan_01", "Securities-backed lines of credit use 50 percent advance rate on concentrated single-stock collateral."),
    ("prod_loan_02", "Margin loan interest accrues daily using the broker call rate plus 1.25 percent spread."),
]


def _variants(doc_id: str, text: str) -> list[tuple[str, str]]:
    """Generate multiple natural queries that should retrieve this passage."""
    # Topic-aware templates by doc prefix
    prefix = doc_id.split("_")[1] if "_" in doc_id else "gen"
    templates = {
        "kyc": [
            "What is the KYC refresh deadline after ownership changes?",
            "How soon must KYC be refreshed when beneficial ownership changes?",
            "KYC policy for ownership change above 25 percent",
        ],
        "aml": [
            "When must a SAR be filed after detection?",
            "AML alert aging dual control requirement",
            "Suspicious activity reporting timeline",
        ],
        "wire": [
            "When is callback verification required for wires?",
            "International wire approval rules above 50000",
            "Same-day wire release cutoff policy",
        ],
        "margin": [
            "What is the maintenance margin call deadline?",
            "House margin requirement for concentrated equity",
            "When are positions liquidated for unmet margin?",
        ],
        "settle": [
            "How are failed equity settlements retried?",
            "DTC partial settlement allocation rules",
            "Settlement fail escalation to operations",
        ],
        "nav": [
            "When is official fund NAV struck?",
            "NAV error correction approval threshold",
            "Fair value pricing for late prices",
        ],
        "liq": [
            "What happens on LCR breach warning?",
            "Redemption gate trigger for short-duration credit fund",
            "Liquidity contingency playbook timing",
        ],
        "cpty": [
            "Prime broker counterparty exposure limit",
            "ISDA CSA dispute escalation path",
            "Counterparty limit review frequency",
        ],
        "sox": [
            "SOX control for suspense account reconciliation",
            "General ledger privileged access recertification",
            "FR-CTRL-17 dual sign-off requirement",
        ],
        "mkt": [
            "Interest rate DV01 limit for rates desk",
            "VaR exception model risk review threshold",
            "Market risk limit breach notification",
        ],
        "inc": [
            "Runbook for trade capture latency spike",
            "What causes overnight pricing job to delay NAV?",
            "Wire callback void root cause pattern",
            "Settlement fail cascade from short inventory",
            "Matching engine freeze failover steps",
        ],
        "tax": [
            "1099-B cost basis method for mutual funds",
            "Wash sale rule window for identical securities",
        ],
        "acct": [
            "ACH for margin deficit payments",
            "Corporate action election cutoff time",
        ],
        "esg": [
            "Article 8 principal adverse impact disclosure",
            "Green bond unallocated proceeds review period",
        ],
        "credit": [
            "Issuer downgrade below BB- hold to maturity review",
            "Single-name CDS notional approval limit",
        ],
        "ops": [
            "Material reconciliation break threshold",
            "STP exception rate quality review trigger",
            "Cash break aging to ops manager",
            "Options expiration position break clearing",
        ],
        "reg": [
            "Reg BI documentation retention period",
            "MiFID II best execution report cadence",
            "GDPR DSAR fulfillment timeline",
        ],
        "data": [
            "Golden source for LEI identifiers",
            "Authoritative intraday trade blotter topic",
        ],
        "sec": [
            "Break-glass access expiry for payment systems",
            "Market data API key rotation period",
        ],
        "eq": ["Equity order ADV soft block threshold"],
        "fi": ["Corporate bond RFQ stale quote age"],
        "fx": ["NDF fixing source WM/R London"],
        "svc": [
            "Tier-1 institutional ticket response SLA",
            "Complaint acknowledgment deadline",
        ],
        "audit": [
            "Internal audit sample for wire dual control",
            "Model inventory attestation due date",
        ],
        "treasury": [
            "Intraday HQLA liquidity buffer target",
            "Fedwire third-party cutoff time",
        ],
        "fraud": [
            "New payee same-day wire fraud escalation",
            "Voice phishing payment freeze indicators",
        ],
        "comp": [
            "Material risk taker deferral percentage",
            "Personal trading pre-clearance rule",
        ],
        "vendor": [
            "SOC 2 requirement for critical payment vendors",
            "Market data redistribution license rule",
        ],
        "dr": [
            "Trade capture disaster recovery RTO",
            "Annual market-close DR test scope",
        ],
        "fund": [
            "Global Credit Income Fund HY sleeve limit",
            "Ultra-Short Liquidity Fund WAM target",
        ],
        "loan": [
            "SBLOC advance rate on concentrated stock",
            "Margin loan interest spread over broker call",
        ],
    }

    # Map middle token of doc_id (pol_kyc_01 -> kyc)
    parts = doc_id.split("_")
    key = parts[1] if len(parts) >= 2 else "gen"
    # special cases: run_inc -> inc, risk_credit -> credit, risk_ops -> ops, legal_reg -> reg,
    # tech_data -> data, tech_sec -> sec, desk_eq -> eq, desk_fi -> fi, desk_fx -> fx,
    # client_svc -> svc, bc_dr -> dr, prod_fund -> fund, prod_loan -> loan, hr_comp -> comp
    if parts[0] == "run":
        key = "inc"
    elif parts[0] == "risk":
        key = parts[1]
    elif parts[0] == "legal":
        key = "reg"
    elif parts[0] == "tech":
        key = parts[1]
    elif parts[0] == "desk":
        key = parts[1]
    elif parts[0] == "client":
        key = "svc"
    elif parts[0] == "bc":
        key = "dr"
    elif parts[0] == "prod":
        key = parts[1]
    elif parts[0] == "hr":
        key = "comp"
    elif parts[0] == "faq":
        key = parts[1]
    elif parts[0] == "ops":
        key = "ops"

    qs = templates.get(key, [])
    # Always add generic paraphrases tied to distinctive phrases
    words = [w for w in text.replace(".", "").split() if len(w) > 5][:6]
    hint = " ".join(words[:4])
    qs = list(qs) + [
        f"Policy about {hint}",
        f"Explain: {text[:90]}",
        f"Where is guidance on {words[0] if words else 'this control'}?",
    ]
    # Dedup while preserving order
    seen = set()
    out = []
    for q in qs:
        qn = q.strip()
        if qn and qn not in seen:
            seen.add(qn)
            out.append((qn, text))
    return out


EVAL_QUERIES = [
    ("When must KYC be refreshed after a major ownership change?", ["pol_kyc_01"]),
    ("Do PEPs need enhanced due diligence before onboarding?", ["pol_kyc_02"]),
    ("How fast must we file a SAR after detecting suspicious wires?", ["pol_aml_01"]),
    ("What is the aging limit for AML transaction monitoring alerts?", ["pol_aml_02"]),
    ("Are callbacks required for large international wires?", ["pol_wire_01"]),
    ("Can we release a same-day wire after 3 PM Eastern?", ["pol_wire_02"]),
    ("By when must a maintenance margin call be met?", ["pol_margin_01"]),
    ("What house margin applies to concentrated equity holdings?", ["pol_margin_02"]),
    ("How does the firm retry failed equity settlements?", ["pol_settle_01"]),
    ("What is required before end-of-day NAV for DTC partials?", ["pol_settle_02"]),
    ("What time is official fund NAV struck?", ["pol_nav_01"]),
    ("Who approves large NAV error corrections?", ["pol_nav_02"]),
    ("What playbook runs on an LCR breach warning?", ["pol_liq_01"]),
    ("When do redemption gates activate on the short-duration credit fund?", ["pol_liq_02"]),
    ("What is the prime broker exposure limit versus Tier 1 capital?", ["pol_cpty_01"]),
    ("How are large ISDA CSA disputes escalated?", ["pol_cpty_02"]),
    ("Which SOX control covers suspense account reconciliation?", ["pol_sox_01"]),
    ("How often is privileged GL access recertified?", ["pol_sox_02"]),
    ("What is the rates desk DV01 limit?", ["pol_mkt_01"]),
    ("When must Model Risk review VaR exceptions?", ["pol_mkt_02"]),
    ("What should we do if trade capture p95 latency exceeds 2 seconds?", ["run_inc_01"]),
    ("How do we respond to a matching engine freeze?", ["run_inc_02"]),
    ("Why might the overnight pricing job delay NAV?", ["run_inc_03"]),
    ("What root cause delays wires when callbacks void?", ["run_inc_04"]),
    ("How do short inventory issues cascade into settlement fails?", ["run_inc_05"]),
    ("How is mutual fund cost basis reported on 1099-B?", ["faq_tax_01"]),
    ("What is the wash-sale repurchase window?", ["faq_tax_02"]),
    ("Can ACH satisfy a margin deficit?", ["faq_acct_01"]),
    ("When do corporate action elections close?", ["faq_acct_02"]),
    ("What must Article 8 funds disclose annually?", ["faq_esg_01"]),
    ("When are unallocated green bond proceeds reviewed?", ["faq_esg_02"]),
    ("What happens if an issuer is downgraded below BB-?", ["risk_credit_01"]),
    ("What CDS notional needs Credit Committee approval?", ["risk_credit_02"]),
    ("When is a reconciliation break material to the CFO dashboard?", ["risk_ops_01"]),
    ("What STP exception rate triggers an ops quality review?", ["risk_ops_02"]),
    ("How long must Reg BI suitability docs be retained?", ["legal_reg_01"]),
    ("How often are MiFID II best-execution reports published?", ["legal_reg_02"]),
    ("GDPR access request deadline for EU clients?", ["legal_reg_03"]),
    ("What is the golden source for LEIs?", ["tech_data_01"]),
    ("Which Kafka topic is authoritative for intraday risk?", ["tech_data_02"]),
    ("How long does payment break-glass access last?", ["tech_sec_01"]),
    ("How often do market-data API keys rotate?", ["tech_sec_02"]),
    ("Equity order size versus ADV soft block?", ["desk_eq_01"]),
    ("When is a corporate bond RFQ response considered stale?", ["desk_fi_01"]),
    ("Default NDF fix source?", ["desk_fx_01"]),
    ("Cash break aging requiring Ops Manager comment?", ["ops_rec_01"]),
    ("Tier-1 institutional trade query SLA?", ["client_svc_01"]),
    ("Complaint acknowledgment timing?", ["client_svc_02"]),
    ("Internal Audit wire dual-control sample size?", ["audit_01"]),
    ("Model inventory attestation due date?", ["audit_02"]),
    ("Intraday HQLA buffer target?", ["treasury_01"]),
    ("Fedwire cutoff for third-party wires?", ["treasury_02"]),
    ("Fraud escalation for new payee then quick wire?", ["fraud_01"]),
    ("Indicators of voice phishing on payment instructions?", ["fraud_02"]),
    ("Deferral percentage for material risk takers?", ["hr_comp_01"]),
    ("Personal trading pre-clearance for covered employees?", ["hr_comp_02"]),
    ("SOC 2 expectation for critical payment vendors?", ["vendor_01"]),
    ("Trade capture DR RTO and RPO?", ["bc_dr_01"]),
    ("HY sleeve limit in Global Credit Income Fund?", ["prod_fund_01"]),
    ("WAM target for Ultra-Short Liquidity Fund?", ["prod_fund_02"]),
    ("SBLOC advance rate on concentrated single-stock collateral?", ["prod_loan_01"]),
    ("Margin loan interest formula?", ["prod_loan_02"]),
]


CAUSAL = [
    {"earlier": "CRM phone sync job failed overnight for institutional clients.", "later": "Wire callback verification voided and international wires delayed past cutoff."},
    {"earlier": "Overnight batch pricing job overran past 4 PM ET.", "later": "Official fund NAV publication slipped and investor portal showed stale prices."},
    {"earlier": "Borrow desk could not locate shares before 11 AM ET.", "later": "Equity settlement fails cascaded and DTC partials required allocation updates."},
    {"earlier": "Market-data API keys expired without rotation.", "later": "Trader UI showed silent quote stalls and RFQ hit ratios dropped."},
    {"earlier": "Liquidity buffer usage crossed 70 percent of HQLA target.", "later": "Treasury contingency playbook was paged and discretionary lending was paused."},
    {"earlier": "New payee was created and a wire was submitted within 24 hours.", "later": "Fraud Operations auto-escalated and froze the payment instruction."},
    {"earlier": "Trade capture p95 latency exceeded 2 seconds after a release.", "later": "Non-critical enrichments were disabled and STP fail-open mode was engaged."},
    {"earlier": "Matching engine primary froze during a volatility spike.", "later": "Secondary matcher was activated and new order intake was frozen."},
    {"earlier": "AML alert aged beyond 5 business days without dual review.", "later": "Compliance opened an exception and escalated to AML Investigations management."},
    {"earlier": "Maintenance margin call remained unmet after 2 PM ET on T+1.", "later": "Risk liquidated the concentrated equity positions under house rules."},
    {"earlier": "Issuer was downgraded below BB-.", "later": "Credit reviewed hold-to-maturity eligibility and flagged the position for committee."},
    {"earlier": "Weekly outflows on the short-duration credit fund exceeded 15 percent of AUM.", "later": "Redemption gates were activated per liquidity policy."},
]


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    documents = [{"document_id": d, "chunk_text": t} for d, t in DOCS]
    pairs = []
    for doc_id, text in DOCS:
        for q, p in _variants(doc_id, text):
            pairs.append({"query": q, "passage": p, "source_doc_id": doc_id})

    # Hold out eval queries from being exact duplicates of training by slight paraphrase already;
    # still keep pairs abundant.
    eval_rows = [
        {"query": q, "relevant_doc_ids": ids}
        for q, ids in EVAL_QUERIES
    ]

    write_jsonl(OUT / "documents.jsonl", documents)
    write_jsonl(OUT / "dense_pairs.jsonl", pairs)
    write_jsonl(OUT / "eval.jsonl", eval_rows)
    write_jsonl(OUT / "causal.jsonl", CAUSAL)

    print(f"Wrote {len(documents)} documents")
    print(f"Wrote {len(pairs)} dense pairs")
    print(f"Wrote {len(eval_rows)} eval queries")
    print(f"Wrote {len(CAUSAL)} causal pairs")
    print(f"Output directory: {OUT}")


if __name__ == "__main__":
    main()
