"""
Generate an adversarial hard-eval pack that targets dense Miss@10.

Design constraints (see HARD_EVAL_ADVERSARIAL.md):
  - Cluster size >= 18 so Top-10 cannot cover an entire topic
  - Dense negatives reuse query surface keywords / symptom language
  - Gold docs describe root cause / exception / sub-clause WITHOUT the
    query's distinctive symptom tokens when possible
  - Four patterns: causal, hyperbolic, relational, epistemic

Output: demos/finance_demo/hard_adversarial/{documents,dense_pairs,eval,causal}.jsonl

Calibration (required before claiming hardness):
  docker compose run --rm vectorprism python demos/finance_demo/calibrate_adversarial_pack.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parent / "hard_adversarial"


def _doc(doc_id: str, text: str) -> dict:
    return {"document_id": doc_id, "chunk_text": text}


def _cluster(
    prefix: str,
    *,
    pattern: str,
    gold_text: str,
    distractors: list[str],
    fillers: list[str],
    query: str,
    miss_reason: str,
    channel_hint: str,
    causal_pair: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Build one adversarial cluster: 1 gold + >=15 distractors + fillers (>=18 total)."""
    docs = [_doc(f"{prefix}_gold", gold_text)]
    for i, t in enumerate(distractors, 1):
        docs.append(_doc(f"{prefix}_d{i:02d}", t))
    for i, t in enumerate(fillers, 1):
        docs.append(_doc(f"{prefix}_f{i:02d}", t))
    assert len(docs) >= 18, f"{prefix} cluster too small: {len(docs)}"
    return {
        "prefix": prefix,
        "pattern": pattern,
        "docs": docs,
        "eval": {
            "query": query,
            "relevant_doc_ids": [f"{prefix}_gold"],
            "dense_should_miss": True,
            "miss_reason": miss_reason,
            "channel_hint": channel_hint,
            "difficulty": "hard",
            "pattern": pattern,
        },
        "causal": (
            {"earlier": causal_pair[0], "later": causal_pair[1]} if causal_pair else None
        ),
        # Dense training pairs: prefer distractor-safe paraphrases of gold mechanism
        # (not the adversarial eval query — avoid train/eval leakage)
        "train_passages": [gold_text],
    }


def build_clusters() -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []

    # ---- Pattern 1: Multi-hop causal disconnect ----
    clusters.append(
        _cluster(
            "ADV_C01",
            pattern="causal_multihop",
            channel_hint="causal",
            query=(
                "Why did the London trading desk's aggressive order routing halt at 14:15 "
                "even though the exchange FIX sessions still showed connected status?"
            ),
            miss_reason=(
                "multi-hop causal disconnect: query symptom is routing halt / FIX; "
                "gold is NY collateral revaluation → ICL breach → routing gate; "
                "distractors monopolize routing/FIX/SOR vocabulary"
            ),
            gold_text=(
                "Intraday Credit Limit Policy POL-ICL-220: When scheduled sovereign bond "
                "haircut revaluation reduces unencumbered collateral value, the Risk Gatekeeper "
                "recomputes each desk's Intraday Credit Ceiling. If available credit falls below "
                "the desk's open gross exposure, outbound electronic order flow is blocked until "
                "additional eligible collateral is pledged. The blockage is a credit-capacity "
                "control, not a market-connectivity failure. Revaluation cycles run at 12:00 ET; "
                "downstream credit exhaustion may surface on overseas desks later the same day."
            ),
            distractors=[
                "Electronic Trading Incident Runbook: When aggressive order routing halts at 14:15, first verify FIX session status and MsgSeqNum continuity on all lit venues.",
                "Smart Order Router Circuit Breaker SOP: Round-trip latency spikes above 50 ms for three consecutive cycles de-route venues and can halt aggressive order routing at 14:15.",
                "FIX Engine Disconnect Playbook: Connected status LEDs can remain green while application-level reject storms halt London trading desk aggressive order routing.",
                "Market Data Multicast Loss SOP: Dual feed packet loss trips the volatility circuit breaker and cancels peg orders, appearing as an aggressive order routing halt.",
                "Exchange Matching Engine Freeze Guide: Secondary matcher failover at 14:15 may pause aggressive order routing until blotter reconciliation completes.",
                "SOR Venue De-route Policy: NASDAQ or Direct Edge latency incidents commonly halt London desk aggressive order routing during the 14:00–14:30 window.",
                "Drop Copy Recovery Runbook: Risk monitor freezes after drop-copy gaps can look like an aggressive order routing halt even when FIX sessions report connected.",
                "Trading Floor Network SOP: SD-WAN failover events at 14:15 often interrupt aggressive order routing while exchange sessions still show connected.",
                "Algo Kill-Switch Policy: Manual or automated kill-switch activation stops aggressive order routing immediately without tearing down FIX connections.",
                "Throttle Gate Runbook: Exchange-imposed 50 TPS caps can halt London aggressive order routing while session heartbeats remain healthy.",
                "Order Cancel Storm SOP: Cancel-to-fill spikes above 90% may auto-disable aggressive order routing strategies at 14:15.",
                "Venue Status Broadcast Guide: Exchange maintenance notices around 14:15 can force aggressive order routing halts despite connected FIX sockets.",
                "DMA Client Isolation Policy: Isolating a misbehaving DMA session stops aggressive order routing for that desk without a FIX logout.",
                "Clock Sync Incident SOP: PTP drift can desynchronize SOR timers and halt aggressive order routing while venues report connected status.",
                "Pre-Trade Risk Control Guide: Fat-finger limit breaches block aggressive order routing even though exchange FIX sessions remain connected.",
            ],
            fillers=[
                "Equity cash desk ADV soft-block thresholds for large child orders during London afternoon session.",
                "Broker-dealer best execution attestation checklist for multi-venue SOR configurations.",
            ],
            causal_pair=(
                "Scheduled sovereign bond haircut revaluation reduces unencumbered collateral and depletes a trading desk Intraday Credit Ceiling",
                "Risk Gatekeeper blocks outbound electronic order flow on the affected desk until additional collateral is pledged",
            ),
        )
    )

    clusters.append(
        _cluster(
            "ADV_C02",
            pattern="causal_multihop",
            channel_hint="causal",
            query=(
                "Why did prime brokerage place a settlement lock on the client's Friday morning "
                "DVP deliveries when trade economics already matched?"
            ),
            miss_reason=(
                "causal multihop: symptom is Friday DVP settlement lock; gold is Wednesday "
                "variation-margin shortfall cascading to credit freeze; distractors are SSI/DVP/lock text"
            ),
            gold_text=(
                "Uncleared Margin Escalation SOP-COL-240: If a covered counterparty fails to meet "
                "an undisputed variation margin obligation by the contractual transfer deadline, "
                "Credit Operations may impose a temporary financing freeze on that counterparty's "
                "prime brokerage credit line. While the freeze is active, Delivery-Versus-Payment "
                "releases remain blocked even when trade economics affirm. The initiating event is "
                "the unmet margin transfer, which may precede the visible settlement blockage by "
                "one or more business days."
            ),
            distractors=[
                "Prime Brokerage Settlement Lock SOP: Friday morning DVP deliveries enter settlement lock when net unsettled debit exceeds institutional thresholds.",
                "DVP Gateway Policy: Settlement locks on Friday morning DVP deliveries commonly follow SSI mismatches in DTCC ALERT despite matched economics.",
                "TradeSuite Affirmation Runbook: Matched economics with settlement lock on Friday morning DVP usually indicate PSET or clearing-agent drift.",
                "CNS Fail Management Guide: Settlement locks appear on Friday morning DVP deliveries when prior fails age into buy-in windows.",
                "Custodian Instruction SOP: Friday morning DVP settlement locks occur when safekeeping accounts are closed for corporate action processing.",
                "Reg SHO Close-Out Policy: Pre-borrow restrictions can place settlement locks on Friday morning DVP stock deliveries.",
                "Omgeo CTM Exception Guide: Net amount rounding breaks can trigger settlement locks even when Friday morning DVP economics look matched.",
                "Prime Brokerage Credit Override SOP: Settlement locks on Friday morning DVP releases require 130% collateral coverage before unlock.",
                "Partial Settlement Policy: Incomplete allocations create Friday morning DVP settlement locks despite trade-level affirmation.",
                "Stock Record Break SOP: Position breaks between street and firm books surface as Friday morning DVP settlement locks.",
                "Buy-In Notice Playbook: Pending FINRA 11810 notices hold Friday morning DVP deliveries under settlement lock.",
                "Euroclear Bridge Matching Guide: Identifier mismatches create settlement locks on cross-border DVP deliveries arriving Friday morning.",
                "Standing Settlement Instruction Policy: Unauthenticated SSI changes place settlement locks on Friday morning DVP pipelines.",
                "Clearing Cap Exhaustion SOP: Daylight credit caps can freeze Friday morning DVP deliveries with settlement lock status.",
                "Trade Cancel-Rebook Runbook: Mid-cycle cancel/rebook storms leave Friday morning DVP items in settlement lock queues.",
            ],
            fillers=[
                "Monthly DTCC participant billing description for CNS fail penalty invoices.",
                "Prime brokerage client onboarding checklist for sponsored repo eligibility.",
            ],
            causal_pair=(
                "Covered counterparty misses undisputed variation margin transfer deadline",
                "Credit Operations freezes prime brokerage financing and blocks subsequent DVP releases",
            ),
        )
    )

    clusters.append(
        _cluster(
            "ADV_C03",
            pattern="causal_multihop",
            channel_hint="causal",
            query=(
                "Why did Fraud Operations freeze all outbound wires on a wealth account that "
                "submitted a valid wire request with matching MFA?"
            ),
            miss_reason=(
                "causal sequence: query focuses on valid wire+MFA freeze; gold is prior "
                "SIM-swap / phone-change within 24h; distractors are callback/$5M wire rules"
            ),
            gold_text=(
                "Account Takeover Sequence Control SOP-FRD-230: A profile telephone-number change "
                "or carrier SIM replacement followed within twenty-four hours by a funds-transfer "
                "initiation is classified as a high-risk takeover sequence. The fraud engine must "
                "apply a forty-eight-hour outbound payment security freeze. Credential MFA success "
                "does not clear the freeze; only out-of-band biometric video verification by a "
                "fraud specialist may release it."
            ),
            distractors=[
                "Outbound Wire Dual-Auth Policy: Fraud Operations may freeze outbound wires when dual authorization credentials fail on wealth accounts.",
                "Voice Callback SOP: Valid wire requests still freeze outbound wires if out-of-band phone callbacks cannot be completed.",
                "High-Value Wire Policy: Wealth accounts initiating wires above five million dollars face automatic outbound wire freezes pending supervisor review.",
                "MFA Device Binding Guide: Matching MFA that originates from a new device fingerprint can freeze outbound wires on wealth accounts.",
                "Beneficiary Change Policy: Recent payee edits trigger Fraud Operations outbound wire freezes even on otherwise valid wire requests.",
                "Travel Notice SOP: Absence of a travel notice may freeze outbound wires for wealth clients submitting valid international requests.",
                "Velocity Rule Guide: Multiple valid wire requests in one hour cause Fraud Operations to freeze outbound wires on the wealth account.",
                "Sanctions Screen Hold SOP: OFAC indeterminate hits freeze outbound wires despite valid wire request forms and matching MFA.",
                "Private Banking Approval Policy: Missing relationship-manager attestation freezes outbound wires on wealth accounts.",
                "Wire Template Abuse SOP: First-time use of a free-form beneficiary freezes outbound wires awaiting Fraud Operations clearance.",
                "Geo-IP Anomaly Guide: Login country mismatches freeze outbound wires even when MFA matches on a valid wire request.",
                "Daily Aggregate Cap Policy: Crossing the wealth desk daily wire aggregate freezes further outbound wires automatically.",
                "Call-Back Number Mismatch SOP: Callback numbers taken from the wire form (not core records) force outbound wire freezes.",
                "Dormant Account Wake Policy: Long-dormant wealth accounts see outbound wires frozen on the first valid wire after reactivation.",
                "Push-Payment Scam SOP: Suspected APP scam language in remittance freezes outbound wires pending Fraud Operations review.",
            ],
            fillers=[
                "Wealth desk client communication templates for temporary payment freezes.",
                "Card soft-block SMS confirmation workflow for retail debit velocity alerts.",
            ],
            causal_pair=(
                "Customer profile phone number change or SIM-swap occurs within 24 hours before a funds transfer request",
                "Fraud engine applies a 48-hour outbound payment security freeze requiring biometric video release",
            ),
        )
    )

    # ---- Pattern 2: Hyperbolic sibling / level collision ----
    clusters.append(
        _cluster(
            "ADV_H01",
            pattern="hyperbolic_sibling",
            channel_hint="hyperbolic",
            query=(
                "Which operational approval applies specifically to an unlisted holding company "
                "under the Tier-2 discretionary trust onboarding path?"
            ),
            miss_reason=(
                "hyperbolic level collision: distractors are parent Tier-2 trust, Tier-1 trust, "
                "and corporate holding guides; gold is deep sub-clause for unlisted holdcos"
            ),
            gold_text=(
                "Fiduciary Structures Manual Section 4.3.b (Child Node): For an unlisted holding "
                "company nested beneath a Tier-2 discretionary trust, onboarding requires "
                "Protector acknowledgment plus Regional Fiduciary Counsel countersignature before "
                "account activation. This child rule does not inherit the simplified approvals "
                "used for listed holdcos or for Tier-1 absolute trusts. Depth of the structure "
                "matters: parent Tier-2 trust procedures alone are insufficient."
            ),
            distractors=[
                "Tier-2 Discretionary Trust Policy (Parent): Operational approvals for Tier-2 discretionary trust onboarding follow the standard fiduciary committee checklist.",
                "Tier-2 Trust Onboarding Guide: Holding company structures under discretionary trusts use the general Tier-2 discretionary trust onboarding path approvals.",
                "Tier-1 Discretionary Trust Rules: Absolute and discretionary Tier-1 trusts require Central Fiduciary Board approval during onboarding.",
                "Corporate Holding Company Onboarding Guide: Unlisted holding companies opened as corporate clients follow commercial KYC approvals, not trust paths.",
                "Discretionary Trust Parent Manual Section 4: Tier-2 discretionary trust onboarding path covers settlors, trustees, and named beneficiaries at the parent policy level.",
                "Private Foundation Onboarding SOP: Tier-2 fiduciary vehicles use the same operational approvals as discretionary trust onboarding for most holding entities.",
                "Listed Holdco Exception Note: Listed holding companies under discretionary trusts may use simplified Tier-2 onboarding approvals.",
                "Trust Protector Overview: Protectors are documented in Tier-2 discretionary trust onboarding files but approvals remain at parent-policy level.",
                "Offshore Trust Taxonomy: Cayman and BVI Tier-2 discretionary trusts share a common onboarding path for holding company structures.",
                "Fiduciary Risk Tiering Matrix: Tier-2 versus Tier-1 discretionary trust onboarding paths differ mainly by committee level, not entity subtype.",
                "Beneficial Owner Hierarchy Guide: Holding company ownership under trusts is assessed with standard Tier-2 discretionary trust onboarding thresholds.",
                "Family Office Trust Playbook: Unlisted entities in family structures usually follow the Tier-2 discretionary trust onboarding path parent checklist.",
                "Trustee Capacity Policy: Operational approvals for discretionary trust onboarding emphasize trustee fitness rather than holdco listing status.",
                "Simplified Due Diligence Note: Some Tier-2 discretionary trust onboarding paths allow simplified approvals for low-risk holding entities.",
                "Cross-Border Fiduciary SOP: Unlisted holding companies in trust stacks are processed under the generic Tier-2 discretionary trust onboarding path.",
            ],
            fillers=[
                "Annual fiduciary policy attestation calendar for regional counsel.",
                "Glossary of settlor, protector, and absolute beneficiary definitions.",
            ],
        )
    )

    clusters.append(
        _cluster(
            "ADV_H02",
            pattern="hyperbolic_sibling",
            channel_hint="hyperbolic",
            query=(
                "For SFDR product classification, which rule governs a fund that promotes ESG "
                "characteristics but does not have sustainable investment as its core objective?"
            ),
            miss_reason=(
                "taxonomy sibling collision: Article 9 / GAR / exclusions distractors share ESG "
                "vocabulary; gold is Article 8 light-green promotion criteria"
            ),
            gold_text=(
                "Sustainable Product Taxonomy POL-ESG-108 (Article 8 Branch): Products that promote "
                "environmental or social characteristics without adopting sustainable investment as "
                "the principal objective are classified under the Article 8 branch. They must publish "
                "a binding minimum allocation to ESG-aligned assets as stated in the prospectus and "
                "enforce the exclusions list pre-trade. This branch is distinct from Article 9 "
                "dark-green products and from firm-level Green Asset Ratio reporting."
            ),
            distractors=[
                "SFDR Article 9 Dark Green Policy: Funds with sustainable investment as core objective require 100% sustainable allocation except cash hedges.",
                "PAI Violation Divestment SOP: Article 9 portfolios must divest holdings with severe principal adverse impact breaches within 30 business days.",
                "EU Taxonomy Green Asset Ratio Guide: Banking books compute GAR on taxonomy-aligned loans and mortgages annually.",
                "ESG Exclusion List Policy: Weapons, thermal coal, and tobacco exclusions apply across sustainable product onboarding.",
                "CSRD Supply Chain Audit SOP: Borrowers above €150M turnover need tier-1 human rights due diligence for green lending.",
                "Article 9 DNSH Checklist: Do No Significant Harm tests across fourteen PAI indicators for dark-green funds.",
                "Sustainable Prospectus Drafting Guide: Marketing language for ESG characteristics must map to a defined SFDR product class.",
                "Climate Risk Overlay Policy: Portfolio carbon intensity overlays used in ESG promotion and sustainable investment strategies.",
                "Green Bond Proceeds SOP: Earmarked ledgers track taxonomy-aligned use of proceeds for sustainable investment products.",
                "ESG Watchlist Procedure: Red controversy flags exclude issuers from sustainable and ESG-promoting portfolios.",
                "Principal Adverse Impact Reporting Manual: Annual PAI disclosures for funds promoting environmental or social characteristics.",
                "Dark Green Board Oversight Policy: Boards oversee core sustainable investment objectives for Article 9 products.",
                "Light-Touch ESG Marketing Note: Some brochures mention ESG characteristics without specifying the binding SFDR branch.",
                "Taxonomy Technical Screening Guide: Engineers verify environmentally sustainable economic activities for GAR and Article 9.",
                "Cross-Cutting ESG Governance Policy: Enterprise ESG committee oversees both promotion-only and sustainable-objective products.",
            ],
            fillers=[
                "Quarterly Sustainability Committee agenda template.",
                "Vendor ESG ratings data dictionary for controversy flags.",
            ],
        )
    )

    # ---- Pattern 3: Multi-constraint relational boundary ----
    clusters.append(
        _cluster(
            "ADV_R01",
            pattern="relational_boundary",
            channel_hint="relational",
            query=(
                "Why was a seven-million-dollar commercial outbound Fedwire released without "
                "completing an out-of-band voice callback to the client?"
            ),
            miss_reason=(
                "relational boundary: distractors restate >$5M dual-auth+callback; gold is "
                "repetitive SSI exemption within 180 days"
            ),
            gold_text=(
                "Payments Exception Matrix POL-PAY-201-EX: Out-of-band voice callbacks are waived "
                "for repetitive commercial Fedwire instructions that match Standing Settlement "
                "Instructions authenticated within the prior 180 days, even when notional exceeds "
                "five million dollars. Dual authorization credentials remain mandatory. The waiver "
                "applies only to exact SSI matches; free-form beneficiaries never qualify."
            ),
            distractors=[
                "Treasury Payments Policy POL-PAY-201: All outbound Fedwire commercial wires exceeding five million dollars require dual authorization and out-of-band voice callback.",
                "Wire Room Callback SOP: Seven-million-dollar commercial outbound Fedwire releases must complete voice callback to a core-verified signer before release.",
                "High-Value Wire Controls: Commercial outbound Fedwire above five million dollars cannot release without out-of-band voice callback documentation.",
                "Non-Repetitive Wire Guide: Free-form seven-million-dollar Fedwire payments always require out-of-band voice callback under POL-PAY-201.",
                "Dual Authorization Standard: Initiator and approver separation is required for commercial outbound Fedwire at seven million dollars with callback.",
                "Fraud Wire Intercept SOP: Missing out-of-band voice callback on a seven-million-dollar commercial Fedwire is an automatic release block.",
                "Correspondent Payment Policy: International commercial outbound Fedwire over five million dollars needs dual auth and voice callback.",
                "Branch Escalation Note: Large commercial outbound Fedwire items inherit the five-million callback rule used by Treasury Operations.",
                "Standing Settlement Instruction Overview: SSIs store beneficiary details but POL-PAY-201 still cites voice callback for wires over five million.",
                "Payment Supervisor Checklist: Verify dual auth and out-of-band voice callback before releasing any seven-million-dollar commercial Fedwire.",
                "CHIPS High-Value SOP: Parallel CHIPS releases over five million dollars also require out-of-band voice callback controls.",
                "Vendor Payment Playbook: Established vendor seven-million-dollar Fedwire payments still require callback under the standard commercial rule.",
                "Callback Phone Source Rule: Phone numbers for out-of-band voice callback must come from core records, not the wire request form.",
                "Emergency Release Policy: Even urgent seven-million-dollar commercial outbound Fedwire needs documented out-of-band voice callback.",
                "Audit Sample Guide: Internal Audit samples commercial outbound Fedwire over five million for dual auth and callback evidence.",
            ],
            fillers=[
                "Fedwire tag {1510} monitoring duties for the wire room supervisor.",
                "Retail wire dual-approval threshold for consumer transfers above $250,000.",
            ],
        )
    )

    clusters.append(
        _cluster(
            "ADV_R02",
            pattern="relational_boundary",
            channel_hint="relational",
            query=(
                "Why did Sanctions Operations return an incoming payment tied to a comprehensively "
                "sanctioned geography instead of placing the funds into a blocked interest-bearing account?"
            ),
            miss_reason=(
                "relational distinction Block vs Reject: distractors push SDN freeze/block account; "
                "gold is territory nexus without SDN property interest → Reject"
            ),
            gold_text=(
                "Sanctions Disposition Matrix SOP-SAN-002: When an inbound payment references a "
                "comprehensively sanctioned geography but screening confirms no SDN party and no "
                "SDN property interest, Operations must Reject and return the payment across the "
                "clearing chain. Asset freezing into an interest-bearing blocked account is reserved "
                "for SDN property interests. Geography alone does not authorize a Block disposition."
            ),
            distractors=[
                "OFAC Block Procedure: SDN-listed parties on inbound payments require immediate freeze into an interest-bearing blocked account.",
                "Sanctions Operations SOP: Incoming payments involving sanctioned names are placed into blocked interest-bearing accounts within 48 business hours.",
                "SDN Screening Runbook: Positive SDN matches on incoming payments mandate Block disposition and OFAC reporting.",
                "Blocked Property Accounting Guide: Funds under OFAC Block must sit in interest-bearing blocked accounts and be reported on Form TD F 90-22.50.",
                "Comprehensive Sanctions Overview: Cuba, Iran, and North Korea exposures often lead Sanctions Operations to freeze related incoming payments.",
                "Wire Sanctions Hold SOP: Incoming payments with sanctions hits are held and frequently moved to blocked interest-bearing accounts.",
                "OFAC Reporting Timeline: Blocked funds from incoming payments must be reported to OFAC within 10 business days.",
                "Secondary Sanctions Guide: High-risk corridors may still result in Block treatment for incoming payments pending Enhanced Due Diligence.",
                "Sanctions Legal Counsel Note: Prefer Block over return when uncertainty exists on SDN property interest for incoming payments.",
                "Clearing Chain Freeze Playbook: Correspondent banks expect blocked interest-bearing escrow for sanctioned-geography incoming payments.",
                "MT103 Sanctions Filter SOP: Real-time filters instruct Block for many sanctioned-geography incoming payments by default.",
                "Rejected vs Blocked FAQ (Generic): Many training decks incorrectly treat all sanctioned-geography incoming payments as Block events.",
                "Interest-Bearing Escrow Policy: Operations maintains blocked accounts specifically for sanctions freezes on incoming payments.",
                "OFAC License Unblock SOP: Release from blocked interest-bearing accounts requires authenticated specific licenses.",
                "Sanctions QA Checklist: Auditors sample whether incoming payments with sanctions flags were placed into blocked accounts.",
            ],
            fillers=[
                "SSI list refresh procedure for sectoral sanctions identification filters.",
                "Wolfsberg correspondent questionnaire filing calendar.",
            ],
        )
    )

    # ---- Pattern 4: Epistemic / version trap ----
    clusters.append(
        _cluster(
            "ADV_E01",
            pattern="epistemic_version",
            channel_hint="dense",
            query=(
                "What is the currently effective Initial Margin segregation requirement for "
                "Uncleared Margin Rules Phase 6 covered counterparties?"
            ),
            miss_reason=(
                "epistemic version trap: distractors are 2021 drafts / omnibus summaries with "
                "IM/Phase 6 keywords; gold is active third-party ACA segregation rule"
            ),
            gold_text=(
                "Active Margin Segregation Standard POL-UMR-6.4 (Effective): For Phase 6 in-scope "
                "counterparties, two-way Initial Margin must be segregated at an unaffiliated "
                "third-party custodian under an Account Control Agreement. Omnibus house accounts "
                "and legacy 2021 consultation drafts are not authoritative for production "
                "settlement. This active standard supersedes prior transitional guidance."
            ),
            distractors=[
                "2021 UMR Consultation Draft: Phase 6 Initial Margin may be held in firm omnibus accounts during the transition period discussed in the consultation.",
                "Legacy IM Overview (2021): Initial Margin segregation for Phase 6 covered counterparties was described with flexible custodial options in early drafts.",
                "Omnibus Clearing Summary: Many desks still summarize Phase 6 Initial Margin as omnibus-eligible in internal 2021 training slides.",
                "Transitional UMR FAQ: Phase 6 covered counterparties were told Initial Margin segregation requirements would be finalized later.",
                "House Account Margin Note: Historical guidance allowed Initial Margin for Phase 6 names to remain in house pools pending custodian onboarding.",
                "UMR Phase 6 Kickoff Deck (2021): Lists Initial Margin segregation as 'TBD custodian' for covered counterparties.",
                "Draft CSA Annex Commentary: Discusses Phase 6 Initial Margin without mandating unaffiliated third-party segregation.",
                "Old Playbook Excerpt: Initial Margin segregation requirement for Phase 6 covered counterparties references omnibus as acceptable.",
                "Industry Working Group Minutes 2021: Debated whether Phase 6 Initial Margin needed third-party segregation in all cases.",
                "Archived Compliance Bulletin: Temporary relief language on Initial Margin segregation for Phase 6 onboarding delays.",
                "Vendor Whitepaper 2021: Describes Phase 6 Initial Margin segregation options including non-segregated models.",
                "Internal Wiki Stub: Phase 6 Initial Margin segregation requirement copied from consultation text, not the active standard.",
                "Training Quiz Answer Key (2021): Marks omnibus as acceptable for Phase 6 Initial Margin segregation.",
                "Retired SOP Header: Initial Margin segregation for Phase 6 covered counterparties — superseded content retained for audit history.",
                "Cross-Reference Index: Points readers to multiple 2021 documents mentioning Phase 6 Initial Margin segregation requirements.",
            ],
            fillers=[
                "Custodian connectivity test script for ACA account control messaging.",
                "AANA observation month calendar reminder for March–May window.",
            ],
        )
    )

    clusters.append(
        _cluster(
            "ADV_E02",
            pattern="epistemic_version",
            channel_hint="dense",
            query=(
                "What NAV error threshold currently requires full retrospective shareholder "
                "transaction reprocessing and direct investor compensation?"
            ),
            miss_reason=(
                "threshold/version collision: distractors emphasize $0.01, 0.10–0.49%, and "
                "immaterial bands; gold is active >=0.50% full reprocessing rule"
            ),
            gold_text=(
                "Active Fund Accounting Error Standard POL-FA-620: When a published NAV error is "
                "equal to or greater than fifty basis points of the correct NAV, the fund must "
                "perform full retrospective transaction reprocessing and compensate affected "
                "transacting shareholders directly. Lower bands that only restate fund-level "
                "books without investor-level reprocessing do not satisfy this active threshold."
            ),
            distractors=[
                "NAV Immaterial Band Note: Errors under $0.01 per share are logged only and need no shareholder reprocessing.",
                "Mid-Band NAV Policy: Errors between 0.10% and 0.49% restate fund accounting without individual shareholder reprocessing.",
                "Legacy NAV Memo: Some older memos treat 0.25% NAV errors as the point for direct investor compensation.",
                "Board Reporting Guide: Quarterly boards review all NAV errors including those below full retrospective reprocessing thresholds.",
                "Fund-Level Compensation SOP: Mid-tier NAV errors compensate the fund entity rather than performing shareholder reprocessing.",
                "Pricing Incident FAQ: Discusses NAV error thresholds and often highlights the $0.01 per share immaterial cut.",
                "Historical Restatement Playbook: Emphasizes accounting restatement workflows used when full shareholder reprocessing is not required.",
                "Transfer Agent Job Aid: Lists NAV error cases that avoid direct investor compensation.",
                "Draft Threshold Table: Shows alternative NAV error percentages under consideration before the active standard.",
                "Auditor Sample Guide: Samples NAV error logs focusing on sub-50bp cases that skip shareholder reprocessing.",
                "Shareholder Letter Templates: Used when voluntary courtesy payments occur below the full retrospective threshold.",
                "Fair Value Overlay Note: Valuation debates that create small NAV errors without triggering full reprocessing.",
                "Swing Pricing Interaction Memo: Explains NAV movements that are not NAV errors and need no shareholder reprocessing.",
                "Legacy 0.50% Debate Deck: Argues against full retrospective shareholder transaction reprocessing at fifty basis points.",
                "Ops Dashboard Caption: Flags NAV error % without stating which band requires direct investor compensation.",
            ],
            fillers=[
                "Money market shadow NAV board notification workflow under Rule 2a-7.",
                "ETF creation cash-in-lieu variance hold procedure at 15 bps.",
            ],
        )
    )

    # Extra causal clusters to grow N
    clusters.append(
        _cluster(
            "ADV_C04",
            pattern="causal_multihop",
            channel_hint="causal",
            query=(
                "Why did the Payment Queue Manager pause multiple large customer Fedwire releases "
                "at mid-morning even though customer ledger balances were sufficient?"
            ),
            miss_reason=(
                "causal: symptom is paused Fedwire with good customer balances; gold is bank "
                "daylight overdraft cap >90%; distractors are fraud/dual-auth/customer-funds text"
            ),
            gold_text=(
                "Payment System Risk Cap SOP-CLR-920: Customer-available balances do not authorize "
                "release when the bank's consolidated daylight overdraft utilization exceeds ninety "
                "percent of the uncollateralized net debit cap. The Payment Queue Manager must pause "
                "outbound Fedwire transfers above ten million dollars until Treasury pledges "
                "additional eligible collateral via the Fedwire Securities Service."
            ),
            distractors=[
                "Customer Funds Control SOP: Large customer Fedwire releases pause when ledger balances are insufficient at mid-morning.",
                "Fraud Hold Policy: Payment Queue Manager pauses large customer Fedwire releases when fraud scores breach thresholds.",
                "Dual Auth Wire SOP: Mid-morning pauses on large customer Fedwire releases often mean dual authorization is incomplete.",
                "Callback Pending Guide: Customer Fedwire releases pause until out-of-band callbacks finish, even with sufficient ledger balances.",
                "Sanctions Review Hold: Large customer Fedwire releases pause in mid-morning queues during sanctions analyst review.",
                "Beneficiary Repair Queue: Syntax repair on pacs.008 can pause large customer Fedwire releases despite good balances.",
                "Cut-Off Proximity SOP: Approaching Fedwire cut-off causes Payment Queue Manager to pause large customer releases.",
                "Liquidity Desk Memo (Generic): Discusses pausing Fedwire releases when customers lack cleared funds.",
                "ACH NSF Parallel Note: Operators sometimes confuse ACH NSF logic with mid-morning Fedwire release pauses.",
                "Client Credit Line SOP: Customer-level credit line exhaustion pauses large Fedwire releases at mid-morning.",
                "Manual Supervisor Queue: Large customer Fedwire items pause for supervisor eyesheet even when balances suffice.",
                "OFAC Indeterminate Hold: Mid-morning Fedwire pauses attributed to sanctions indeterminates on customer payments.",
                "Duplicate Payment Detector: Suspected duplicate large customer Fedwire releases are paused by Payment Queue Manager.",
                "Standing Instruction Mismatch: SSI problems pause large customer Fedwire releases regardless of ledger balances.",
                "Weekend Funding Gap FAQ: Explains mid-morning Fedwire pauses after holiday funding shortfalls in customer accounts.",
            ],
            fillers=[
                "CHIPS pre-funding replenishment cutoff reminder at 16:30 EST.",
                "ON RRP bid window times for the money market desk.",
            ],
            causal_pair=(
                "Bank consolidated daylight overdraft utilization exceeds 90% of uncollateralized net debit cap",
                "Payment Queue Manager pauses outbound Fedwire transfers above $10M until Treasury pledges collateral",
            ),
        )
    )

    clusters.append(
        _cluster(
            "ADV_C05",
            pattern="causal_multihop",
            channel_hint="causal",
            query=(
                "Why did Datacenter Alpha suddenly reject new settlement writes and flip its "
                "database to read-only during a network event?"
            ),
            miss_reason=(
                "causal split-brain: distractors are generic DR/RTO failover; gold is loss of "
                "quorum with Witness C forcing read-only within 500ms"
            ),
            gold_text=(
                "Active-Active Quorum Control POL-DR-702: If Datacenter Alpha loses authenticated "
                "quorum contact with both the peer datacenter and the independent cloud Witness, "
                "it must enter read-only mode within 500 milliseconds and reject settlement writes. "
                "This prevents split-brain ledger divergence. The trigger is quorum loss, not a "
                "planned failover to meet RTO targets."
            ),
            distractors=[
                "Disaster Recovery Failover SOP: Datacenter Alpha rejects writes when a planned failover switches primary roles to Datacenter Beta.",
                "RTO Playbook: Tier-1 settlement systems flip to secondary within 30 minutes RTO during a network event.",
                "Backup Restore Guide: Database read-only mode is used while restoring snapshots after a network event at Datacenter Alpha.",
                "Async Log Shipping SOP: Replay gaps cause administrators to pause writes on Datacenter Alpha during network events.",
                "Maintenance Window Policy: Datacenter Alpha may reject new settlement writes during scheduled network maintenance.",
                "Load Balancer Drain SOP: Connection drains during a network event make Datacenter Alpha appear to reject settlement writes.",
                "Storage Array Failover Note: SAN path loss can force database read-only behavior on Datacenter Alpha.",
                "Kubernetes Disruption Guide: Node pressure during a network event can flip services to read-only configurations.",
                "DNS Failover Runbook: Clients see rejected settlement writes when DNS moves traffic off Datacenter Alpha.",
                "Classic Active-Passive DR: Secondary promotion causes the former primary Datacenter Alpha to reject writes.",
                "Network Partition FAQ: Generic advice to stop writes during partitions without describing quorum Witness rules.",
                "RPO Zero Explainer: Synchronous replication marketing text about never losing settlement data in a network event.",
                "Incident Commander Checklist: Includes switching Datacenter Alpha to read-only as a manual safety step.",
                "Chaos Test Script: Intentionally rejects settlement writes on Datacenter Alpha to validate client retry logic.",
                "Certificate Expiry SOP: mTLS failures during a network event can block settlement writes into Datacenter Alpha.",
            ],
            fillers=[
                "Semi-annual DR drill evidence package requirements for examiners.",
                "Emergency Delegation of Authority matrix for unreachable executives.",
            ],
            causal_pair=(
                "Datacenter Alpha loses quorum contact with peer site and cloud Witness during WAN partition",
                "Local database engine enters read-only within 500ms and rejects settlement write transactions",
            ),
        )
    )

    clusters.append(
        _cluster(
            "ADV_C06",
            pattern="causal_multihop",
            channel_hint="causal",
            query=(
                "Why did LCH declare a clearing-member default after a morning volatility spike "
                "even though overnight margin balances looked sufficient?"
            ),
            miss_reason=(
                "causal timing: distractors are overnight VM/IM schedules; gold is 60-minute "
                "intraday CCP pledge window miss"
            ),
            gold_text=(
                "Intraday CCP Demand SOP-COL-311: Central counterparties may issue intraday variation "
                "margin calls during volatility. Clearing Operations must pledge eligible cash via "
                "Fedwire within sixty minutes of electronic notification. Prior-day end-of-day balances "
                "do not satisfy an unmet intraday call; missing the window is a default event trigger."
            ),
            distractors=[
                "Overnight Margin Schedule: LCH clearing-member default reviews often cite insufficient overnight margin balances after volatility spikes.",
                "EOD VM Call SOP: Morning volatility spikes lead to default declarations when end-of-day variation margin remains unpaid.",
                "IM Shortfall Playbook: Clearing-member default after volatility is commonly attributed to initial margin deficits showing on overnight statements.",
                "Default Management Auction Guide: LCH default processes begin when margin balances look insufficient following a volatility spike.",
                "House Excess Policy: Overnight margin balances that looked sufficient can still be re-aged into default if haircuts jump.",
                "Client Asset Transfer SOP: Volatility spikes cause default declarations when client margin cannot be transferred overnight.",
                "Liquidity Buffer FAQ: Explains clearing-member default risk when overnight margin balances are misestimated after volatility.",
                "Variation Margin Dispute Note: Disputed overnight VM after a volatility spike can escalate toward clearing-member default.",
                "Collateral Schedule Guide: Eligible bond lists for overnight margin that looked sufficient before morning volatility.",
                "CCP Stress Add-on Memo: Morning volatility spikes increase requirements; default follows if overnight postings lag.",
                "Auto-Debit Fail SOP: Failed overnight cash sweeps create apparent clearing-member default after volatility events.",
                "Member Capital Policy: Capital add-ons after volatility can make overnight margin balances look sufficient yet still fail tests.",
                "Intraday Monitoring Dashboard Caption: Shows volatility spikes next to default flags without stating the sixty-minute rule.",
                "Legacy Default Checklist: Emphasizes overnight margin balances rather than intraday pledge timers.",
                "Operations Handover Note: Night desk leaves morning volatility default triage focused on overnight margin screens.",
            ],
            fillers=[
                "GMRA repo MTA description for bilateral margin calls.",
                "SIMM backtesting green-zone exception counting overview.",
            ],
            causal_pair=(
                "CCP issues electronic intraday variation margin call during market volatility",
                "Clearing member misses 60-minute Fedwire pledge window and is declared in default",
            ),
        )
    )

    clusters.append(
        _cluster(
            "ADV_C07",
            pattern="causal_multihop",
            channel_hint="causal",
            query=(
                "Why were non-critical audit consumers disconnected from the Kafka cluster during "
                "yesterday's market-close processing window?"
            ),
            miss_reason=(
                "causal load-shedding: distractors are security/audit topics; gold is consumer lag "
                ">500k on tx-ledger-events requiring shed of non-critical consumers"
            ),
            gold_text=(
                "Ledger Pipeline Backpressure SOP-INC-318: When the primary consumer group on topic "
                "tx-ledger-events exceeds five hundred thousand uncommitted messages, SRE must scale "
                "consumers and detach non-critical downstream audit consumers so the core balance "
                "updater retains throughput through the market-close window."
            ),
            distractors=[
                "Kafka Security Audit SOP: Non-critical audit consumers are disconnected when ACL reviews fail during market-close processing.",
                "Compliance Tap Policy: Audit consumers are taken offline during market-close windows for evidence packaging.",
                "Cluster Upgrade Runbook: Kafka rolling upgrades disconnect non-critical audit consumers around market close.",
                "PII Redaction Job Guide: Audit consumers pause at market close while redaction jobs run.",
                "Topic Compaction Maintenance: Compaction storms disconnect audit consumers during market-close processing.",
                "Quota Enforcement Note: Broker quotas shed audit consumers first in the market-close processing window.",
                "Schema Registry Freeze: Incompatible schemas disconnect audit consumers yesterday near market close.",
                "Network ACL Change SOP: Firewall changes disconnected non-critical audit consumers from Kafka at close.",
                "Cost Control Memo: Finance asked SRE to disconnect non-critical audit consumers overnight including market close.",
                "MirrorMaker Lag FAQ: Cross-region mirror lag blamed for audit consumer disconnects at market close.",
                "Consumer Group Rebalance Guide: Frequent rebalances drop audit consumers during market-close bursts.",
                "Dead Letter Review Policy: Audit teams disconnect consumers while triaging DLQ overload at close.",
                "Kerberos Ticket Expiry SOP: Expired keytabs disconnect audit consumers during market-close processing.",
                "Observability Sampling Note: To reduce cardinality, audit consumers are paused at market close.",
                "Manual Incident Toggle: Incident commanders sometimes disconnect audit consumers as a blunt market-close mitigation.",
            ],
            fillers=[
                "RTP gateway timeout triage thresholds at 5% over 10 minutes.",
                "CAT/MiFIR DLQ overflow emergency patch steps before 08:00 EST.",
            ],
            causal_pair=(
                "Kafka consumer lag on tx-ledger-events exceeds 500,000 uncommitted messages",
                "SRE detaches non-critical audit consumers to protect core ledger balance updater",
            ),
        )
    )

    clusters.append(
        _cluster(
            "ADV_R03",
            pattern="relational_boundary",
            channel_hint="relational",
            query=(
                "Why was a repo margin call notice rejected for same-day settlement when delivered "
                "at 10:45 AM London time?"
            ),
            miss_reason=(
                "relational timing boundary: distractors cite 10:00 AM EST CSA rules; gold is "
                "GMRA after 10:00 AM London → next-day delivery"
            ),
            gold_text=(
                "GMRA Timing Matrix SOP-COL-116: Under the 2011 GMRA Annex I, a margin call served "
                "after 10:00 AM London time rolls to next-business-day collateral delivery. Same-day "
                "settlement applies only to notices served before that London cutoff. US 10:00 AM EST "
                "derivative CSA clocks do not govern GMRA repo margin timing."
            ),
            distractors=[
                "ISDA CSA VM SOP: Margin call notices must be delivered by 10:00 AM EST for same-day collateral settlement.",
                "Derivative Margin Timing Guide: Calls after 10:00 AM EST often miss same-day settlement windows.",
                "US Desk Checklist: Repo-like margin language incorrectly applies the 10:00 AM EST CSA clock to London notices.",
                "Collateral Dispute SOP: Same-day settlement disputes reference EST cutoffs rather than London GMRA clocks.",
                "Tri-Party Margin FAQ: Mentions morning cutoffs and same-day settlement without distinguishing London time.",
                "Variation Margin Playbook: 10:45 AM notices are discussed under EST frameworks for same-day settlement.",
                "Ops Calendar Note: 10:45 AM London delivery attempts are compared against New York CSA schedules.",
                "Eligible Collateral Guide: Lists cash and sovereigns for same-day settlement margin calls generally.",
                "Minimum Transfer Amount SOP: MTA checks occur before same-day settlement of margin calls.",
                "Cross-Timezone Handover Memo: Blames 10:45 AM London notices on EST cutoff confusion for same-day settlement.",
                "Legacy Repo CSA Hybrid Deck: Mixes ISDA and repo clocks; implies 10:00 AM EST controls same-day settlement.",
                "Margin Call Template: Timestamp field examples use EST and talk about same-day settlement.",
                "London Branch FAQ: Says morning margin calls should settle same day without stating the 10:00 AM London rule.",
                "Settlement Supervisor Tip: Reject same-day settlement requests that arrive late morning without citing GMRA.",
                "Audit Finding Draft: Criticizes late morning margin calls missing same-day settlement under 'US timing'.",
            ],
            fillers=[
                "CCP intraday 60-minute pledge reminder.",
                "UMR Phase 6 ACA segregation one-liner.",
            ],
        )
    )

    clusters.append(
        _cluster(
            "ADV_R04",
            pattern="relational_boundary",
            channel_hint="relational",
            query=(
                "Why did screening allow a secondary-market trade in short-tenor notes of an SSI-listed "
                "name to settle without blocking funds?"
            ),
            miss_reason=(
                "relational SSI exception: distractors push full SDN/block; gold is grandfathered "
                "secondary trading / tenor rules under sectoral directives"
            ),
            gold_text=(
                "Sectoral Sanctions Trading Matrix SOP-SAN-015: SSI-listed names are subject to new "
                "debt tenor and equity restrictions, not automatic full asset blocking. Secondary-market "
                "trades in grandfathered instruments may settle without a Block when they do not extend "
                "prohibited credit. Operators must not treat every SSI hit like an SDN freeze."
            ),
            distractors=[
                "SDN Block SOP: Screening hits on sanctioned names require blocking funds before settlement.",
                "Sanctions Freeze Guide: Secondary-market trades in sanctioned names should not settle without blocking funds.",
                "OFAC Payment Filter: SSI-looking hits are often handled with block-and-report playbooks by default.",
                "Trade Surveillance Note: Short-tenor notes of sanctioned names trigger settlement stops and fund blocks.",
                "Blocked Property Policy: Advises blocking funds whenever screening flags a listed name on a trade.",
                "Reject vs Block Training: Many decks still push block for any listed-name secondary-market trade.",
                "Correspondent Caution Memo: Prefer blocking funds on SSI-listed short-tenor note settlements.",
                "Equity Sanctions SOP: New equity in listed names is prohibited; teams over-apply block to debt secondaries.",
                "Directive Confusion FAQ: Mixes Directive 1/2 tenors with SDN blocking language for short-tenor notes.",
                "Settlement Ops Checklist: 'Listed name → block funds' heuristic applied to SSI secondary trades.",
                "License Required Banner: Assumes OFAC licenses before any listed-name secondary-market settlement.",
                "Frozen Inventory Guide: Talks about blocking funds on inventory of sanctioned issuers generally.",
                "Alert Disposition SOP: Default disposition for SSI alerts set to Block in some regional books.",
                "Short-Tenor Debt Watch: Emphasizes stopping settlement of short-tenor notes without explaining grandfathering.",
                "Compliance QA Sample: Scores reviewers on whether they blocked funds after SSI name hits.",
            ],
            fillers=[
                "EO 14114 dual-use monitoring overview for FFIs.",
                "Specific license four-eye unblock steps.",
            ],
        )
    )

    clusters.append(
        _cluster(
            "ADV_H03",
            pattern="hyperbolic_sibling",
            channel_hint="hyperbolic",
            query=(
                "Which control tier applies to a departmental trade checklist spreadsheet used only "
                "for transient operational tracking with no general-ledger posting?"
            ),
            miss_reason=(
                "hyperbolic level: distractors are Category 1 EUC / Tier-1 model rules; gold is "
                "Category 3 low-risk spreadsheet exemption"
            ),
            gold_text=(
                "EUC Taxonomy Child Node POL-SOX-118 (Category 3): Spreadsheets used solely for "
                "non-accounting operational checklists or transient formatting are Category 3 Low Risk. "
                "They are exempt from cell-level locking and formal Model Risk validation required of "
                "Category 1 financial-reporting EUCs and Tier-1 quantitative models. Backup on a secured "
                "share remains required."
            ),
            distractors=[
                "Category 1 High-Risk EUC Policy: Financial reporting spreadsheets need cell locking, access control, and change logs.",
                "Tier-1 Model Validation Standard: Quantitative loss and curve models require independent Model Risk validation before production.",
                "SOX Spreadsheet Control SOP: Trade-related spreadsheets are often classified as Category 1 High-Risk EUCs by default.",
                "EUC Parent Manual: Discusses departmental spreadsheets under the same hardening expectations as reporting workbooks.",
                "Model Inventory Guide: Encourages registering operational spreadsheets alongside Tier-1 models.",
                "Change Tracking Mandate: Broad language requiring change tracking on any spreadsheet touching trading operations.",
                "Formula Lock Standard: Cell-level locking described as mandatory for operations checklists in some training.",
                "Internal Audit EUC Program: Samples departmental trade checklists as if they were Category 1 High-Risk EUCs.",
                "GL Interface Warning: Any spreadsheet near finance processes treated as needing Category 1 controls.",
                "Macro Password Policy: Password-protected macros emphasized for all operations spreadsheets.",
                "Access Recertification Note: Quarterly access reviews applied uniformly across EUC tiers in slideware.",
                "End-User Computing Overview: Collapses Category 1 and Category 3 distinctions for departmental tools.",
                "Trading Ops Toolkit Memo: Lists checklist spreadsheets next to capital models without tier labels.",
                "Control Deficiency Examples: Cite unlocked checklist sheets as SOX issues incorrectly.",
                "Enterprise Spreadsheet Standard: One-size hardening checklist that ignores low-risk transient tracking use.",
            ],
            fillers=[
                "Break-glass production access 24-hour ITSO review rule.",
                "Manual journal entry dual-approval thresholds overview.",
            ],
        )
    )

    clusters.append(
        _cluster(
            "ADV_E03",
            pattern="epistemic_version",
            channel_hint="dense",
            query=(
                "Under the active US T+1 equity settlement regime, what fail penalty rate does DTCC "
                "CNS assess on the net short clearing member?"
            ),
            miss_reason=(
                "jurisdiction/version trap: distractors are CSDR European bps rates; gold is DTCC "
                "CNS 100 bps annualized fail fee"
            ),
            gold_text=(
                "Active DTCC CNS Fail Fee Standard POL-SET-402: Under the US T+1 regime, Continuous "
                "Net Settlement fails are assessed an annualized one-hundred basis point penalty on "
                "the market value of the failed position against the net short clearing member. "
                "European CSDR per-day basis-point schedules are not the active US CNS fee."
            ),
            distractors=[
                "CSDR Settlement Discipline: Liquid European equities incur 1.0 basis point per day cash penalties on fails.",
                "CSDR Illiquid Share Schedule: 0.5 bps per day fail penalties for illiquid shares in EU CSDs.",
                "CSDR Government Bond Penalties: 0.1 bps per day for government bond settlement fails.",
                "CSDR Corporate Bond Penalties: 0.2 bps per day for corporate and SME bond fails.",
                "Euroclear Penalty FAQ: Explains redistribution of CSDR cash penalties to receiving participants.",
                "Clearstream Discipline Note: Monthly netting of European settlement fail penalties.",
                "Legacy US Fail Memo: Older US memos cite different fail fee constructions before T+1 CNS updates.",
                "Buy-In Cost Guide: Discusses open-market buy-in losses often confused with CNS fail fees.",
                "Reg SHO Close-Out SOP: Mandatory close-out timelines mistaken for DTCC CNS penalty rates.",
                "FINRA 11810 Playbook: Buy-in notice timing conflated with CNS fail penalty calculations.",
                "Global Settlement Comparison Deck: Puts CSDR bps tables beside US fails without labeling active CNS 100 bps.",
                "Ops Quiz Answer Key: Marks 1.0 bps/day as the fail penalty without jurisdiction.",
                "Vendor Rate Card: Lists European penalty rates under a 'fail fees' heading used by US ops.",
                "Training Excerpt: 'basis points fail penalty' language drawn from CSDR for US equity examples.",
                "Dashboard Widget: Shows fail penalty bps without stating DTCC CNS annualized 100 bps.",
            ],
            fillers=[
                "CTM affirmation cutoff 21:00 EST under T+1.",
                "ALERT SSI cancel-rebook cutoff 16:00 EST.",
            ],
        )
    )

    return clusters


def build_dense_pairs(clusters: list[dict[str, Any]]) -> list[dict]:
    """Train pairs from gold mechanism text — not adversarial eval queries."""
    pairs = []
    templates = [
        "What policy mechanism governs this control outcome?",
        "Which standard describes the root operational rule for this case?",
        "Summarize the authoritative control requirement in this domain.",
        "What condition changes the default operational treatment here?",
    ]
    for c in clusters:
        gold_id = f"{c['prefix']}_gold"
        gold_text = c["docs"][0]["chunk_text"]
        for t in templates:
            pairs.append({"query": t, "passage": gold_text, "source_doc_id": gold_id})
        # Also index a few distractor paraphrases so dense learns the neighborhood
        for d in c["docs"][1:6]:
            pairs.append(
                {
                    "query": "Which related operational procedure is in this topic neighborhood?",
                    "passage": d["chunk_text"],
                    "source_doc_id": d["document_id"],
                }
            )
    return pairs


def main() -> None:
    clusters = build_clusters()
    OUT.mkdir(parents=True, exist_ok=True)

    docs: list[dict] = []
    eval_rows: list[dict] = []
    causal_rows: list[dict] = []
    for c in clusters:
        docs.extend(c["docs"])
        eval_rows.append(c["eval"])
        if c["causal"]:
            causal_rows.append(c["causal"])

    pairs = build_dense_pairs(clusters)

    for name, rows in [
        ("documents.jsonl", docs),
        ("dense_pairs.jsonl", pairs),
        ("eval.jsonl", eval_rows),
        ("causal.jsonl", causal_rows),
    ]:
        path = OUT / name
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {path} n={len(rows)}")

    meta = {
        "n_clusters": len(clusters),
        "n_docs": len(docs),
        "n_eval": len(eval_rows),
        "n_causal": len(causal_rows),
        "n_dense_pairs": len(pairs),
        "min_cluster_size": min(len(c["docs"]) for c in clusters),
        "patterns": sorted({c["pattern"] for c in clusters}),
        "note": "Run calibrate_adversarial_pack.py before trusting Miss@10 labels",
    }
    (OUT / "pack_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
