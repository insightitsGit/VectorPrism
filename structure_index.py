"""
structure_index.py - Taxonomy + relational attribute indexes for Stage-1/2.

Complements CausalDocGraph:
  - TaxonomyGraph: parent/child/sibling expansion (hyperbolic / is-a)
  - RelationalAttrIndex: predicate match from query heuristics → doc_ids
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _tokens(text: str) -> Set[str]:
    return {t for t in re.findall(r"[a-z0-9$%]{3,}", text.lower())}


class TaxonomyGraph:
    """Undirected lineage over parent/child doc ids for Stage-1 expansion."""

    def __init__(self) -> None:
        self.children: Dict[str, Set[str]] = defaultdict(set)
        self.parents: Dict[str, Set[str]] = defaultdict(set)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TaxonomyGraph":
        g = cls()
        path = Path(path)
        if not path.exists():
            return g
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            p = r.get("parent_doc_id") or r.get("parent")
            c = r.get("child_doc_id") or r.get("child")
            # Only accept doc-id style keys (ADV_*)
            if p and c and str(p).startswith("ADV_") and str(c).startswith("ADV_"):
                g.add_edge(str(p), str(c))
        return g

    def add_edge(self, parent_doc_id: str, child_doc_id: str) -> None:
        self.children[parent_doc_id].add(child_doc_id)
        self.parents[child_doc_id].add(parent_doc_id)

    def get_lineage(self, seed_ids: Iterable[str], depth_delta: int = 2) -> Dict[str, int]:
        """BFS over undirected parent/child edges; returns {doc_id: hops}."""
        dist: Dict[str, int] = {}
        q: deque[Tuple[str, int]] = deque()
        for s in seed_ids:
            s = str(s)
            dist[s] = 0
            q.append((s, 0))
        while q:
            node, d = q.popleft()
            if d >= depth_delta:
                continue
            nbrs = set(self.children.get(node, ())) | set(self.parents.get(node, ()))
            # also siblings: parent's other children
            for par in self.parents.get(node, ()):
                nbrs |= self.children.get(par, set())
            for nb in nbrs:
                if nb not in dist:
                    dist[nb] = d + 1
                    q.append((nb, d + 1))
        return dist

    @property
    def n_edges(self) -> int:
        return sum(len(v) for v in self.children.values())


class RelationalAttrIndex:
    """Doc attribute store + simple query→predicate matching for Stage-1/2."""

    def __init__(self) -> None:
        self.attrs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "RelationalAttrIndex":
        idx = cls()
        path = Path(path)
        if not path.exists():
            return idx
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            did = r.get("doc_id")
            attrs = r.get("attributes") or {}
            if did and isinstance(attrs, dict):
                idx.attrs[str(did)] = attrs
        return idx

    def extract_query_constraints(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        out: Dict[str, Any] = {}
        # money amounts (numeric + common word forms)
        m = re.search(r"\$?\s*([\d,.]+)\s*(million|m\b)", q)
        if m:
            out["amount"] = float(m.group(1).replace(",", "")) * 1_000_000
        else:
            m2 = re.search(r"\$\s*([\d,]+)", q)
            if m2:
                out["amount"] = float(m2.group(1).replace(",", ""))
        word_millions = {
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "fifty": 50,
        }
        if "amount" not in out:
            mw = re.search(
                r"\b(" + "|".join(word_millions) + r")[\s-]*million",
                q,
            )
            if mw:
                out["amount"] = float(word_millions[mw.group(1)]) * 1_000_000
        if "without" in q and ("callback" in q or "phone" in q or "voice" in q):
            out["requires_callback"] = False
        elif "callback" in q or "voice" in q:
            out["requires_callback"] = True
        if "non-repetitive" in q or "non repetitive" in q or "free-form" in q:
            out["is_repetitive"] = False
        elif "repetitive" in q or "standing settlement" in q:
            out["is_repetitive"] = True
        # Waiver-without-callback commercial wires are typically SSI / repetitive
        if out.get("requires_callback") is False and "fedwire" in q:
            out.setdefault("is_repetitive", True)
            out.setdefault("transfer_type", "commercial_fedwire")
        if any(w in q for w in ("return an", "returned", "reject", "rejected for same-day")):
            out["disposition"] = "reject"
        if "instead of placing" in q and "interest-bearing" in q:
            out["disposition"] = "reject"
        if "block" in q and "interest-bearing" in q and "instead of" not in q:
            out["disposition"] = "block"
        if "london" in q:
            out["timezone"] = "london"
            out["same_day_if_before_cutoff"] = True
            out["agreement"] = "gmra"
        if (" est" in f" {q}") or "new york" in q:
            out["timezone"] = "est"
        if "phase 6" in q or "phase6" in q or "uncleared margin rules phase" in q:
            out["umr_phase"] = 6
        if "initial margin" in q and "segregat" in q:
            out["im_segregation"] = "third_party_aca"
        if "third-party" in q or "aca" in q:
            out["im_segregation"] = "third_party_aca"
        if "omnibus" in q:
            out["im_segregation"] = "omnibus"
        if "ssi" in q and (
            "sectoral" in q or "secondary" in q or "short-tenor" in q or "short tenor" in q
        ):
            out["sanctions_list"] = "ssi"
            out["allow_secondary_grandfathered"] = True
        if "comprehensively sanctioned geography" in q or "sanctioned geography" in q:
            out["sanctions_list"] = "geography_only"
            out["sdn_property_interest"] = False
        if "sdn" in q:
            out["sanctions_list"] = "sdn"
        if "unlisted" in q:
            out["entity_listing"] = "unlisted"
        if "tier-2" in q or "tier 2" in q:
            out["trust_tier"] = 2
            out["requires_protector_ack"] = True
        if "currently effective" in q or "active standard" in q:
            out["doc_status"] = "active"
        if "nav error" in q or ("shareholder" in q and "reprocessing" in q):
            out["shareholder_reprocessing"] = True
            out["nav_error_bps_min"] = 50
        return out

    def match_predicates(self, query: str, limit: int = 30) -> List[str]:
        cons = self.extract_query_constraints(query)
        if not cons:
            return []
        scored: List[Tuple[int, str]] = []
        for did, attrs in self.attrs.items():
            score = 0
            for k, v in cons.items():
                if k == "amount":
                    amin = attrs.get("amount_min")
                    amax = attrs.get("amount_max")
                    if amin is not None and v >= float(amin):
                        score += 2
                    if amax is not None and v <= float(amax):
                        score += 1
                elif attrs.get(k) == v:
                    score += 3
                elif k in attrs and attrs.get(k) is not None:
                    score -= 1
            if score > 0:
                scored.append((score, did))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [did for _, did in scored[:limit]]

    def predicate_score(self, query: str, doc_id: str) -> float:
        cons = self.extract_query_constraints(query)
        attrs = self.attrs.get(doc_id)
        if not cons or not attrs:
            return 0.0
        score = 0.0
        for k, v in cons.items():
            if k == "amount":
                amin = attrs.get("amount_min")
                amax = attrs.get("amount_max")
                if amin is not None and v >= float(amin):
                    score += 1.0
                if amax is not None and v <= float(amax):
                    score += 0.5
            elif attrs.get(k) == v:
                score += 2.0
            elif k in attrs:
                score -= 1.0
        # Exact multi-constraint satisfaction bonus
        keys = [k for k in cons if k != "amount" or "amount_min" in attrs]
        if keys and all(
            (k == "amount" and attrs.get("amount_min") is not None and cons["amount"] >= float(attrs["amount_min"]))
            or attrs.get(k) == cons[k]
            for k in cons
        ):
            score += 3.0
        return score
