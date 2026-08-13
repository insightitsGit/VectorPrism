"""
structure_extract.py — Automated extraction of VectorPrism structure indexes.

Produces the same artifacts Stage-1 expects:
  - causal_graph.jsonl      {earlier_doc_id, later_doc_id, relation, confidence}
  - hyperbolic_graph.jsonl  {parent_doc_id, child_doc_id, relation, confidence}
  - relational_attrs.jsonl  {doc_id, attributes, confidence}

Backends:
  - heuristic  (default, offline, deterministic)
  - llm        OpenAI-compatible Chat Completions with JSON-schema output
               (OPENAI_API_KEY or VECTORPRISM_LLM_API_KEY + optional base URL)

Does not invent edges from eval gold labels — extraction uses document text only.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


CAUSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "earlier_doc_id": {"type": "string"},
                    "later_doc_id": {"type": "string"},
                    "relation": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["earlier_doc_id", "later_doc_id"],
            },
        }
    },
    "required": ["edges"],
}

TAXONOMY_SCHEMA = {
    "type": "object",
    "properties": {
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "parent_doc_id": {"type": "string"},
                    "child_doc_id": {"type": "string"},
                    "relation": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["parent_doc_id", "child_doc_id"],
            },
        }
    },
    "required": ["edges"],
}

RELATIONAL_SCHEMA = {
    "type": "object",
    "properties": {
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                    "attributes": {"type": "object"},
                    "confidence": {"type": "number"},
                },
                "required": ["doc_id", "attributes"],
            },
        }
    },
    "required": ["documents"],
}


@dataclass
class ExtractionResult:
    causal_edges: List[Dict[str, Any]] = field(default_factory=list)
    taxonomy_edges: List[Dict[str, Any]] = field(default_factory=list)
    relational_attrs: List[Dict[str, Any]] = field(default_factory=list)
    backend: str = "heuristic"
    meta: Dict[str, Any] = field(default_factory=dict)


def load_documents_jsonl(path: str | Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[str(r["document_id"])] = str(r["chunk_text"])
    return out


def write_extraction(result: ExtractionResult, out_dir: str | Path) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "causal_graph": out_dir / "causal_graph.jsonl",
        "hyperbolic_graph": out_dir / "hyperbolic_graph.jsonl",
        "relational_attrs": out_dir / "relational_attrs.jsonl",
        "meta": out_dir / "extraction_meta.json",
    }
    _write_jsonl(paths["causal_graph"], result.causal_edges)
    _write_jsonl(paths["hyperbolic_graph"], result.taxonomy_edges)
    _write_jsonl(paths["relational_attrs"], result.relational_attrs)
    paths["meta"].write_text(
        json.dumps(
            {
                "backend": result.backend,
                "n_causal": len(result.causal_edges),
                "n_taxonomy": len(result.taxonomy_edges),
                "n_relational": len(result.relational_attrs),
                **result.meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _tokens(text: str) -> Set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", text.lower())}


def _cluster_id(doc_id: str) -> str:
    parts = doc_id.split("_")
    if doc_id.startswith("ADV_") and len(parts) >= 2:
        return "_".join(parts[:2])
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return doc_id


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


_CAUSE_MARKERS = re.compile(
    r"\b(when|if|because|due to|triggered by|followed by|after|causes?|results? in|imposes?|must)\b",
    re.I,
)
_SYMPTOM_MARKERS = re.compile(
    r"\b(halt|freeze|lock|pause|reject|disconnect|default|incident|runbook|queued|blocked|returned)\b",
    re.I,
)
_POLICY_MARKERS = re.compile(r"\b(POL-|SOP-|policy|manual|matrix|section|tier-|category|child node)\b", re.I)


def extract_relational_attrs_from_text(doc_id: str, text: str) -> Dict[str, Any]:
    """Deterministic attribute parse used by heuristic and as LLM post-check."""
    q = text.lower()
    attrs: Dict[str, Any] = {}
    m = re.search(r"\$?\s*([\d,.]+)\s*(million|m\b)", q)
    if m:
        attrs["amount_min"] = float(m.group(1).replace(",", "")) * 1_000_000
    else:
        m2 = re.search(r"exceeds?\s+([\d,.]+)\s*million", q)
        if m2:
            attrs["amount_min"] = float(m2.group(1).replace(",", "")) * 1_000_000
        elif "five million" in q:
            attrs["amount_min"] = 5_000_000.0

    if "waiv" in q and ("callback" in q or "voice" in q):
        attrs["requires_callback"] = False
    elif "callback" in q or "voice" in q:
        attrs["requires_callback"] = True

    if "repetitive" in q or "standing settlement" in q or "ssi" in q:
        attrs["is_repetitive"] = "non-repetitive" not in q and "non repetitive" not in q

    if re.search(r"\breject\b", q) and ("disposition" in q or "returned" in q or "return" in q):
        attrs["disposition"] = "reject"
    if "interest-bearing" in q and "block" in q:
        attrs.setdefault("disposition", "block")

    if "london" in q:
        attrs["timezone"] = "london"
        attrs["agreement"] = "gmra" if "gmra" in q else attrs.get("agreement")
    if " est" in f" {q}" or "new york" in q:
        attrs["timezone"] = "est"

    if "phase 6" in q or "phase6" in q:
        attrs["umr_phase"] = 6
    if "third-party" in q or "account control agreement" in q or " aca" in q:
        attrs["im_segregation"] = "third_party_aca"
    if "omnibus" in q:
        attrs["im_segregation"] = "omnibus"

    if "ssi" in q and ("sectoral" in q or "grandfather" in q or "secondary" in q):
        attrs["sanctions_list"] = "ssi"
        attrs["allow_secondary_grandfathered"] = True
    if "sdn" in q:
        attrs["sanctions_list"] = "sdn"
    if "comprehensively sanctioned geography" in q or "sanctioned geography" in q:
        attrs["sanctions_list"] = "geography_only"
        attrs["sdn_property_interest"] = False

    if "tier-2" in q or "tier 2" in q:
        attrs["trust_tier"] = 2
    if "unlisted" in q:
        attrs["entity_listing"] = "unlisted"
    if "protector" in q:
        attrs["requires_protector_ack"] = True

    if "effective" in q or "active standard" in q:
        attrs["doc_status"] = "active"
    if "superseded" in q or "retired" in q:
        attrs["doc_status"] = "superseded"

    if "nav" in q and ("basis point" in q or "bps" in q or "fifty" in q):
        attrs["nav_error_bps_min"] = 50
        attrs["shareholder_reprocessing"] = "reprocess" in q or "shareholder" in q

    if "fedwire" in q and "commercial" in q:
        attrs["transfer_type"] = "commercial_fedwire"

    return attrs


class HeuristicStructureExtractor:
    """Offline extractor: cluster-local causal/taxonomy links + regex attributes."""

    def __init__(self, min_overlap: float = 0.08, max_causes_per_symptom: int = 3):
        self.min_overlap = min_overlap
        self.max_causes_per_symptom = max_causes_per_symptom

    def extract(self, documents: Dict[str, str]) -> ExtractionResult:
        by_cluster: Dict[str, List[str]] = defaultdict(list)
        for did in documents:
            by_cluster[_cluster_id(did)].append(did)

        causal: List[Dict[str, Any]] = []
        tax: List[Dict[str, Any]] = []
        rel: List[Dict[str, Any]] = []

        for cluster, members in by_cluster.items():
            texts = {m: documents[m] for m in members}
            tok = {m: _tokens(t) for m, t in texts.items()}

            # Role scores without using *_gold labels
            cause_score = {}
            symptom_score = {}
            for m, t in texts.items():
                cause_score[m] = len(_CAUSE_MARKERS.findall(t)) + (2 if _POLICY_MARKERS.search(t) else 0)
                symptom_score[m] = len(_SYMPTOM_MARKERS.findall(t))
                if "runbook" in t.lower() or "incident" in t.lower() or "playbook" in t.lower():
                    symptom_score[m] += 2

            # Causal: high-cause docs → high-symptom docs with lexical overlap
            causes = sorted(members, key=lambda d: (-cause_score[d], d))
            symptoms = sorted(members, key=lambda d: (-symptom_score[d], d))
            for sym in symptoms:
                if symptom_score[sym] <= 0:
                    continue
                scored: List[Tuple[float, str]] = []
                for cau in causes:
                    if cau == sym:
                        continue
                    if cause_score[cau] <= 0:
                        continue
                    ov = _jaccard(tok[cau], tok[sym])
                    if ov < self.min_overlap and cause_score[cau] < 2:
                        continue
                    scored.append((cause_score[cau] + 5.0 * ov, cau))
                scored.sort(reverse=True)
                for score, cau in scored[: self.max_causes_per_symptom]:
                    causal.append(
                        {
                            "earlier_doc_id": cau,
                            "later_doc_id": sym,
                            "relation": "auto_causes_symptom",
                            "confidence": float(min(1.0, score / 10.0)),
                            "cluster": cluster,
                        }
                    )

            # Taxonomy: pick a root (most policy-like) and attach others as children
            root = max(members, key=lambda d: (_POLICY_MARKERS.search(texts[d]) is not None, cause_score[d], -len(texts[d]), d))
            for ch in members:
                if ch == root:
                    continue
                # Prefer attaching leaf-like / section-like children
                conf = 0.55
                if "section" in texts[ch].lower() or "child" in texts[ch].lower() or "tier-" in texts[ch].lower():
                    conf = 0.8
                tax.append(
                    {
                        "parent_doc_id": root,
                        "child_doc_id": ch,
                        "relation": "auto_policy_child",
                        "confidence": conf,
                        "cluster": cluster,
                    }
                )

        for did, text in documents.items():
            attrs = extract_relational_attrs_from_text(did, text)
            if attrs:
                rel.append({"doc_id": did, "attributes": attrs, "confidence": 0.7})

        # Dedup causal edges
        seen = set()
        dedup_causal = []
        for e in causal:
            key = (e["earlier_doc_id"], e["later_doc_id"])
            if key in seen:
                continue
            seen.add(key)
            dedup_causal.append(e)

        return ExtractionResult(
            causal_edges=dedup_causal,
            taxonomy_edges=tax,
            relational_attrs=rel,
            backend="heuristic",
            meta={"n_clusters": len(by_cluster), "n_docs": len(documents)},
        )


class LLMStructureExtractor:
    """OpenAI-compatible structured extraction over document batches."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        batch_size: int = 12,
        timeout_s: float = 90.0,
    ):
        self.api_key = api_key or os.environ.get("VECTORPRISM_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("VECTORPRISM_LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("VECTORPRISM_LLM_MODEL") or "gpt-4o-mini"
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        if not self.api_key:
            raise RuntimeError(
                "LLM backend requires OPENAI_API_KEY or VECTORPRISM_LLM_API_KEY"
            )

    def _chat_json(self, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "structure_extract", "schema": schema, "strict": False},
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # Fallback if provider lacks json_schema: ask for raw JSON
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Retry without response_format for incompatible gateways
            err = e.read().decode("utf-8", errors="replace")
            if e.code in (400, 404, 422):
                payload.pop("response_format", None)
                payload["messages"][0]["content"] += (
                    "\nReturn ONLY valid JSON matching the schema. Schema: "
                    + json.dumps(schema)
                )
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=data,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            else:
                raise RuntimeError(f"LLM HTTP {e.code}: {err[:500]}") from e

        content = body["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        content = str(content).strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        return json.loads(content)

    def extract(self, documents: Dict[str, str]) -> ExtractionResult:
        ids = sorted(documents)
        causal: List[Dict[str, Any]] = []
        tax: List[Dict[str, Any]] = []
        rel: List[Dict[str, Any]] = []
        batches = 0

        for i in range(0, len(ids), self.batch_size):
            batch_ids = ids[i : i + self.batch_size]
            batch = [{"doc_id": d, "text": documents[d][:1200]} for d in batch_ids]
            batches += 1
            blob = json.dumps(batch, ensure_ascii=False)

            sys_c = (
                "You extract causal document edges for retrieval. "
                "earlier_doc_id is the cause/root-policy; later_doc_id is the symptom/effect doc. "
                "Only use provided doc_ids. Prefer high-precision edges."
            )
            c = self._chat_json(sys_c, f"Documents:\n{blob}", CAUSAL_SCHEMA)
            for e in c.get("edges") or []:
                if e.get("earlier_doc_id") in documents and e.get("later_doc_id") in documents:
                    causal.append(
                        {
                            "earlier_doc_id": e["earlier_doc_id"],
                            "later_doc_id": e["later_doc_id"],
                            "relation": e.get("relation") or "llm_causes",
                            "confidence": float(e.get("confidence") or 0.6),
                        }
                    )

            sys_t = (
                "You extract taxonomy parent→child edges among policy documents. "
                "Only use provided doc_ids."
            )
            t = self._chat_json(sys_t, f"Documents:\n{blob}", TAXONOMY_SCHEMA)
            for e in t.get("edges") or []:
                if e.get("parent_doc_id") in documents and e.get("child_doc_id") in documents:
                    tax.append(
                        {
                            "parent_doc_id": e["parent_doc_id"],
                            "child_doc_id": e["child_doc_id"],
                            "relation": e.get("relation") or "llm_sub_policy",
                            "confidence": float(e.get("confidence") or 0.6),
                        }
                    )

            sys_r = (
                "Extract structured attributes per document for predicate matching. "
                "Useful keys: amount_min, requires_callback, is_repetitive, disposition, "
                "timezone, umr_phase, im_segregation, sanctions_list, trust_tier, "
                "entity_listing, doc_status, agreement."
            )
            r = self._chat_json(sys_r, f"Documents:\n{blob}", RELATIONAL_SCHEMA)
            for row in r.get("documents") or []:
                did = row.get("doc_id")
                attrs = row.get("attributes") or {}
                if did in documents and isinstance(attrs, dict) and attrs:
                    # Merge regex attrs as safety net
                    merged = extract_relational_attrs_from_text(did, documents[did])
                    merged.update(attrs)
                    rel.append(
                        {
                            "doc_id": did,
                            "attributes": merged,
                            "confidence": float(row.get("confidence") or 0.6),
                        }
                    )

        # Dedup
        def _dedup(rows: List[Dict[str, Any]], keys: Tuple[str, str]) -> List[Dict[str, Any]]:
            seen = set()
            out = []
            for r in rows:
                k = (r[keys[0]], r[keys[1]])
                if k in seen:
                    continue
                seen.add(k)
                out.append(r)
            return out

        return ExtractionResult(
            causal_edges=_dedup(causal, ("earlier_doc_id", "later_doc_id")),
            taxonomy_edges=_dedup(tax, ("parent_doc_id", "child_doc_id")),
            relational_attrs=rel,
            backend=f"llm:{self.model}",
            meta={"n_docs": len(documents), "batches": batches, "base_url": self.base_url},
        )


def extract_structure(
    documents: Dict[str, str],
    *,
    backend: str = "auto",
) -> ExtractionResult:
    """
    backend:
      - heuristic
      - llm
      - auto  → llm if API key present else heuristic
    """
    if backend == "auto":
        if os.environ.get("VECTORPRISM_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"):
            backend = "llm"
        else:
            backend = "heuristic"
    if backend == "llm":
        return LLMStructureExtractor().extract(documents)
    if backend == "heuristic":
        return HeuristicStructureExtractor().extract(documents)
    raise ValueError(f"unknown backend {backend}")


def _edge_set(path: Path, a: str, b: str) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get(a) and r.get(b):
            out.add((str(r[a]), str(r[b])))
    return out


def _attr_pairs(path: Path) -> Set[Tuple[str, str, str]]:
    """(doc_id, key, canonical_value_str)"""
    out: Set[Tuple[str, str, str]] = set()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        did = r.get("doc_id")
        attrs = r.get("attributes") or {}
        if not did or not isinstance(attrs, dict):
            continue
        for k, v in attrs.items():
            out.add((str(did), str(k), json.dumps(v, sort_keys=True)))
    return out


def compare_extractions(pred_dir: Path, gold_dir: Path) -> Dict[str, Any]:
    """Edge / attribute precision-recall of auto extraction vs curated gold files."""

    def prf(pred: Set, gold: Set) -> Dict[str, float]:
        if not pred and not gold:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
        tp = len(pred & gold)
        fp = len(pred - gold)
        fn = len(gold - pred)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    causal = prf(
        _edge_set(pred_dir / "causal_graph.jsonl", "earlier_doc_id", "later_doc_id"),
        _edge_set(gold_dir / "causal_graph.jsonl", "earlier_doc_id", "later_doc_id"),
    )
    tax = prf(
        _edge_set(pred_dir / "hyperbolic_graph.jsonl", "parent_doc_id", "child_doc_id"),
        _edge_set(gold_dir / "hyperbolic_graph.jsonl", "parent_doc_id", "child_doc_id"),
    )
    rel = prf(_attr_pairs(pred_dir / "relational_attrs.jsonl"), _attr_pairs(gold_dir / "relational_attrs.jsonl"))
    return {"causal_edges": causal, "taxonomy_edges": tax, "relational_attrs": rel}
