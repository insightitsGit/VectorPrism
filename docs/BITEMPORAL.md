# Bitemporal retrieval (design — VectorPrism 0.1.3+)

**Status:** Implemented · **opt-in only**.  
**Not:** a 7th cosine “time embedding” channel.  
**Default product path:** unchanged if you never pass `as_of` / `as_of_transaction`.

## Opt-in contract (read this first)

| Call | Temporal Stage-1 gate |
|------|------------------------|
| `engine.search(q, text, top_k=5)` | **Off** — same as pre-0.1.3 |
| `vectorprism search ...` (no `--as-of*`) | **Off** |
| `engine.search(..., as_of=T)` | **On** — valid-time window |
| `engine.search(..., as_of_transaction=T)` | **On** — transaction-time ceiling |
| Both flags | **On** — both clocks |

Storing `timestamp` / `valid_to` / `transaction_time` at ingest does **not** by itself
change ranking. Filters apply only when search opts in.

## Problem

Typical RAG stacks either ignore document time or leave `created_at` as optional metadata the app must filter. Compliance / policy / runbook retrieval fails as **right topic, wrong era** (expired policy, superseded limit, old incident note).

## Design choice

| Approach | Verdict |
|----------|---------|
| Learned temporal embedding channel | Rejected for v1 — soft proximity cannot prove compliance |
| Exact **valid time** + **transaction time** as Stage-1 gates | **Shipped, opt-in** — same class as `model_version` / truth / anchor |

## Two clocks

| Clock | Fields | Meaning |
|-------|--------|---------|
| **Valid time** | `valid_timestamp` (from), `valid_to_timestamp` (exclusive end, NULL=open) | When the *fact* was true in the world |
| **Transaction time** | `transaction_timestamp` (+ header `[6:8)`) | When *your system* recorded this version |

Interval semantics: **half-open** `[valid_from, valid_to)`.

## Header packing (data-type safety)

- Valid time: header `[3:5)` — int64 ↔ 2×float32 **bit reinterpret** (unix **seconds**)
- Transaction time: header `[6:8)` — same packing; **zeros = unset** (backward compatible)
- Postgres: `BIGINT` columns — **never** `FLOAT` / `REAL`
- Memory NPZ: `int64` arrays (`-1` sentinel = NULL)

See `vectorprism/temporal_types.py`, `vectorprism/bitemporal.py`, `tensor_contract.py`.

## Retrieval API

```python
# Default — no temporal filtering
hits_default = engine.search(query_1024d, query_text, top_k=5)

# Opt-in
hits_as_of = engine.search(query_1024d, query_text, top_k=5, as_of=1_700_000_000)
hits_bi = engine.search(
    ...,
    as_of="2024-01-15T12:00:00Z",
    as_of_transaction="2024-01-10T00:00:00Z",
)
```

CLI:

```bash
# Default
vectorprism search --checkpoint ckpt.pt --query "wire limit"

# Opt-in
vectorprism search --checkpoint ckpt.pt --query "wire limit" --as-of 2024-06-01T00:00:00Z
```

When enabled, Stage-1 applies filters **before** multi-channel Stage-2 (memory / pgvector / Qdrant).

## Ingest

```python
IngestDocument(
    document_id="policy-v2",
    chunk_text="...",
    timestamp=valid_from,       # unix seconds or ISO
    valid_to=valid_to_or_None,  # exclusive end
    transaction_time=ingested_at,
)
```

Labels are stored for later opt-in search; they do not activate gates by themselves.

## What this is not

- Not a replacement for the **causal** channel (cause→effect order)
- Not automatic date parsing from free-text queries (pass `as_of` explicitly)
- Not claimed on the adversarial finance Miss@10 scorecard (orthogonal feature)
- Not PrismManifest (money / tax_year / Ed25519 gate — different product layer)

## Tests

- `tests/test_temporal_types.py` — int64 packing safety  
- `tests/test_bitemporal.py` — filters + **with vs without** `as_of` parity  
