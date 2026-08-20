"""Bitemporal Stage-1 gates (exact int64 epochs — not a 7th embedding channel).

Clocks
------
* **valid time** — when the fact was true in the world:
  ``valid_timestamp`` (= valid_from) .. ``valid_to_timestamp`` (exclusive end, None = open)
* **transaction time** — when the system recorded / knew the chunk:
  ``transaction_timestamp`` (header ``[6:8)``, optional)

As-of semantics
---------------
* ``as_of=T`` keeps rows with ``valid_from <= T`` and (``valid_to is None`` or ``valid_to > T``)
* ``as_of_transaction=T`` keeps rows with missing tx (legacy) OR ``transaction_time <= T``

Never store these as float embedding *values*; use ``temporal_types`` packing only.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


def row_in_valid_time(row: Mapping[str, Any], as_of: int) -> bool:
    """Half-open interval [valid_from, valid_to)."""
    valid_from = int(row.get("valid_timestamp", 0))
    if valid_from > as_of:
        return False
    raw_to = row.get("valid_to_timestamp")
    if raw_to is None:
        return True
    return int(raw_to) > as_of


def row_in_transaction_time(row: Mapping[str, Any], as_of_transaction: int) -> bool:
    """What the system knew at ``as_of_transaction`` (legacy rows without tx pass)."""
    raw = row.get("transaction_timestamp")
    if raw is None:
        return True
    return int(raw) <= as_of_transaction


def passes_bitemporal_filters(
    row: Mapping[str, Any],
    *,
    as_of: Optional[int] = None,
    as_of_transaction: Optional[int] = None,
) -> bool:
    if as_of is not None and not row_in_valid_time(row, int(as_of)):
        return False
    if as_of_transaction is not None and not row_in_transaction_time(
        row, int(as_of_transaction)
    ):
        return False
    return True
