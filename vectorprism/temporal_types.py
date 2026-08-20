"""Safe temporal / epoch helpers for the VectorPrism header contract.

Data-type rules (do not relax):

1. **Canonical form** is Unix epoch **seconds** as a Python ``int`` / ``numpy.int64``.
2. Inside the 1024d header, the epoch is stored ONLY as an **int64 bit-reinterpreted
   as two float32** slots (``HDR_TIMESTAMP``). Never write the epoch as a float
   *value* into a header slot — that loses precision above 2^24.
3. In Postgres, ``valid_timestamp`` is ``BIGINT`` (seconds). Never ``FLOAT`` / ``REAL``.
4. In memory NPZ sidecars, prefer ``int64`` arrays for epochs (float64 is OK for
   unix-seconds ranges; float32 is not).
5. Future **transaction time** (true bitemporal) must use reserved header ``[6:8)``
   with the same int64↔2×f32 packing — leave ``[6:16)`` zero until then.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Union

import numpy as np

# Signed int64 bounds (numpy / Postgres BIGINT)
_INT64_MIN = -9223372036854775808
_INT64_MAX = 9223372036854775807

# Practical RAG window (not a hard DB limit) — reject absurd values early
_PRACTICAL_MIN = -62135596800  # year ~0001 UTC
_PRACTICAL_MAX = 4102444800  # year ~2100 UTC


UnixEpochSeconds = Union[int, np.integer]


def pack_int64_as_2f32(value: int) -> np.ndarray:
    """Bit-reinterpret one int64 → shape (2,) float32 (endian-native, exact)."""
    v = np.int64(int(value))
    return np.frombuffer(v.tobytes(), dtype=np.float32).copy()


def unpack_2f32_as_int64(pair: np.ndarray) -> int:
    """Inverse of :func:`pack_int64_as_2f32` (exact)."""
    buf = np.ascontiguousarray(pair, dtype=np.float32).reshape(-1)[:2]
    if buf.size != 2:
        raise ValueError(f"expected 2 float32 slots for int64, got shape {pair.shape}")
    return int(np.frombuffer(buf.tobytes(), dtype=np.int64)[0])


def coerce_unix_epoch_seconds(
    value: Any,
    *,
    field_name: str = "timestamp",
    allow_none: bool = False,
    default: Optional[int] = None,
    practical_range: bool = True,
) -> Optional[int]:
    """Normalize partner input to int epoch seconds.

    Accepts: int, numpy integer, whole-number float (exact), digit strings,
    ISO-8601 strings (``Z`` or offset). Rejects fractional floats and
    float32-unsafe paths by never returning a float.
    """
    if value is None:
        if allow_none:
            return default
        raise ValueError(f"{field_name} is required (unix epoch seconds int)")

    if isinstance(value, np.integer):
        out = int(value)
    elif isinstance(value, bool):
        # bool is a subclass of int — reject (True → 1 is never a real epoch)
        raise TypeError(f"{field_name} must be unix epoch seconds, not bool")
    elif isinstance(value, int):
        out = value
    elif isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{field_name} must be finite, got {value!r}")
        if abs(value - round(value)) > 1e-9:
            raise ValueError(
                f"{field_name} must be whole unix seconds (got fractional {value!r}). "
                "Do not store epochs as float embeddings."
            )
        out = int(round(value))
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            raise ValueError(f"{field_name} is an empty string")
        if s.isdigit() or (s[0] == "-" and s[1:].isdigit()):
            out = int(s)
        else:
            # ISO-8601 → UTC epoch seconds
            iso = s.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(iso)
            except ValueError as e:
                raise ValueError(
                    f"{field_name} must be unix seconds or ISO-8601 datetime, got {value!r}"
                ) from e
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            out = int(dt.timestamp())
    elif isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        out = int(dt.timestamp())
    else:
        raise TypeError(
            f"{field_name} must be int/str/datetime unix epoch seconds, got {type(value).__name__}"
        )

    if out < _INT64_MIN or out > _INT64_MAX:
        raise ValueError(f"{field_name}={out} outside int64 range")

    if practical_range and (out < _PRACTICAL_MIN or out > _PRACTICAL_MAX):
        raise ValueError(
            f"{field_name}={out} outside practical RAG range "
            f"[{_PRACTICAL_MIN}, {_PRACTICAL_MAX}] (~year 0001–2100). "
            "If intentional, call coerce_unix_epoch_seconds(..., practical_range=False)."
        )
    return out
