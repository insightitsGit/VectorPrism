"""Guards against the common integration mistake:

  chunk elsewhere (Onyx / LangChain / custom) → embed with a foreign model →
  expect VectorPrism multi-channel recovery.

VectorPrism is not a chunker and not a drop-in for foreign embeddings.
Retrieval requires the frozen base encoder + MultiTaskProjectionAdapter
checkpoint that produced the 1024d PSM tensors in the index.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any, Mapping, Optional

# Shown in README, CLI, and pipeline docs — keep wording aligned.
INTEGRATION_BANNER = """
VectorPrism is NOT a document chunker and NOT compatible with foreign embeddings
(Onyx / OpenAI / Voyage / etc.) as the vectors in a VectorPrism index.

Correct path:
  1. Chunk text however you like (Onyx, LangChain, your splitter) → chunk_text
  2. Encode EVERY chunk with VectorPrism: frozen base encoder + adapter checkpoint
     → contiguous 1024d PSM tensor (6 channel slices + header)
  3. Search with the SAME checkpoint + SAME base encoder (encode_query → Stage-1/2)

Wrong path (will not get multi-channel recovery):
  VectorPrism-inspired chunks + Onyx/default dense embeddings + cosine search
""".strip()


def print_integration_banner(*, file=None) -> None:
    print(INTEGRATION_BANNER, file=file or sys.stderr)
    print(file=file or sys.stderr)


def expected_encoder_from_checkpoint(ckpt: Mapping[str, Any]) -> Optional[str]:
    meta = ckpt.get("meta") or {}
    if not isinstance(meta, Mapping):
        return None
    name = meta.get("encoder")
    return str(name) if name else None


def check_encoder_matches_checkpoint(
    encoder_name: str,
    ckpt: Mapping[str, Any],
    *,
    strict: bool = False,
    context: str = "VectorPrism",
) -> None:
    """Warn (or exit) when --encoder differs from the name stored at train time."""
    expected = expected_encoder_from_checkpoint(ckpt)
    if not expected:
        return
    # Normalize common aliases
    a = encoder_name.strip().lower()
    b = expected.strip().lower()
    if a in {"hash", "hashing", "test"}:
        warnings.warn(
            f"{context}: using HashingEncoder — fine for plumbing smoke tests only. "
            "Production retrieval needs the same sentence-transformers (or other) "
            "base encoder used when the checkpoint was trained, plus the VectorPrism "
            "adapter. Foreign embeddings (e.g. Onyx default) are not a substitute.",
            UserWarning,
            stacklevel=2,
        )
        return
    if a != b and not (a.endswith(b) or b.endswith(a)):
        msg = (
            f"{context}: encoder mismatch — CLI/API encoder={encoder_name!r} but "
            f"checkpoint meta.encoder={expected!r}. Ingest and search must use the "
            "same frozen base encoder that trained this adapter. "
            "Do not substitute Onyx/OpenAI/other embedding vectors for VectorPrism "
            "1024d tensors."
        )
        if strict:
            raise SystemExit(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
