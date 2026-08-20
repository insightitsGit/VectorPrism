"""
tensor_contract.py - Fixed 1024-Dimensional Tensor Contract (VectorPrism)

Pillars unchanged:
  - 16d header + 6 channel slices totaling 1024
  - Exact bitmask/timestamp bit-reinterpret packing

Improvements (layout-preserving):
  - Reserved header slot [5] now stores model_version (uint32↔f32)
  - Remaining reserved slots stay zero unless transaction_time is set
    (planned true bitemporal: valid_time [3:5), transaction_time [6:8);
     never store epochs as float *values*)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from vectorprism.temporal_types import (
    coerce_unix_epoch_seconds,
    pack_int64_as_2f32,
    unpack_2f32_as_int64,
)


@dataclass(frozen=True)
class TensorSlice:
    start: int
    end: int
    name: str

    @property
    def length(self) -> int:
        return self.end - self.start


class PSMTensorContract:
    TOTAL_DIM: int = 1024

    # Top-level memory layout (unchanged offsets — VectorPrism pillar)
    HEADER: TensorSlice = TensorSlice(0, 16, "Header Control Manifest")
    DENSE_CORE: TensorSlice = TensorSlice(16, 384, "Dense Semantic Core")
    RELATIONAL: TensorSlice = TensorSlice(384, 512, "Relational Group Algebra")
    DISENTANGLED: TensorSlice = TensorSlice(512, 640, "Disentangled Latent")
    HYPERBOLIC: TensorSlice = TensorSlice(640, 768, "Hyperbolic Taxonomy")
    IDENTITY: TensorSlice = TensorSlice(768, 896, "Identity Anchor")
    CAUSAL_TIME: TensorSlice = TensorSlice(896, 1024, "Time ODE & Causality")

    # Sub-layout WITHIN the header
    HDR_BITMASK = TensorSlice(0, 1, "Channel Bitmask (uint32 reinterpreted)")
    HDR_TRUTH = TensorSlice(1, 2, "Epistemic Truth Score")
    HDR_ANCHOR = TensorSlice(2, 3, "Identity Anchor Distance")
    # valid_time (unix seconds). Second clock (transaction_time) at [6:8).
    HDR_TIMESTAMP = TensorSlice(3, 5, "Valid-time Timestamp (int64 as 2xf32)")
    HDR_MODEL_VERSION = TensorSlice(5, 6, "Model Version (uint32 reinterpreted)")
    HDR_TX_TIMESTAMP = TensorSlice(6, 8, "Reserved transaction_time (int64 as 2xf32)")
    HDR_RESERVED = TensorSlice(8, 16, "Reserved (zero-filled)")

    # Channel enable bits for bitmask (optional bookkeeping)
    BIT_DENSE = 1 << 0
    BIT_RELATIONAL = 1 << 1
    BIT_DISENTANGLED = 1 << 2
    BIT_HYPERBOLIC = 1 << 3
    BIT_IDENTITY = 1 << 4
    BIT_CAUSAL = 1 << 5

    @classmethod
    def _unpack_one_header(cls, h: np.ndarray) -> Dict[str, Any]:
        h = np.ascontiguousarray(h, dtype=np.float32).reshape(16)
        bitmask = int(np.frombuffer(np.ascontiguousarray(h[0:1]).tobytes(), dtype=np.uint32)[0])
        epistemic_truth = float(np.clip(float(h[1]), 0.0, 1.0))
        anchor_distance = float(h[2])
        timestamp = unpack_2f32_as_int64(h[3:5])
        model_version = int(
            np.frombuffer(np.ascontiguousarray(h[5:6]).tobytes(), dtype=np.uint32)[0]
        )
        tx_raw = h[6:8]
        transaction_time = (
            unpack_2f32_as_int64(tx_raw) if not np.allclose(tx_raw, 0.0) else None
        )
        return {
            "bitmask": bitmask,
            "epistemic_truth": epistemic_truth,
            "anchor_distance": anchor_distance,
            "timestamp": timestamp,
            "model_version": model_version,
            "transaction_time": transaction_time,
        }

    @classmethod
    def unpack_header(cls, tensor_1024d: np.ndarray) -> Dict[str, Any]:
        """Extracts metadata from the header slice [0..15] in O(1) time."""
        assert tensor_1024d.shape[-1] == cls.TOTAL_DIM, f"Expected {cls.TOTAL_DIM}d tensor"
        header_raw = np.ascontiguousarray(
            tensor_1024d[..., cls.HEADER.start : cls.HEADER.end], dtype=np.float32
        )
        if header_raw.ndim == 1:
            return cls._unpack_one_header(header_raw)

        # Batched (N, 16) — preserve array outputs for callers that pass batches
        n = header_raw.shape[0]
        bitmask = np.empty(n, dtype=np.uint32)
        epistemic_truth = np.empty(n, dtype=np.float32)
        anchor_distance = np.empty(n, dtype=np.float32)
        timestamp = np.empty(n, dtype=np.int64)
        model_version = np.empty(n, dtype=np.uint32)
        transaction_time = []
        for i in range(n):
            row = cls._unpack_one_header(header_raw[i])
            bitmask[i] = row["bitmask"]
            epistemic_truth[i] = row["epistemic_truth"]
            anchor_distance[i] = row["anchor_distance"]
            timestamp[i] = row["timestamp"]
            model_version[i] = row["model_version"]
            transaction_time.append(row["transaction_time"])
        return {
            "bitmask": bitmask,
            "epistemic_truth": epistemic_truth,
            "anchor_distance": anchor_distance,
            "timestamp": timestamp,
            "model_version": model_version,
            "transaction_time": transaction_time,
        }

    @classmethod
    def pack_header(
        cls,
        bitmask: int,
        epistemic_truth: float,
        anchor_distance: float,
        timestamp: Any,
        model_version: int = 0,
        transaction_time: Any = None,
    ) -> np.ndarray:
        """Packs metadata into a 16-element float32 header block.

        ``timestamp`` (valid time) is Unix epoch **seconds**, coerced then
        bit-reinterpreted as int64 → 2×f32. Optional ``transaction_time`` uses
        the same packing in ``[6:8)``; leave None to keep reserved zeros
        (backward compatible with existing tensors).
        """
        ts = coerce_unix_epoch_seconds(timestamp, field_name="timestamp")
        assert ts is not None
        header = np.zeros(16, dtype=np.float32)
        header[0:1] = np.frombuffer(np.uint32(int(bitmask)).tobytes(), dtype=np.float32)
        header[1] = np.clip(float(epistemic_truth), 0.0, 1.0)
        header[2] = float(anchor_distance)
        header[3:5] = pack_int64_as_2f32(ts)
        header[5:6] = np.frombuffer(
            np.uint32(int(model_version)).tobytes(), dtype=np.float32
        )
        if transaction_time is not None:
            tx = coerce_unix_epoch_seconds(
                transaction_time, field_name="transaction_time"
            )
            assert tx is not None
            header[6:8] = pack_int64_as_2f32(tx)
        # header[8:16] stays zero — reserved
        return header

    @classmethod
    def default_channel_bitmask(cls, enabled: Optional[Dict[str, bool]] = None) -> int:
        """Build a bitmask for which channels are considered active at ingest."""
        flags = {
            "dense": True,
            "relational": False,
            "disentangled": False,
            "hyperbolic": False,
            "identity": True,
            "causal": False,
        }
        if enabled:
            flags.update(enabled)
        mask = 0
        if flags.get("dense"):
            mask |= cls.BIT_DENSE
        if flags.get("relational"):
            mask |= cls.BIT_RELATIONAL
        if flags.get("disentangled"):
            mask |= cls.BIT_DISENTANGLED
        if flags.get("hyperbolic"):
            mask |= cls.BIT_HYPERBOLIC
        if flags.get("identity"):
            mask |= cls.BIT_IDENTITY
        if flags.get("causal"):
            mask |= cls.BIT_CAUSAL
        return mask


# Product-facing alias (internals keep PSM* for compatibility)
VectorPrismTensorContract = PSMTensorContract
