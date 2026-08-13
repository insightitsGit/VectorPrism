"""
tensor_contract.py - Fixed 1024-Dimensional Tensor Contract (VectorPrism)

Pillars unchanged:
  - 16d header + 6 channel slices totaling 1024
  - Exact bitmask/timestamp bit-reinterpret packing

Improvements (layout-preserving):
  - Reserved header slot [5] now stores model_version (uint32↔f32)
  - Remaining [6..15] stay reserved/zero for future fields
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


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
    HDR_TIMESTAMP = TensorSlice(3, 5, "Bitemporal Timestamp (int64 as 2xf32)")
    HDR_MODEL_VERSION = TensorSlice(5, 6, "Model Version (uint32 reinterpreted)")
    HDR_RESERVED = TensorSlice(6, 16, "Reserved (zero-filled)")

    # Channel enable bits for bitmask (optional bookkeeping)
    BIT_DENSE = 1 << 0
    BIT_RELATIONAL = 1 << 1
    BIT_DISENTANGLED = 1 << 2
    BIT_HYPERBOLIC = 1 << 3
    BIT_IDENTITY = 1 << 4
    BIT_CAUSAL = 1 << 5

    @classmethod
    def unpack_header(cls, tensor_1024d: np.ndarray) -> Dict[str, Any]:
        """Extracts metadata from the header slice [0..15] in O(1) time."""
        assert tensor_1024d.shape[-1] == cls.TOTAL_DIM, f"Expected {cls.TOTAL_DIM}d tensor"
        header_raw = np.ascontiguousarray(
            tensor_1024d[..., cls.HEADER.start:cls.HEADER.end]
        ).astype(np.float32)

        bitmask = header_raw[..., 0:1].view(np.uint32)[..., 0]
        epistemic_truth = header_raw[..., 1]
        anchor_distance = header_raw[..., 2]
        timestamp = header_raw[..., 3:5].view(np.int64)[..., 0]
        model_version = header_raw[..., 5:6].view(np.uint32)[..., 0]

        def _scalar(x):
            return int(x) if np.isscalar(x) or getattr(x, "ndim", 1) == 0 else x

        return {
            "bitmask": _scalar(bitmask),
            "epistemic_truth": float(np.clip(epistemic_truth, 0.0, 1.0)),
            "anchor_distance": float(anchor_distance),
            "timestamp": _scalar(timestamp),
            "model_version": _scalar(model_version),
        }

    @classmethod
    def pack_header(
        cls,
        bitmask: int,
        epistemic_truth: float,
        anchor_distance: float,
        timestamp: int,
        model_version: int = 0,
    ) -> np.ndarray:
        """Packs metadata into a 16-element float32 header block.

        `timestamp` is Unix epoch seconds, bit-reinterpreted as int64 -> 2xf32.
        `model_version` is a monotonic uint32 encoder/adapter revision id.
        """
        header = np.zeros(16, dtype=np.float32)
        header[0:1] = np.array([int(bitmask)], dtype=np.uint32).view(np.float32)
        header[1] = np.clip(float(epistemic_truth), 0.0, 1.0)
        header[2] = float(anchor_distance)
        header[3:5] = np.array([int(timestamp)], dtype=np.int64).view(np.float32)
        header[5:6] = np.array([int(model_version)], dtype=np.uint32).view(np.float32)
        # header[6:16] stays zero — reserved
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
