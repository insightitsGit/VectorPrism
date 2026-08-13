"""
ingestion_adapter.py - Multi-Task Projection Adapter (PyTorch)

CHANGES FROM PRIOR DRAFT:
  - Disentangled head now returns (mu, logvar) and is a real VIBHead
    (see losses.py) instead of a bare nn.Linear with no bottleneck.
  - Hyperbolic squash kept as-is (this was already correct: tanh(norm)/norm
    saturates smoothly, no exploding gradient at the boundary).
  - Added `causal_matrix` as an explicit nn.Parameter owned by the adapter
    (previously implied by losses.py but not instantiated anywhere).
  - forward() now returns both the assembled 1024d tensor (for storage)
    AND the raw per-head tensors (for computing losses during training) —
    the prior version only returned the concatenated tensor, which is
    fine for inference but unusable for training since you need direct
    access to e.g. (mu, logvar) for the VIB KL term.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensor_contract import PSMTensorContract
from losses import VIBHead


class MultiTaskProjectionAdapter(nn.Module):
    def __init__(self, base_dim: int = 768, num_disentangled_classes: int = 32):
        super().__init__()
        self.base_dim = base_dim

        self.head_dense = nn.Linear(base_dim, PSMTensorContract.DENSE_CORE.length)
        self.head_relational = nn.Linear(base_dim, PSMTensorContract.RELATIONAL.length)
        self.head_disentangled = VIBHead(
            in_dim=base_dim,
            latent_dim=PSMTensorContract.DISENTANGLED.length,
            num_classes=num_disentangled_classes,
        )
        self.head_hyperbolic = nn.Linear(base_dim, PSMTensorContract.HYPERBOLIC.length)
        self.head_identity = nn.Linear(base_dim, PSMTensorContract.IDENTITY.length)
        self.head_causal = nn.Linear(base_dim, PSMTensorContract.CAUSAL_TIME.length)

        # Owned here (not floating loose in losses.py) so it's part of the
        # model's state_dict and gets saved/loaded/optimized with everything else.
        self.causal_matrix = nn.Parameter(
            torch.randn(PSMTensorContract.CAUSAL_TIME.length, PSMTensorContract.CAUSAL_TIME.length) * 0.01
        )

        # Fixed identity anchor v0: NOT a learned parameter (see losses.py
        # center_loss docstring — must stay frozen). Registered as a buffer
        # so it moves with .to(device) and saves in state_dict, but receives
        # no gradient. Call `set_identity_anchor()` once after computing it
        # from a real in-domain reference corpus.
        self.register_buffer(
            "identity_anchor_v0", torch.zeros(PSMTensorContract.IDENTITY.length)
        )

    @torch.no_grad()
    def set_identity_anchor(self, v0: torch.Tensor) -> None:
        assert v0.shape == self.identity_anchor_v0.shape
        self.identity_anchor_v0.copy_(v0)

    def _hyperbolic_squash(self, hyp_raw: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(hyp_raw, p=2, dim=-1, keepdim=True)
        return hyp_raw * (torch.tanh(norm) / (norm + 1e-7)) * 0.99

    def forward(
        self, base_embedding: torch.Tensor, header: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            base_embedding: (batch, base_dim) frozen encoder output.
            header: (batch, 16) precomputed header block — the header is
                NOT learned from base_embedding (bitmask/timestamp are
                metadata, not a function of text semantics; epistemic
                truth needs its own labeled classifier — see
                GAP_RESOLUTION.md item on the truth score). Pass it in
                from the ingestion pipeline via tensor_contract.pack_header.
        Returns:
            tensor_1024d: (batch, 1024) assembled buffer for storage.
            raw: dict of per-head raw outputs for computing training losses.
        """
        dense = F.normalize(self.head_dense(base_embedding), p=2, dim=-1)
        relational = self.head_relational(base_embedding)

        z_dis, mu_dis, logvar_dis, _ = self.head_disentangled(base_embedding)

        hyp_raw = self.head_hyperbolic(base_embedding)
        hyperbolic = self._hyperbolic_squash(hyp_raw)

        identity = self.head_identity(base_embedding)
        causal = self.head_causal(base_embedding)

        tensor_1024d = torch.cat(
            [header, dense, relational, z_dis, hyperbolic, identity, causal], dim=-1
        )

        raw = {
            "dense": dense,
            "relational": relational,
            "disentangled_z": z_dis,
            "disentangled_mu": mu_dis,
            "disentangled_logvar": logvar_dis,
            "hyperbolic": hyperbolic,
            "identity": identity,
            "causal": causal,
        }
        return tensor_1024d, raw


# Product-facing alias (layout/pillars unchanged)
VectorPrismProjectionAdapter = MultiTaskProjectionAdapter
