"""
losses.py - Real, runnable loss functions for each PSM channel head.

IMPORTANT HONESTY NOTE (read before using):
Correct loss math does not equal a trained model. Every function below is
mathematically correct and unit-tested (see test_psm.py::TestLosses), but each one
still needs REAL LABELED DATA to produce a meaningful embedding space:

  - dense           -> (anchor, positive, [negatives]) text pairs
  - relational       -> (subject, relation, object) KG-style triples
  - disentangled     -> (input, label_you_want_predictable_from_Z) pairs
  - hyperbolic       -> (parent, child) taxonomy/hierarchy edges
  - identity/anchor  -> (in-domain, out-of-domain) examples
  - causal           -> (earlier_event, later_event) ordered pairs

No code fix can substitute for these datasets. If you don't have one of
these datasets for a given channel, the honest move is to drop that head
(set its intent-weight to 0 permanently) rather than ship an untrained
subspace that LOOKS structured but isn't.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Dense Semantic Core -> InfoNCE (Multiple Negatives Ranking Loss)
# ---------------------------------------------------------------------------
def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """
    anchor, positive: (batch, d) L2-normalized embeddings, index-aligned
    (anchor[i] should be close to positive[i], far from positive[j != i]).
    Uses in-batch negatives -> needs batch_size >= ~16 to be useful.
    """
    anchor = F.normalize(anchor, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    logits = anchor @ positive.T / temperature          # (batch, batch)
    labels = torch.arange(anchor.shape[0], device=anchor.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# 2. Relational Group Algebra -> TransE margin loss
# ---------------------------------------------------------------------------
def transe_margin_loss(
    subject: torch.Tensor,
    relation: torch.Tensor,
    obj: torch.Tensor,
    neg_obj: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    subject, relation, obj, neg_obj: (batch, d)
    Trains so that subject + relation ~= object (L2), and further from a
    corrupted/negative object than the true one by `margin`.

    DATA NOTE: `subject`/`obj` must be embeddings of two DIFFERENT entities
    (e.g. two different document chunks or KG entities), and `relation`
    must be a learned embedding from a small closed relation vocabulary
    (e.g. "causes", "is_a", "part_of") — NOT a per-passage projection of
    arbitrary free text. A single passage's relational-head output is not
    automatically a valid (S, R, O) triple; that structure only exists if
    you construct triples explicitly (e.g. via a relation-extraction
    pipeline or an existing KG) and train on those triples.
    """
    pos_dist = torch.norm(subject + relation - obj, p=2, dim=-1)
    neg_dist = torch.norm(subject + relation - neg_obj, p=2, dim=-1)
    return F.relu(margin + pos_dist - neg_dist).mean()


# ---------------------------------------------------------------------------
# 3. Disentangled Latent -> Variational Information Bottleneck (real, tractable form)
# ---------------------------------------------------------------------------
class VIBHead(nn.Module):
    """
    A true Information Bottleneck needs I(X;Z) and I(Y;Z), which are not
    directly computable. The standard tractable approximation (Alemi et al.,
    "Deep Variational Information Bottleneck") is used here: Z is Gaussian
    (mu, logvar), reparameterized, and the loss is:
        L = E[-log q(y|z)]  +  beta * KL(q(z|x) || N(0, I))
    which upper-bounds the true IB objective. This REQUIRES a downstream
    label y (whatever you want the disentangled slice to predict, e.g. a
    topic/category tag) — without it there is nothing to disentangle FROM.
    """
    def __init__(self, in_dim: int, latent_dim: int, num_classes: int):
        super().__init__()
        self.mu_head = nn.Linear(in_dim, latent_dim)
        self.logvar_head = nn.Linear(in_dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor):
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        logits = self.classifier(z)
        return z, mu, logvar, logits


def vib_loss(logits: torch.Tensor, y: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1e-3) -> torch.Tensor:
    ce = F.cross_entropy(logits, y)
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    return ce + beta * kl


# ---------------------------------------------------------------------------
# 4. Hyperbolic Taxonomy -> Poincare distance + negative sampling
# ---------------------------------------------------------------------------
def poincare_distance(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """u, v: (..., d), both with norm < 1 (enforce via the tanh squash in the adapter)."""
    u_norm_sq = torch.clamp(torch.sum(u * u, dim=-1), max=1 - eps)
    v_norm_sq = torch.clamp(torch.sum(v * v, dim=-1), max=1 - eps)
    sq_dist = torch.sum((u - v) ** 2, dim=-1)
    gamma = 1 + 2 * sq_dist / ((1 - u_norm_sq) * (1 - v_norm_sq) + eps)
    # Clamp to exactly 1.0, not 1+eps: acosh is well-defined at 1 (=0), and
    # flooring above 1 was previously inflating genuinely-identical points
    # to a spurious nonzero distance (acosh(1+1e-5) ~= 0.0045). The only
    # thing that needs guarding is going BELOW 1 from floating-point noise.
    gamma = torch.clamp(gamma, min=1.0)
    return torch.acosh(gamma)


def poincare_negative_sampling_loss(
    parent: torch.Tensor, child: torch.Tensor, negatives: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """
    parent, child: (batch, d) — true (parent, child) taxonomy edges, norm < 1.
    negatives: (batch, k, d) — k sampled non-related nodes per anchor.
    DATA NOTE: requires an explicit hierarchy/taxonomy edge list (e.g. a
    category tree, WordNet-style is-a edges). Free text alone has no
    hierarchy signal until you extract or supply one.
    """
    pos_dist = poincare_distance(parent, child)                              # (batch,)
    neg_dist = poincare_distance(parent.unsqueeze(1), negatives)             # (batch, k)
    losses = F.relu(margin + pos_dist.unsqueeze(1) - neg_dist)               # (batch, k)
    return losses.mean()


# ---------------------------------------------------------------------------
# 5. Identity Anchor -> Center loss around a FIXED, precomputed anchor v0
# ---------------------------------------------------------------------------
def center_loss(v: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """
    v: (batch, d) in-domain embeddings. v0: (d,) FIXED anchor point,
    precomputed as the mean embedding over a large in-domain reference
    corpus (compute once, freeze — do not learn v0 jointly with v, or the
    anchor will chase the data and the "out-of-domain distance" becomes
    meaningless as a security signal).
    """
    return 0.5 * torch.mean(torch.sum((v - v0) ** 2, dim=-1))


def anchor_distance_score(v: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """Inference-time OOD/injection-risk score: L2 distance to the fixed anchor.
    NOTE (unchanged from review): embedding-distance-to-anchor is a weak,
    unvalidated prompt-injection signal on its own. Treat this as one soft
    feature into a real safety pipeline (e.g. a classifier trained on
    known injection examples), not a standalone security gate."""
    return torch.norm(v - v0, p=2, dim=-1)


# ---------------------------------------------------------------------------
# 6. Time ODE & Causality -> asymmetric pairwise ranking loss
# ---------------------------------------------------------------------------
def causal_asymmetric_loss(
    earlier: torch.Tensor, later: torch.Tensor, M: torch.Tensor, margin: float = 0.5
) -> torch.Tensor:
    """
    earlier, later: (batch, d) embeddings of a *known-ordered* event pair
    (earlier happened before later). M: learned (d, d) asymmetric matrix
    (a free nn.Parameter, NOT required to be antisymmetric by construction
    unless you explicitly parameterize it that way).
    Score(u -> v) = u^T M v. Trained so forward score > backward score.
    DATA NOTE: requires labeled temporal/causal order pairs. Without them
    this is just two random linear maps with no directional meaning.
    """
    fwd = torch.einsum("bi,ij,bj->b", earlier, M, later)
    bwd = torch.einsum("bi,ij,bj->b", later, M, earlier)
    return F.relu(margin - (fwd - bwd)).mean()
