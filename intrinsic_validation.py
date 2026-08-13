"""
intrinsic_validation.py - Per-channel structural validation.

WHY THIS EXISTS: eval_harness.py tells you if the whole system retrieves
well. It does NOT tell you which of the 6 channels is responsible for a
good or bad score, or whether a channel has learned the structure it was
trained for at all. These functions test each channel in isolation
against its own held-out labeled data, BEFORE you wire it into the full
retrieval engine. Run these first; if a channel fails its own intrinsic
test, fixing the retrieval-level ablation won't help — go back to that
channel's training data/loss.

Each function requires real held-out labels for that specific channel.
None of these can be computed from unlabeled data — that's not a
limitation of this code, it's what "validation" means.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Dense: standard IR metrics (recall@k already in eval_harness; add nDCG)
# ---------------------------------------------------------------------------
def ndcg_at_k(ranked_relevance: List[float], k: int) -> float:
    """ranked_relevance: graded relevance scores (0-3 typical) in retrieved order."""
    ranked_relevance = ranked_relevance[:k]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ranked_relevance))
    ideal = sorted(ranked_relevance, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Relational: link-prediction Hits@k and MRR (standard KG embedding eval)
# ---------------------------------------------------------------------------
def link_prediction_eval(
    scores_true: np.ndarray, scores_corrupted: np.ndarray, k_values: List[int] = [1, 3, 10]
) -> dict:
    """
    scores_true: (N,) score (LOWER = more plausible, since TransE uses distance)
        for each true (s, r, o) triple.
    scores_corrupted: (N, num_corruptions) scores for corrupted objects per triple.
    Standard filtered-setting KG eval: rank the true score against corruptions.
    """
    ranks = []
    for true_s, corrupt_s in zip(scores_true, scores_corrupted):
        rank = 1 + int(np.sum(corrupt_s < true_s))  # count corruptions ranked better
        ranks.append(rank)
    ranks = np.array(ranks)
    report = {"MRR": float(np.mean(1.0 / ranks)), "mean_rank": float(np.mean(ranks))}
    for k in k_values:
        report[f"Hits@{k}"] = float(np.mean(ranks <= k))
    return report


# ---------------------------------------------------------------------------
# Hyperbolic: tree distortion (does embedding distance preserve graph distance?)
# ---------------------------------------------------------------------------
def embedding_distortion(graph_distances: np.ndarray, embedding_distances: np.ndarray) -> dict:
    """
    graph_distances: (N,) true shortest-path distance in the taxonomy graph
        for N sampled node pairs.
    embedding_distances: (N,) Poincare distance for the same pairs.
    Standard metric from Nickel & Kiela (2017): average distortion
        (1/N) * sum( |d_emb/d_graph - 1| )  -- lower is better, 0 = perfect.
    Also reports Spearman correlation (rank-order preservation matters more
    than absolute scale for retrieval purposes).
    """
    from scipy.stats import spearmanr
    ratio = embedding_distances / np.maximum(graph_distances, 1e-9)
    distortion = float(np.mean(np.abs(ratio - np.mean(ratio))))
    rho, _ = spearmanr(graph_distances, embedding_distances)
    return {"mean_distortion": distortion, "spearman_rho": float(rho)}


# ---------------------------------------------------------------------------
# Disentangled: probe accuracy on intended label vs leakage on a nuisance label
# ---------------------------------------------------------------------------
def disentanglement_probe(
    z: np.ndarray, intended_label: np.ndarray, nuisance_label: np.ndarray
) -> dict:
    """
    z: (N, d) latent codes. intended_label: what Z SHOULD predict well.
    nuisance_label: something Z should NOT encode (e.g. writing style,
    document length bucket, source) if disentanglement is working.
    Trains two cheap linear probes and reports both accuracies.
    A well-disentangled Z: high intended_acc, near-chance nuisance_acc.
    A collapsed/entangled Z: both high, or both low.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    intended_acc = float(np.mean(cross_val_score(
        LogisticRegression(max_iter=1000), z, intended_label, cv=5
    )))
    nuisance_acc = float(np.mean(cross_val_score(
        LogisticRegression(max_iter=1000), z, nuisance_label, cv=5
    )))
    chance = 1.0 / len(np.unique(nuisance_label))
    return {
        "intended_label_accuracy": intended_acc,
        "nuisance_label_accuracy": nuisance_acc,
        "nuisance_chance_accuracy": chance,
        "disentanglement_gap": intended_acc - nuisance_acc,  # want this large
        "leakage_above_chance": nuisance_acc - chance,        # want this near 0
    }


# ---------------------------------------------------------------------------
# Identity Anchor: OOD detection AUROC
# ---------------------------------------------------------------------------
def ood_detection_auroc(in_domain_dist: np.ndarray, out_of_domain_dist: np.ndarray) -> float:
    """
    in_domain_dist: anchor distances for known in-domain examples.
    out_of_domain_dist: anchor distances for known OOD/adversarial examples.
    Uses distance as the OOD score directly (higher = more OOD).
    AUROC > 0.5 means the anchor distance separates the two classes at all;
    treat AUROC < 0.7 as "not reliable enough to gate on" per the earlier
    caution about this being a weak standalone signal.
    """
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.zeros_like(in_domain_dist), np.ones_like(out_of_domain_dist)])
    scores = np.concatenate([in_domain_dist, out_of_domain_dist])
    return float(roc_auc_score(y, scores))


# ---------------------------------------------------------------------------
# Causal: pairwise temporal order accuracy
# ---------------------------------------------------------------------------
def causal_order_accuracy(fwd_scores: np.ndarray, bwd_scores: np.ndarray) -> float:
    """fwd_scores, bwd_scores: (N,) causal_matrix scores for (earlier->later)
    and (later->earlier) on N held-out KNOWN-ORDER pairs.
    Fraction where the model correctly scores forward > backward."""
    return float(np.mean(fwd_scores > bwd_scores))


# ---------------------------------------------------------------------------
# Epistemic Truth Score: calibration (does P(truth)=0.7 mean ~70% correct?)
# ---------------------------------------------------------------------------
def expected_calibration_error(predicted_prob: np.ndarray, is_actually_true: np.ndarray, n_bins: int = 10) -> dict:
    """
    Standard ECE (Guo et al., 2017). predicted_prob: model's P(truth) output.
    is_actually_true: (N,) binary ground truth labels.
    Returns ECE (lower is better, 0 = perfectly calibrated) plus per-bin
    data for a reliability diagram. DO NOT hard-filter on this score in
    production until ECE has been measured and is acceptably low — an
    uncalibrated filter silently drops correct results (see GAP_RESOLUTION.md #11).
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_report = []
    n = len(predicted_prob)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (predicted_prob >= lo) & (predicted_prob < hi)
        if mask.sum() == 0:
            continue
        bin_conf = predicted_prob[mask].mean()
        bin_acc = is_actually_true[mask].mean()
        weight = mask.sum() / n
        ece += weight * abs(bin_acc - bin_conf)
        bin_report.append({"range": (float(lo), float(hi)), "confidence": float(bin_conf),
                            "accuracy": float(bin_acc), "n": int(mask.sum())})
    return {"ECE": float(ece), "bins": bin_report}
