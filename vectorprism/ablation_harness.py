"""
ablation_harness.py - System-level validation: which channels earn their complexity?

Runs the SAME labeled eval set through the retrieval engine multiple times,
each time fixing w_intent to isolate a subset of channels, and reports
recall/MRR per configuration. This is what turns "6 channels beat 1" from
an assertion into a number.

Order of operations should be:
  1. intrinsic_validation.py on each trained channel FIRST (does the
     channel have the structure it claims to?)
  2. THIS harness SECOND (does adding a structurally-valid channel to
     retrieval actually improve recall over dense-only?)
A channel can pass step 1 and still fail step 2 (structurally correct but
not useful for THIS retrieval task) — that's a legitimate, useful outcome,
not a bug. Cut channels that fail step 2 even if they pass step 1.
"""

from typing import List, Dict
import numpy as np

from vectorprism.retrieval_engine import PSMRetrievalEngine
from vectorprism.eval_harness import EvalExample, evaluate

CHANNEL_NAMES = ["dense", "relational", "disentangled", "hyperbolic", "causal"]


def _fixed_weight_classifier(weight_vector: np.ndarray):
    """Returns a classify() that ignores query text and always returns the
    given fixed weight vector, so we test pure channel contribution
    without intent-routing noise confounding the comparison."""
    def classify(query_text: str):
        filters = {
            "min_truth": 0.0,
            "max_anchor_dist": 1.0,
            "model_version": None,
        }  # neutral filters for ablation
        return weight_vector.astype(np.float32), filters
    return classify


def run_ablation(
    engine: PSMRetrievalEngine, eval_set: List[EvalExample], k_values: List[int] = [1, 5, 10]
) -> Dict[str, Dict[str, float]]:
    """Returns {config_name: {metric: value}} for dense_only (baseline),
    dense + each single channel, and all channels equally weighted."""
    original_classify = engine.classifier.classify
    results = {}

    configs = {"dense_only": np.array([1.0, 0, 0, 0, 0])}
    for i, name in enumerate(CHANNEL_NAMES[1:], start=1):
        w = np.zeros(5)
        w[0] = 0.7
        w[i] = 0.3
        configs[f"dense+{name}"] = w
    configs["all_channels_equal"] = np.ones(5) / 5

    try:
        for config_name, weights in configs.items():
            engine.classifier.classify = _fixed_weight_classifier(weights)
            results[config_name] = evaluate(engine, eval_set, k_values=k_values)
    finally:
        engine.classifier.classify = original_classify  # always restore, even on error

    return results


def print_ablation_report(results: Dict[str, Dict[str, float]], baseline: str = "dense_only") -> None:
    """Prints each config's metrics and its delta vs the dense-only baseline,
    so you can see at a glance which channels are pulling their weight."""
    if baseline not in results:
        raise ValueError(f"baseline '{baseline}' not in results")
    base = results[baseline]
    metric_keys = [k for k in base if k != "n_eval_examples"]

    header = f"{'config':<22}" + "".join(f"{m:>14}" for m in metric_keys)
    print(header)
    print("-" * len(header))
    for config_name, metrics in results.items():
        row = f"{config_name:<22}"
        for m in metric_keys:
            val = metrics[m]
            delta = val - base[m]
            cell = f"{val:.3f}" if config_name == baseline else f"{val:.3f}({delta:+.3f})"
            row += f"{cell:>14}"
        print(row)
