"""Recall at a fixed false-positive rate, with paired bootstrap intervals (paper §3.3)."""

import math

import numpy as np
from sklearn.metrics import roc_auc_score


def threshold_at_fpr(negatives: np.ndarray, fpr: float = 0.01) -> float:
    """The (k+1)-th largest negative score, k = floor(fpr * n). With detection = score > threshold, at most k negatives
    are flagged, so the realized FPR is <= fpr. With fewer than 100 negatives this is the largest negative score."""
    k = math.floor(fpr * len(negatives) + 1e-9)
    return float(np.sort(negatives)[::-1][k])


def recall(positives: np.ndarray, threshold: float) -> float:
    return float(np.mean(positives > threshold))


def auroc(positives: np.ndarray, negatives: np.ndarray) -> float:
    labels = np.r_[np.ones(len(positives)), np.zeros(len(negatives))]
    return float(roc_auc_score(labels, np.r_[positives, negatives]))


def summarize(negatives: np.ndarray, positives: dict[str, np.ndarray], fpr: float = 0.01, n_boot: int = 1000,
              seed: int = 0) -> dict:
    """Per condition (clean = "normal", or a trigger word): recall at the clean negatives' threshold, AUROC, and paired
    bootstrap 95% intervals for recall and for the drop from the clean recall. Positives are resampled by example, the
    same indices for every condition; negatives separately; the threshold is refit in every resample."""
    negatives = np.asarray(negatives, dtype=float)
    positives = {k: np.asarray(v, dtype=float) for k, v in positives.items()}
    n_pos = len(positives["normal"])
    assert all(len(v) == n_pos for v in positives.values()), "conditions must score the same positives"
    threshold = threshold_at_fpr(negatives, fpr)
    rng = np.random.default_rng(seed)
    boot = {k: np.empty(n_boot) for k in positives}
    for i in range(n_boot):
        neg = negatives[rng.integers(0, len(negatives), len(negatives))]
        idx = rng.integers(0, n_pos, n_pos)
        t = threshold_at_fpr(neg, fpr)
        for k, v in positives.items():
            boot[k][i] = np.mean(v[idx] > t)
    out = {"threshold": threshold, "realized_fpr": float(np.mean(negatives > threshold)), "n_pos": n_pos,
           "n_neg": len(negatives), "conditions": {}}
    for k, v in positives.items():
        drop = boot["normal"] - boot[k]
        out["conditions"][k] = {
            "recall": recall(v, threshold),
            "recall_ci": [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))],
            "drop": recall(positives["normal"], threshold) - recall(v, threshold),
            "drop_ci": [float(np.percentile(drop, 2.5)), float(np.percentile(drop, 97.5))],
            "auroc": auroc(v, negatives),
            "mean_score": float(v.mean()),
        }
    return out
