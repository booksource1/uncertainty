from __future__ import annotations

import numpy as np


def _stable_argsort(x: np.ndarray) -> np.ndarray:
    # mergesort is stable; helps deterministic tie handling.
    return np.argsort(x, kind="mergesort")


def roc_curve_binary(y_true: np.ndarray, y_score: np.ndarray):
    """
    Compute ROC curve points for binary classification without sklearn.
    Returns (fpr, tpr, thresholds) where thresholds are descending unique scores.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1 or y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must be 1D arrays with same length")

    pos = (y_true == 1)
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([np.nan]), np.array([np.nan]), np.array([np.nan])

    order = _stable_argsort(-y_score)  # descending
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    # Cumulative counts at each index
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    # Only keep points where score changes (thresholds)
    distinct = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    tp = tp[distinct]
    fp = fp[distinct]
    thresholds = s_sorted[distinct]

    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)

    # Add (0,0) point with threshold +inf (sklearn-style)
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[np.inf, thresholds]
    return fpr, tpr, thresholds


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC via rank statistic (Mann–Whitney U), tie-aware."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = _stable_argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # average ranks for ties
    uniq, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for k, c in enumerate(counts):
            if c <= 1:
                continue
            idx = np.where(inv == k)[0]
            ranks[idx] = ranks[idx].mean()

    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Average precision (AUPR) without sklearn.
    Uses step-wise integration of precision over recall as threshold decreases.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = y_true == 1
    n_pos = int(pos.sum())
    if n_pos == 0:
        return float("nan")

    order = _stable_argsort(-y_score)  # descending score
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    # precision/recall at each distinct threshold
    distinct = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    tp_d = tp[distinct]
    fp_d = fp[distinct]

    precision = tp_d / np.maximum(tp_d + fp_d, 1)
    recall = tp_d / n_pos

    # AP is sum over recall increments * precision at that point
    recall_prev = np.r_[0.0, recall[:-1]]
    ap = np.sum((recall - recall_prev) * precision)
    return float(ap)


def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, tpr_target: float = 0.95) -> float:
    """FPR at the smallest threshold achieving TPR >= target (FPR95 when target=0.95)."""
    fpr, tpr, _ = roc_curve_binary(y_true, y_score)
    if np.isnan(fpr).any():
        return float("nan")
    idx = np.where(tpr >= tpr_target)[0]
    if idx.size == 0:
        return float("nan")
    return float(np.min(fpr[idx]))


