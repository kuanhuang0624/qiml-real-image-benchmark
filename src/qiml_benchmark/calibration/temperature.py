"""Validation-only scalar temperature scaling and binary threshold selection."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, softmax
from sklearn.metrics import balanced_accuracy_score


def probabilities(scores: np.ndarray, temperature: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if temperature <= 0: raise ValueError("temperature must be positive")
    if scores.ndim == 1:
        p1 = expit(scores / temperature); return np.column_stack([1 - p1, p1])
    return softmax(scores / temperature, axis=1)


def fit_temperature(scores: np.ndarray, labels: np.ndarray, *, split: str) -> float:
    if split != "validation": raise ValueError("temperature fitting is validation-only")
    labels = np.asarray(labels, dtype=int)
    def nll(log_t: float) -> float:
        p = np.clip(probabilities(scores, float(np.exp(log_t))), 1e-12, 1 - 1e-12)
        return float(-np.mean(np.log(p[np.arange(len(labels)), labels])))
    result = minimize_scalar(nll, bounds=(-5, 5), method="bounded")
    if not result.success: raise RuntimeError(result.message)
    return float(np.exp(result.x))


def fit_binary_threshold(probability_positive: np.ndarray, labels: np.ndarray, *, split: str) -> float:
    if split != "validation": raise ValueError("threshold fitting is validation-only")
    candidates = np.unique(np.r_[0.0, probability_positive, 1.0])
    scores = np.asarray([balanced_accuracy_score(labels, probability_positive >= threshold) for threshold in candidates])
    best = np.flatnonzero(scores == scores.max())
    return float(candidates[best[np.argmin(np.abs(candidates[best] - .5))]])
