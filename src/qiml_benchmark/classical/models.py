"""Matched classical representations and score-producing classifiers."""
from __future__ import annotations

import itertools
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def trig_features(x: np.ndarray, level: int) -> np.ndarray:
    if level not in (1, 2): raise ValueError(level)
    columns = [np.ones(len(x))]
    bases = [(np.cos(x[:, j]), np.sin(x[:, j])) for j in range(x.shape[1])]
    for j in range(x.shape[1]): columns.extend(bases[j])
    if level == 2:
        for j, k in itertools.combinations(range(x.shape[1]), 2):
            columns.extend(a * b for a in bases[j] for b in bases[k])
    return np.column_stack(columns)


def build(method: str, classes: int, seed: int, **params: float | int):
    if method == "C-LR": return LogisticRegression(C=float(params.get("C", 1)), max_iter=2000, random_state=seed)
    if method == "C-MLP":
        hidden = 7 if classes == 10 else 6
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(hidden,), alpha=float(params.get("alpha", 1e-3)), max_iter=int(params.get("epochs", 300)), random_state=seed, early_stopping=True))
    if method == "C-Poly": return SVC(kernel="poly", degree=int(params.get("degree", 2)), C=float(params.get("C", 1)), random_state=seed)
    if method == "C-RBF": return SVC(kernel="rbf", C=float(params.get("C", 1)), gamma=params.get("gamma", "scale"), random_state=seed)
    if method == "C-RFF":
        return make_pipeline(RBFSampler(gamma=float(params.get("gamma", .1)), n_components=int(params.get("dimension", 128)), random_state=seed), LogisticRegression(C=float(params.get("C", 1)), max_iter=2000, random_state=seed))
    if method == "C-Trig": return LogisticRegression(C=float(params.get("C", 1)), max_iter=2000, random_state=seed)
    if method == "C-ResNet": return LogisticRegression(C=float(params.get("C", 1)), max_iter=2000, random_state=seed)
    if method == "ridge": return RidgeClassifier(alpha=float(params.get("alpha", 1)))
    raise ValueError(method)


def score_matrix(model, x: np.ndarray) -> np.ndarray:
    score = model.decision_function(x)
    if score.ndim == 1: return score
    return np.asarray(score)
