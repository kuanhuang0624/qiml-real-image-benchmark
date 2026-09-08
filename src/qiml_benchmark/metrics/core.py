"""Predictive, calibration, and selective-risk metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score)


def ece_equal_mass(labels: np.ndarray, probs: np.ndarray, bins: int = 15) -> tuple[float, list[dict[str, float]]]:
    confidence = probs.max(1); prediction = probs.argmax(1); correct = prediction == labels
    order = np.argsort(confidence); groups = np.array_split(order, bins); value = 0.0; rows = []
    for index, group in enumerate(groups):
        if not len(group): continue
        acc, conf = correct[group].mean(), confidence[group].mean()
        value += len(group) / len(labels) * abs(acc - conf)
        rows.append({"bin": index, "count": len(group), "accuracy": float(acc), "confidence": float(conf)})
    return float(value), rows


def risk_coverage(labels: np.ndarray, probs: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    uncertainty = 1 - probs.max(1); correct = probs.argmax(1) == labels; order = np.argsort(uncertainty)
    errors = (~correct[order]).astype(float); risk = np.cumsum(errors) / np.arange(1, len(errors) + 1); coverage = np.arange(1, len(errors) + 1) / len(errors)
    def at(value: float) -> float: return float(risk[max(0, int(np.ceil(value * len(risk))) - 1)])
    return {"aurc": float(np.trapezoid(risk, coverage)), "risk_100": at(1), "risk_90": at(.9), "risk_80": at(.8)}, np.column_stack([coverage, risk])


def classification_metrics(labels: np.ndarray, probs: np.ndarray, binary_threshold: float | None = None) -> dict[str, float]:
    labels = np.asarray(labels, int); probs = np.clip(np.asarray(probs, float), 1e-12, 1 - 1e-12)
    prediction = probs.argmax(1) if binary_threshold is None else (probs[:, 1] >= binary_threshold).astype(int)
    onehot = np.eye(probs.shape[1])[labels]; ece, _ = ece_equal_mass(labels, probs); selective, _ = risk_coverage(labels, probs)
    uncertainty = 1 - probs.max(1); errors = (prediction != labels).astype(int)
    error_auc = float(roc_auc_score(errors, uncertainty)) if len(np.unique(errors)) == 2 else float("nan")
    result = {
        "accuracy": float(accuracy_score(labels, prediction)), "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(labels, prediction, average="macro", zero_division=0)), "macro_recall": float(recall_score(labels, prediction, average="macro", zero_division=0)),
        "nll": float(log_loss(labels, probs, labels=np.arange(probs.shape[1]))), "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "ece_15": ece, "error_detection_auroc": error_auc, **selective,
    }
    if probs.shape[1] == 2:
        result.update(auroc=float(roc_auc_score(labels, probs[:, 1])), auprc=float(average_precision_score(labels, probs[:, 1])), balanced_accuracy=float(balanced_accuracy_score(labels, prediction)), sensitivity=float(recall_score(labels, prediction, pos_label=1)), specificity=float(recall_score(labels, prediction, pos_label=0)))
    else:
        result["macro_auroc"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
    return result
