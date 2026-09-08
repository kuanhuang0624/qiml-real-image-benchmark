"""Compute frozen exact and finite-shot uncertainty decompositions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from qiml_benchmark.metrics.core import ece_equal_mass

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()


def _prediction(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    records = []
    for row in group.itertuples():
        z = np.load(ROOT / row.prediction_file)
        records.append((str(row.model_seed), str(getattr(row, "measurement_seed", "exact")), z["calibrated_probability"], z["label"], float(z["threshold"])))
    labels = records[0][3]
    seeds = sorted({x[0] for x in records})
    model_means, shot_values = [], []
    for seed in seeds:
        stack = np.stack([x[2] for x in records if x[0] == seed])
        mean = stack.mean(0); model_means.append(mean)
        shot_values.append(np.mean(np.sum((stack - mean) ** 2, axis=2), axis=0))
    model_stack = np.stack(model_means); mean_prob = model_stack.mean(0)
    model_u = np.mean(np.sum((model_stack - mean_prob) ** 2, axis=2), axis=0)
    shot_u = np.mean(np.stack(shot_values), axis=0)
    thresholds = [x[4] for x in records if np.isfinite(x[4])]
    return labels, mean_prob, model_u, shot_u, np.asarray(thresholds)


def _analyze(group: pd.DataFrame, setting: str, dataset: str, method: str, budget: str) -> tuple[dict, list[dict], list[dict]]:
    labels, prob, model_u, shot_u, thresholds = _prediction(group)
    entropy = -np.sum(prob * np.log(np.clip(prob, 1e-12, 1)), axis=1)
    threshold = float(np.mean(thresholds)) if dataset == "breastmnist" and len(thresholds) else np.nan
    pred = (prob[:, 1] >= threshold).astype(int) if np.isfinite(threshold) else prob.argmax(1)
    errors = pred != labels
    order = np.argsort(entropy); risk = np.cumsum(errors[order]) / np.arange(1, len(labels) + 1); coverage = np.arange(1, len(labels) + 1) / len(labels)
    ece, bins = ece_equal_mass(labels, prob)
    onehot = np.eye(prob.shape[1])[labels]
    key = {"dataset": dataset, "method": method, "setting": setting, "shot_budget": budget}
    row = {**key, "nll": log_loss(labels, prob, labels=np.arange(prob.shape[1])),
           "brier": np.mean(np.sum((prob - onehot) ** 2, axis=1)), "ece_15": ece,
           "predictive_entropy": entropy.mean(), "model_uncertainty": model_u.mean(),
           "shot_uncertainty": shot_u.mean() if setting == "finite_shot" else np.nan,
           "error_detection_auroc": roc_auc_score(errors, entropy) if len(np.unique(errors)) == 2 else np.nan,
           "aurc": np.trapezoid(risk, coverage), "risk_100": risk[-1],
           "risk_90": risk[max(0, int(np.ceil(.9 * len(risk))) - 1)], "risk_80": risk[max(0, int(np.ceil(.8 * len(risk))) - 1)],
           "correct_sample_uncertainty": entropy[~errors].mean() if (~errors).any() else np.nan,
           "incorrect_sample_uncertainty": entropy[errors].mean() if errors.any() else np.nan,
           "runs": len(group), "uncertainty_score": "predictive_entropy"}
    bin_rows = [{**key, **x} for x in bins]
    # Store 101 evenly spaced coverage locations, including the terminal point.
    indexes = np.unique(np.r_[np.linspace(0, len(labels) - 1, 101).astype(int), len(labels) - 1])
    risk_rows = [{**key, "coverage": coverage[i], "risk": risk[i], "samples_retained": i + 1} for i in indexes]
    return row, bin_rows, risk_rows


def main() -> None:
    exact = pd.read_csv(ROOT / "results/raw/exact_test_runs.csv")
    finite = pd.read_csv(ROOT / "results/raw/finite_shot_runs.csv")
    rows, bins, risks = [], [], []
    for (dataset, method), group in exact[exact.status == "COMPLETED"].groupby(["dataset", "method"], sort=True):
        row, b, r = _analyze(group, "exact", dataset, method, "exact"); rows.append(row); bins.extend(b); risks.extend(r)
    for (dataset, method, budget), group in finite[finite.status == "COMPLETED"].groupby(["dataset", "method", "nominal_budget"], sort=True):
        row, b, r = _analyze(group, "finite_shot", dataset, method, str(int(budget))); rows.append(row); bins.extend(b); risks.extend(r)
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results/aggregates/uncertainty_metrics.csv", index=False)
    frame[["dataset", "method", "setting", "shot_budget", "nll", "brier", "ece_15"]].to_csv(ROOT / "results/aggregates/calibration_metrics.csv", index=False)
    frame[["dataset", "method", "setting", "shot_budget", "error_detection_auroc", "aurc", "risk_100", "risk_90", "risk_80"]].to_csv(ROOT / "results/aggregates/risk_coverage_metrics.csv", index=False)
    pd.DataFrame(bins).to_csv(ROOT / "results/aggregates/reliability_bins.csv", index=False)
    pd.DataFrame(risks).to_csv(ROOT / "results/aggregates/risk_coverage_curves.csv", index=False)
    (ROOT / "summary/UNCERTAINTY_SUMMARY.md").write_text(
        "# Uncertainty summary\n\n"
        f"Uncertainty and calibration were evaluated for {len(frame)} exact or finite-shot method conditions. "
        "Test-sampling uncertainty is handled separately by stratified image bootstrap. Model uncertainty is the mean squared probability-vector deviation across model seeds; quantum-measurement uncertainty is the conditional deviation across measurement seeds and is N/A for exact/classical inference. "
        "Predictive entropy is used for error detection and selective risk. Calibration uses validation-frozen temperatures and, for BreastMNIST decisions, validation-frozen thresholds.\n")
    print({"conditions": len(frame), "reliability_bins": len(bins), "risk_points": len(risks)})


if __name__ == "__main__":
    main()
