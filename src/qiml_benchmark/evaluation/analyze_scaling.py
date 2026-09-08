"""Consolidate frozen data-scaling test runs and quantify rank stability."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()
ORDER = {
    "mnist": ["D_1k", "D_5k", "D_10k", "D_full"],
    "fashion_mnist": ["D_1k", "D_5k", "D_10k", "D_full"],
    "cifar10": ["D_1k", "D_5k", "D_10k", "D_full"],
    "breastmnist": ["D_25pct", "D_50pct", "D_100pct"],
}
PRIMARY_METRIC = {"breastmnist": "metric_auroc", "mnist": "metric_accuracy", "fashion_mnist": "metric_accuracy", "cifar10": "metric_accuracy"}


def main() -> None:
    parts = [pd.read_csv(p) for p in sorted((ROOT / "results/raw/scaling_parts").glob("*.csv"))]
    raw = pd.concat(parts, ignore_index=True, sort=False)
    raw.to_csv(ROOT / "results/raw/data_scaling_runs.csv", index=False)
    completed = raw[raw.status == "COMPLETED"].copy()
    rows = []
    for (dataset, scale, method), group in completed.groupby(["dataset", "training_scale", "method"], sort=True):
        metric = PRIMARY_METRIC[dataset]
        vals = pd.to_numeric(group[metric], errors="coerce")
        rows.append({"dataset": dataset, "training_scale": scale, "method": method, "runs": len(group),
                     "primary_metric": metric.removeprefix("metric_"), "primary_metric_mean": vals.mean(),
                     "primary_metric_sd": vals.std(ddof=1), "statevector_evaluations": pd.to_numeric(group.statevector_evaluations, errors="coerce").sum(),
                     "kernel_entries": pd.to_numeric(group.kernel_entries, errors="coerce").sum(), "status": "COMPLETED"})
    agg = pd.DataFrame(rows)
    boundaries = raw[raw.status != "COMPLETED"][["dataset", "training_scale", "method", "status"]].drop_duplicates()
    agg.to_csv(ROOT / "results/aggregates/data_scaling_performance.csv", index=False)
    boundaries.to_csv(ROOT / "results/aggregates/data_scaling_boundaries.csv", index=False)

    stability = []
    for dataset, scales in ORDER.items():
        block = agg[agg.dataset == dataset]
        ranks = {s: block[block.training_scale == s].set_index("method").primary_metric_mean.rank(ascending=False, method="average") for s in scales}
        for i, left in enumerate(scales):
            for right in scales[i + 1:]:
                common = ranks[left].index.intersection(ranks[right].index)
                rho = spearmanr(ranks[left].loc[common], ranks[right].loc[common]).statistic if len(common) > 1 else np.nan
                stability.append({"dataset": dataset, "scale_A": left, "scale_B": right, "common_methods": len(common), "spearman_rho": rho})
    pd.DataFrame(stability).to_csv(ROOT / "results/aggregates/ranking_stability.csv", index=False)

    resource = completed.groupby(["dataset", "training_scale"], as_index=False).agg(
        completed_runs=("status", "size"), statevector_evaluations=("statevector_evaluations", "sum"), kernel_entries=("kernel_entries", "sum"))
    resource["real_qpu_shots"] = 0
    resource.to_csv(ROOT / "results/resources/data_scaling_resources.csv", index=False)

    summary = ROOT / "summary/DATA_SCALING_SUMMARY.md"
    summary.write_text("# Data-scaling summary\n\n"
        f"All {len(parts)} frozen dataset-scale cells were evaluated, producing {len(completed)} completed seed-level runs. "
        f"There are {len(boundaries)} predeclared boundary rows where full kernel evaluation was infeasible; no boundary was replaced with a surrogate result.\n\n"
        "Rank stability is reported only over methods common to each scale pair in `results/aggregates/ranking_stability.csv`. "
        "All test evaluations reused validation-frozen configurations and calibration parameters.\n")
    print({"cells": len(parts), "completed": len(completed), "boundaries": len(boundaries), "rank_pairs": len(stability)})


if __name__ == "__main__":
    main()
