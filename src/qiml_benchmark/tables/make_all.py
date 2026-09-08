"""Regenerate all required CSV and Markdown tables from saved results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()
OUT = ROOT / "results/tables"


def emit(number: int, name: str, frame: pd.DataFrame) -> None:
    stem = OUT / f"table_{number:02d}_{name}"
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    display = frame.copy()
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    def clean(value: object) -> str: return str(value).replace("|", "\\|").replace("\n", " ")
    header = "| " + " | ".join(map(clean, display.columns)) + " |"
    rule = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = ["| " + " | ".join(clean(x) for x in row) + " |" for row in display.itertuples(index=False, name=None)]
    stem.with_suffix(".md").write_text(f"# Table {number}: {name.replace('_', ' ').title()}\n\n" + "\n".join([header, rule, *body]) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Tables 1--4 were produced at their freeze phases; regenerate Markdown for consistency.
    for number, name in [(1, "dataset_splits"), (2, "feature_interface"), (3, "method_registry"), (4, "validation_selection")]:
        path = OUT / f"table_{number:02d}_{name}.csv"; emit(number, name, pd.read_csv(path, keep_default_na=False))
    intervals = pd.read_csv(ROOT / "results/statistics/bootstrap_intervals.csv")
    exact = pd.read_csv(ROOT / "results/aggregates/exact_performance.csv")
    primary_ci = intervals[(intervals.setting == "exact") & (intervals.analysis == "primary")][["dataset", "method", "ci_95_lower", "ci_95_upper"]]
    exact_table = exact.merge(primary_ci, on=["dataset", "method"], how="left")
    exact_table["primary_metric"] = np.where(exact_table.dataset == "breastmnist", "AUROC", "Accuracy")
    exact_table["primary_metric_mean"] = np.where(exact_table.dataset == "breastmnist", exact_table.metric_auroc_mean, exact_table.metric_accuracy_mean)
    exact_table["primary_metric_sd"] = np.where(exact_table.dataset == "breastmnist", exact_table.metric_auroc_sd, exact_table.metric_accuracy_sd)
    exact_table["primary_metric_95%_CI"] = exact_table.apply(lambda r: f"[{r.ci_95_lower:.6f}, {r.ci_95_upper:.6f}]", axis=1)
    exact_table["reference_group"] = np.where(exact_table.method.isin(["C-ResNet", "C-CNN"]), "full-image reference", "matched PCA8 input")
    emit(5, "exact_performance", exact_table[["dataset", "training_scale", "method", "reference_group", "primary_metric", "primary_metric_mean", "primary_metric_sd", "primary_metric_95%_CI", "ci_95_lower", "ci_95_upper", "metric_macro_f1_mean", "metric_auprc_mean", "metric_nll_mean", "metric_brier_mean", "metric_ece_15_mean", "metric_aurc_mean", "status"]])

    finite = pd.read_csv(ROOT / "results/aggregates/finite_shot_performance.csv")
    fci = intervals[intervals.setting == "finite_shot"][["dataset", "method", "shot_budget", "ci_95_lower", "ci_95_upper"]].copy(); fci.shot_budget = pd.to_numeric(fci.shot_budget)
    finite = finite.merge(fci, left_on=["dataset", "method", "nominal_budget"], right_on=["dataset", "method", "shot_budget"], how="left")
    finite["primary_metric_95%_CI"] = finite.apply(lambda r: f"[{r.ci_95_lower:.6f}, {r.ci_95_upper:.6f}]", axis=1)
    emit(6, "finite_shot_performance", finite.drop(columns=["shot_budget", "ci_95_lower", "ci_95_upper"]))
    emit(7, "uncertainty", pd.read_csv(ROOT / "results/aggregates/uncertainty_metrics.csv"))
    emit(8, "primary_significance", pd.read_csv(ROOT / "results/statistics/primary_tests.csv"))
    emit(9, "uncertainty_significance", pd.read_csv(ROOT / "results/statistics/uncertainty_tests.csv"))

    registry = pd.read_csv(OUT / "table_03_method_registry.csv")
    resources = pd.read_csv(ROOT / "results/resources/finite_shot_accounting.csv").merge(registry[["method", "qubits", "quantum_trainable_parameters"]], on="method", how="left")
    resources["training_scale"] = resources.dataset.map({"mnist":"D_5k", "fashion_mnist":"D_5k", "cifar10":"D_5k", "breastmnist":"D_100pct"})
    structures = {"QK-IQP":(8,32,16,32,0), "QF-Product":(4,32,0,32,0), "QF-Ring":(8,32,16,32,0), "VQC-DR":(14,80,16,80,48), "Yomo-Matched":(14,80,16,80,48)}
    def structure(row: pd.Series) -> tuple[int,int,int,int,int]:
        if row.method in structures: return structures[row.method]
        selection = json.loads((ROOT / "results/raw/pdr_selections" / f"{row.dataset}_{row.training_scale}.json").read_text()); sequence = selection["sequence"]; rotations = sum(x in {"RX","RY","RZ"} for x in sequence); rings = len(sequence)-rotations
        return rotations + 2*rings, 8*rotations, 8*rings, 8*rotations, 0
    values = resources.apply(structure, axis=1, result_type="expand"); values.columns = ["depth", "one_qubit_gates", "two_qubit_gates", "non_Clifford_gates", "quantum_parameters"]; resources = pd.concat([resources, values], axis=1)
    def observables(row: pd.Series) -> int:
        fixed = {"QK-IQP":1,"QF-Product":48,"QF-Ring":48,"VQC-DR":8,"Yomo-Matched":256}
        if row.method in fixed: return fixed[row.method]
        return len(json.loads((ROOT / "results/raw/pdr_selections" / f"{row.dataset}_{row.training_scale}.json").read_text())["paulis"])
    resources["observables"] = resources.apply(observables, axis=1)
    resources["total_parameters"] = resources.apply(lambda r: (138 if r.dataset != "breastmnist" else 66) if r.method=="VQC-DR" else 48 if r.method=="Yomo-Matched" else (490 if r.dataset != "breastmnist" else 49) if r.method in {"QF-Product","QF-Ring"} else (10*int(r.observables)+10 if r.dataset != "breastmnist" else int(r.observables)+1) if r.method=="PdrQC-Matched" else "not_applicable_kernel", axis=1)
    resources["status"] = "COMPLETED"; resources["circuit_executions"] = resources.offline_shot_draws; resources["statevector_evaluations"] = "unknown_not_instrumented"
    for column in ["train_time_median", "train_time_IQR", "inference_time_median", "inference_time_IQR", "peak_memory"]: resources[column] = "unknown_not_instrumented"
    emit(10, "resources", resources)
    validation = pd.read_csv(OUT / "table_04_validation_selection.csv")
    emit(11, "selected_hyperparameters", validation[["dataset", "training_scale", "method", "status", "config", "model_seeds", "temperatures"]])

    scaling = pd.read_csv(ROOT / "results/aggregates/data_scaling_performance.csv")
    sci = intervals[intervals.setting == "data_scaling"][["dataset", "method", "shot_budget", "ci_95_lower", "ci_95_upper"]].rename(columns={"shot_budget": "training_scale"})
    scaling = scaling.merge(sci, on=["dataset", "method", "training_scale"], how="left"); scaling["rank"] = scaling.groupby(["dataset", "training_scale"]).primary_metric_mean.rank(ascending=False, method="average")
    scaling["primary_metric_95%_CI"] = scaling.apply(lambda r: f"[{r.ci_95_lower:.6f}, {r.ci_95_upper:.6f}]", axis=1)
    emit(12, "data_scaling", scaling)
    rank = pd.read_csv(ROOT / "results/statistics/data_scaling_intervals.csv")
    rank["methods_excluded"] = "see scalability boundaries"; rank["reason_for_exclusion"] = "not completed at both scales"
    emit(13, "ranking_stability", rank)
    bounds = pd.read_csv(ROOT / "results/aggregates/data_scaling_boundaries.csv")
    largest = scaling.sort_values("training_scale").groupby(["dataset", "method"]).training_scale.last().rename("largest_completed_training_size").reset_index()
    bounds = bounds.merge(largest, on=["dataset", "method"], how="left").rename(columns={"training_scale": "requested_training_size", "status": "boundary_status"})
    bounds["estimated_wall_time"] = "not_executed"; bounds["estimated_memory"] = "not_executed"; bounds["estimated_circuits"] = "not_executed"; bounds["estimated_kernel_entries"] = "quadratic_in_training_size"; bounds["reason"] = bounds.boundary_status
    emit(14, "scalability_boundaries", bounds)
    events = []
    for path in sorted((ROOT / "logs/failures").glob("*.md")): events.append({"event": path.stem, "category": "failure_or_deviation", "source": str(path.relative_to(ROOT)), "status": "RECORDED"})
    for row in bounds.itertuples(): events.append({"event": f"{row.dataset}:{row.method}:{row.requested_training_size}", "category": "scaling_boundary", "source": "results/aggregates/data_scaling_boundaries.csv", "status": row.boundary_status})
    test_log = (ROOT / "TEST_ACCESS_LOG.md").read_text().splitlines()
    events.append({"event": "first_test_access", "category": "test_access", "source": "TEST_ACCESS_LOG.md", "status": next((x.strip() for x in test_log if "2026-" in x), "RECORDED")})
    emit(15, "failures_integrity", pd.DataFrame(events))
    print({"tables": 15, "csv": len(list(OUT.glob('table_*.csv'))), "markdown": len(list(OUT.glob('table_*.md')))})


if __name__ == "__main__":
    main()
