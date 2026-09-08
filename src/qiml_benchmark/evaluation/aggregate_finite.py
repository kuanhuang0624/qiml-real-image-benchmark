from pathlib import Path
import glob
import pandas as pd

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()
raw = pd.concat([pd.read_csv(path) for path in glob.glob(str(ROOT / "results/raw/finite_parts/*.csv"))], ignore_index=True, sort=False); raw.to_csv(ROOT / "results/raw/finite_shot_runs.csv", index=False)
exact = pd.read_csv(ROOT / "results/aggregates/exact_performance.csv"); rows = []
for (dataset, method, definition, budget), group in raw.groupby(["dataset", "method", "budget_definition", "nominal_budget"]):
    primary_column = "metric_auroc" if dataset == "breastmnist" else "metric_accuracy"; values = group[primary_column].dropna(); exact_row = exact[(exact.dataset == dataset) & (exact.method == method)]; exact_value = exact_row[f"{primary_column}_mean"].iloc[0]
    rows.append({"dataset": dataset, "method": method, "budget_definition": definition, "nominal_budget": budget, "actual_state_preparations_per_image_mean": group.actual_state_preparations_per_image.mean(), "measurement_settings": int(group.measurement_settings.max()), "measurement_groups": int(group.measurement_groups.max()), "runs": len(group), "primary_metric_mean": values.mean(), "primary_metric_sd": values.std(ddof=1), "delta_versus_exact": values.mean() - exact_value, "Brier": group.metric_brier.mean(), "shot_uncertainty": group.groupby("model_seed")[primary_column].std(ddof=1).mean(), "status": "COMPLETED"})
aggregate = pd.DataFrame(rows); aggregate.to_csv(ROOT / "results/aggregates/finite_shot_performance.csv", index=False)
accounting = raw.groupby(["dataset", "method", "budget_definition", "nominal_budget"], as_index=False).agg(runs=("status", "size"), offline_shot_draws=("offline_shot_draws", "sum"), real_qpu_shots=("real_qpu_shots", "sum"), kernel_entries=("kernel_entries", "sum"), actual_state_preparations_per_image=("actual_state_preparations_per_image", "mean"), measurement_settings=("measurement_settings", "max"), measurement_groups=("measurement_groups", "max"))
accounting.to_csv(ROOT / "results/resources/finite_shot_accounting.csv", index=False)
print(f"FINITE_RUNS={len(raw)} FINITE_CONDITIONS={len(aggregate)}")
