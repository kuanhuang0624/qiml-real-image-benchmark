from pathlib import Path
import glob
import pandas as pd

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()
parts = [pd.read_csv(path) for path in glob.glob(str(ROOT / "results/raw/exact_parts/*.csv"))]
raw = pd.concat(parts, ignore_index=True, sort=False); raw.to_csv(ROOT / "results/raw/exact_test_runs.csv", index=False)
metric_columns = [c for c in raw.columns if c.startswith("metric_")]
rows = []
for (dataset, scale, method), group in raw.groupby(["dataset", "training_scale", "method"]):
    row = {"dataset": dataset, "training_scale": scale, "method": method, "runs": len(group), "model_seeds": ";".join(map(str, sorted(group.model_seed.unique()))), "status": "COMPLETED"}
    for metric in metric_columns:
        values = group[metric].dropna(); row[f"{metric}_mean"] = values.mean() if len(values) else float("nan"); row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else float("nan")
    rows.append(row)
target = ROOT / "results/aggregates/exact_performance.csv"; pd.DataFrame(rows).to_csv(target, index=False)
print(f"EXACT_RUNS={len(raw)} EXACT_CONDITIONS={len(rows)}")
