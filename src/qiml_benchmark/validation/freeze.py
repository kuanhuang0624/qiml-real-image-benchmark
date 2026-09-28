"""Aggregate validation-only results and freeze selections before test access."""
from __future__ import annotations

import csv
import glob
import hashlib
import json
from pathlib import Path

import pandas as pd

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()
PRIMARY_SCALE = {"mnist": "D_5k", "fashion_mnist": "D_5k", "cifar10": "D_5k", "breastmnist": "D_100pct"}
METHODS = ("C-LR", "C-MLP", "C-Poly", "C-RBF", "C-RFF", "C-Trig", "C-ResNet", "C-CNN", "QK-IQP", "QF-Product", "QF-Ring", "VQC-DR", "Yomo-Matched", "PdrQC-Matched")


def select(group: pd.DataFrame) -> dict[str, object]:
    completed = group[group.status == "COMPLETED"].copy()
    if completed.empty: return {"status": "NOT_COMPLETED"}
    summary = completed.groupby("config_id", as_index=False).agg(primary_metric=("primary_metric", "mean"), brier=("brier", "mean"), train_validation_seconds=("train_validation_seconds", "sum"))
    candidates = summary[summary.primary_metric >= summary.primary_metric.max() - .002].sort_values(["brier", "train_validation_seconds", "config_id"])
    chosen = candidates.iloc[0]; runs = completed[completed.config_id == chosen.config_id]
    return {"status": "FROZEN", "config": json.loads(chosen.config_id), "primary_metric_mean": float(chosen.primary_metric), "brier_mean": float(chosen.brier), "model_seeds": sorted(int(x) for x in runs.model_seed.unique()), "temperatures": {str(int(r.model_seed)): float(r.temperature) for _, r in runs.iterrows()}, "checkpoints": [str(x) for x in runs.get("checkpoint", pd.Series(dtype=str)).dropna() if str(x)]}


def load_parts() -> tuple[pd.DataFrame, pd.DataFrame]:
    files = glob.glob(str(ROOT / "results/raw/validation_parts/*.csv")); frames = [pd.read_csv(path) for path in files]
    primary = pd.concat([frame for path, frame in zip(files, frames) if Path(path).name.startswith(("fixed_", "trainable_")) and not Path(path).name.startswith(("scaling_",))], ignore_index=True, sort=False)
    scaling = pd.concat([frame for path, frame in zip(files, frames) if Path(path).name.startswith("scaling_")], ignore_index=True, sort=False)
    return primary, scaling


def main() -> None:
    primary, scaling = load_parts()
    if (primary.test_accessed.astype(str).str.lower() != "false").any() or (scaling.test_accessed.astype(str).str.lower() != "false").any(): raise RuntimeError("validation part claims test access")
    primary.to_csv(ROOT / "results/raw/validation_runs.csv", index=False); scaling.to_csv(ROOT / "results/raw/data_scaling_validation_runs.csv", index=False)
    primary_selection = {}; primary_rows = []
    for dataset, scale in PRIMARY_SCALE.items():
        primary_selection[dataset] = {}
        for method in METHODS:
            result = select(primary[(primary.dataset == dataset) & (primary.training_scale == scale) & (primary.method == method)]); primary_selection[dataset][method] = result
            primary_rows.append({"dataset": dataset, "training_scale": scale, "method": method, **{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in result.items()}})
    scale_selection = {}; scale_rows = []
    scales = {"mnist": ("D_1k", "D_5k", "D_10k", "D_full"), "fashion_mnist": ("D_1k", "D_5k", "D_10k", "D_full"), "cifar10": ("D_1k", "D_5k", "D_10k", "D_full"), "breastmnist": ("D_25pct", "D_50pct", "D_100pct")}
    for dataset, names in scales.items():
        scale_selection[dataset] = {}
        for scale in names:
            source = primary if scale == PRIMARY_SCALE[dataset] else scaling; scale_selection[dataset][scale] = {}
            for method in METHODS:
                boundary = boundary_status(dataset, scale, method)
                result = {"status": boundary} if boundary else select(source[(source.dataset == dataset) & (source.training_scale == scale) & (source.method == method)])
                scale_selection[dataset][scale][method] = result
                scale_rows.append({"dataset": dataset, "training_scale": scale, "method": method, **{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in result.items()}})
    aggregate_dir = ROOT / "results/aggregates"; aggregate_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(primary_rows).to_csv(aggregate_dir / "validation_summary.csv", index=False); pd.DataFrame(scale_rows).to_csv(aggregate_dir / "data_scaling_validation_summary.csv", index=False)
    frozen = ROOT / "configs/frozen"; frozen.mkdir(parents=True, exist_ok=True)
    write_json_yaml(frozen / "FINAL_CONFIGS.yaml", {"protocol_version": 1, "selection_rule": "primary within 0.002 then lower Brier then lower measured cost then lexical config ID", "test_accessed": False, "datasets": primary_selection})
    write_json_yaml(frozen / "DATA_SCALE_CONFIGS.yaml", {"protocol_version": 1, "maximum_outer_configurations": 4, "test_accessed": False, "datasets": scale_selection})
    contrasts = build_contrasts(primary_selection)
    write_json_yaml(frozen / "PRIMARY_CONTRASTS.yaml", {"protocol_version": 1, "datasets": ["fashion_mnist", "cifar10", "breastmnist"], "MNIST_excluded": True, "permutations": 10000, "bootstrap_replicates": 10000, "global_primary_Holm_family": 12, "uncertainty_Holm_family": 6, "contrasts": contrasts})
    table = ROOT / "results/tables/table_04_validation_selection.csv"; pd.DataFrame(primary_rows).to_csv(table, index=False); write_markdown(ROOT / "results/tables/table_04_validation_selection.md", primary_rows)
    pd.DataFrame(primary_rows).to_csv(ROOT / "results/tables/table_03_validation_selection.csv", index=False); write_markdown(ROOT / "results/tables/table_03_validation_selection.md", primary_rows)
    hashes = frozen / "PRE_TEST_HASHES.sha256"; targets = list(frozen.glob("*.yaml")) + sorted((ROOT / "data_manifests").rglob("*.csv")) + [ROOT / "configs/method_registry.yaml"]
    hashes.write_text("".join(f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in sorted(targets)), encoding="utf-8")
    print(json.dumps({"primary_rows": len(primary_rows), "scaling_rows": len(scale_rows), "validation_trials": len(primary), "scaling_trials": len(scaling), "contrasts": len(contrasts)}, indent=2))


def boundary_status(dataset: str, scale: str, method: str) -> str | None:
    if dataset != "breastmnist" and scale == "D_full" and method == "QK-IQP": return "QUADRATIC_SCALING_BOUNDARY"
    if dataset != "breastmnist" and scale == "D_full" and method in {"C-Poly", "C-RBF"}: return "CLASSICAL_KERNEL_SCALING_BOUNDARY"
    return None


def build_contrasts(selection: dict[str, dict[str, dict[str, object]]]) -> list[dict[str, object]]:
    rows = []
    for dataset in ("fashion_mnist", "cifar10", "breastmnist"):
        selected = selection[dataset]; quantum = [m for m in ("QK-IQP", "QF-Product", "QF-Ring", "VQC-DR", "Yomo-Matched", "PdrQC-Matched") if selected[m]["status"] == "FROZEN"]
        if selected.get("C-RBF", {}).get("status") != "FROZEN":
            raise RuntimeError(f"The paper's C-RBF comparator is not frozen for {dataset}")
        best_q = max(quantum, key=lambda m: selected[m]["primary_metric_mean"])
        rows.extend([
            {"id": f"H1_{dataset}", "dataset": dataset, "method_A": best_q, "method_B": "C-RBF", "setting": "exact", "metric": "AUROC" if dataset == "breastmnist" else "accuracy"},
            {"id": f"H2_{dataset}", "dataset": dataset, "method_A": "QF-Ring", "method_B": "QF-Product", "setting": "finite_shot_total_128", "metric": "AUROC" if dataset == "breastmnist" else "accuracy"},
            {"id": f"H3_{dataset}", "dataset": dataset, "method_A": "Yomo-Matched", "method_B": "VQC-DR", "setting": "finite_shot_total_8", "metric": "AUROC" if dataset == "breastmnist" else "accuracy"},
            {"id": f"H4_{dataset}", "dataset": dataset, "method_A": "PdrQC-Matched", "method_B": "QF-Ring", "setting": "finite_shot_total_128", "metric": "AUROC" if dataset == "breastmnist" else "accuracy"},
        ])
    return rows


def sha256(path: Path) -> str:
    h = hashlib.sha256();
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def write_json_yaml(path: Path, value: object) -> None: path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = ["# Table 4. Validation selection", "", "| Dataset | Scale | Method | Status | Primary | Brier | Frozen config |", "|---|---|---|---|---:|---:|---|"]
    for r in rows: lines.append(f"| {r['dataset']} | {r['training_scale']} | {r['method']} | {r['status']} | {r.get('primary_metric_mean','')} | {r.get('brier_mean','')} | `{r.get('config','')}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
