"""Primary validation for deterministic/fixed-representation methods; never loads test data."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from qiml_benchmark.calibration.temperature import fit_temperature, probabilities
from qiml_benchmark.circuits.statevector import apply_upload_sequence, product_or_ring
from qiml_benchmark.classical.models import build, trig_features
from qiml_benchmark.measurements.pauli import readout_48
from qiml_benchmark.metrics.core import classification_metrics
from qiml_benchmark.pdrqc.model import feature_matrix as pdr_features, select as pdr_select
from qiml_benchmark.quantum_kernels.iqp import exact_kernel

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()
PRIMARY_SCALE = {"mnist": "D_5k", "fashion_mnist": "D_5k", "cifar10": "D_5k", "breastmnist": "D_100pct"}
CLASSES = {"mnist": 10, "fashion_mnist": 10, "cifar10": 10, "breastmnist": 2}


def scores(model, data: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"): return np.asarray(model.decision_function(data))
    return np.log(np.clip(model.predict_proba(data), 1e-12, 1))


def evaluate(dataset: str, scale: str, method: str, config: dict[str, object], seed: int, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray) -> dict[str, object]:
    started = time.perf_counter(); classifier_method = method if method.startswith("C-") else "C-LR"
    model = build(classifier_method, CLASSES[dataset], seed, **config).fit(train_x, train_y); raw = scores(model, val_x)
    temperature = fit_temperature(raw, val_y, split="validation"); prob = probabilities(raw, temperature); metric = classification_metrics(val_y, prob)
    primary = metric["auroc"] if dataset == "breastmnist" else metric["accuracy"]
    return {"dataset": dataset, "training_scale": scale, "method": method, "config_id": json.dumps(config, sort_keys=True, separators=(",", ":")), "model_seed": seed, "primary_metric": primary, "brier": metric["brier"], "temperature": temperature, "train_validation_seconds": time.perf_counter() - started, "status": "COMPLETED", "test_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("dataset", choices=tuple(PRIMARY_SCALE)); parser.add_argument("--scale"); parser.add_argument("--mode", choices=("primary", "scaling"), default="primary"); parser.add_argument("--skip-pdr", action="store_true"); parser.add_argument("--only-pdr", action="store_true"); args = parser.parse_args(); dataset = args.dataset; scale = args.scale or PRIMARY_SCALE[dataset]
    archive = np.load(ROOT / "cache/pca_features" / f"{dataset}_{scale}.npz", allow_pickle=False)
    x, y, xv, yv = (archive[k] for k in ("train_angles", "train_labels", "val_angles", "val_labels")); rows: list[dict[str, object]] = []
    grids = {
        "C-LR": [{"C": c} for c in (.1, 1, 10)],
        "C-MLP": [{"alpha": a, "epochs": 300} for a in (1e-4, 1e-3, 1e-2)],
        "C-Poly": [{"degree": d, "C": c} for d in (2, 3, 4) for c in (.1, 1, 10)],
        "C-RBF": [{"C": c, "gamma": g} for c in (.1, 1, 10) for g in (.03, .1, .3)],
        "C-RFF": [{"dimension": d, "gamma": .1, "C": 1} for d in (48, 128, 256)],
    }
    if args.mode == "scaling":
        selected_gamma = {"mnist": .1, "fashion_mnist": .03, "cifar10": .03, "breastmnist": .3}[dataset]
        grids = {
            "C-LR": [{"C": c} for c in (.1, 1, 10)],
            "C-MLP": [{"alpha": a, "epochs": 300} for a in (1e-4, 1e-3, 1e-2)],
            "C-Poly": [{"degree": 3, "C": c} for c in (.1, 1, 10)],
            "C-RBF": [{"C": c, "gamma": selected_gamma} for c in (.1, 1, 10)],
            "C-RFF": [{"dimension": 256, "gamma": .1, "C": c} for c in (.1, 1, 10)],
        }
    stochastic = {"C-MLP", "C-RFF"}
    full_natural = dataset != "breastmnist" and scale == "D_full"
    for method, grid in (() if args.only_pdr else grids.items()):
        if full_natural and method in {"C-Poly", "C-RBF"}: continue
        for config in grid:
            for seed in ((42, 2026, 3407) if method in stochastic else (42,)):
                try: rows.append(evaluate(dataset, scale, method, config, seed, x, y, xv, yv))
                except Exception as error: rows.append(failure(dataset, scale, method, config, seed, error))
    for level in (() if args.only_pdr else ((2,) if args.mode == "scaling" else (1, 2))):
        tx, vx = trig_features(x, level), trig_features(xv, level)
        for c in (.1, 1, 10):
            config = {"level": level, "C": c, "dimension": tx.shape[1]}
            try: rows.append(evaluate(dataset, scale, "C-Trig", config, 42, tx, y, vx, yv))
            except Exception as error: rows.append(failure(dataset, scale, "C-Trig", config, 42, error))

    raw = np.load(ROOT / "cache/resnet_features" / f"{dataset}.npz", allow_pickle=False)
    if dataset == "breastmnist": resnet_x, resnet_v = raw["train_features"][archive["train_ids"]], raw["val_features"]
    else:
        resnet_x = raw["train_features"][archive["train_ids"]]; resnet_v = raw["train_features"][_validation_ids(dataset)]
    for c in (() if args.only_pdr else (.01, .1, 1)):
        config = {"C": c}
        try: rows.append(evaluate(dataset, scale, "C-ResNet", config, 42, resnet_x, y, resnet_v, yv))
        except Exception as error: rows.append(failure(dataset, scale, "C-ResNet", config, 42, error))

    for method, ring in (() if args.only_pdr else (("QF-Product", False), ("QF-Ring", True))):
        tx, vx = _readout_batched(x, ring), _readout_batched(xv, ring)
        for c in (.1, 1, 10):
            config = {"C": c, "readout_dimension": 48}
            try: rows.append(evaluate(dataset, scale, method, config, 42, tx, y, vx, yv))
            except Exception as error: rows.append(failure(dataset, scale, method, config, 42, error))

    if not full_natural and not args.only_pdr:
        kernel, cross = exact_kernel(x), exact_kernel(xv, x)
        for c in (.1, 1, 10):
            config = {"C": c, "repetitions": 2}
            started = time.perf_counter()
            try:
                model = SVC(kernel="precomputed", C=c).fit(kernel, y); raw_score = np.asarray(model.decision_function(cross)); temperature = fit_temperature(raw_score, yv, split="validation"); prob = probabilities(raw_score, temperature); metric = classification_metrics(yv, prob)
                rows.append({"dataset": dataset, "training_scale": scale, "method": "QK-IQP", "config_id": json.dumps(config, sort_keys=True, separators=(",", ":")), "model_seed": 42, "primary_metric": metric["auroc"] if dataset == "breastmnist" else metric["accuracy"], "brier": metric["brier"], "temperature": temperature, "train_validation_seconds": time.perf_counter() - started, "status": "COMPLETED", "test_accessed": False})
            except Exception as error: rows.append(failure(dataset, scale, "QK-IQP", config, 42, error))

    if not args.skip_pdr:
        started = time.perf_counter()
        try:
            selection = pdr_select(x, y, xv, yv, classes=CLASSES[dataset], max_depth=3)
            tx = pdr_features(apply_upload_sequence(x, selection.sequence), selection.paulis); vx = pdr_features(apply_upload_sequence(xv, selection.sequence), selection.paulis)
            selection_dir = ROOT / "results/raw/pdr_selections"; selection_dir.mkdir(parents=True, exist_ok=True)
            (selection_dir / f"{dataset}_{scale}.json").write_text(json.dumps({"sequence": selection.sequence, "paulis": selection.paulis, "groups": selection.groups, "validation_score": selection.validation_score}, indent=2) + "\n")
            for c in (.1, 1, 10): rows.append(evaluate(dataset, scale, "PdrQC-Matched", {"C": c, "selection_file": f"{dataset}_{scale}.json", "depth": len(selection.sequence), "observables": len(selection.paulis), "groups": len(selection.groups)}, 42, tx, y, vx, yv))
        except Exception as error: rows.append(failure(dataset, scale, "PdrQC-Matched", {"max_depth": 3}, 42, error, time.perf_counter() - started))

    prefix = "fixed" if args.mode == "primary" else "scaling_fixed"
    target = ROOT / "results/raw/validation_parts" / f"{prefix}_{dataset}_{scale}.csv"; target.parent.mkdir(parents=True, exist_ok=True); write_rows(target, rows)
    print(json.dumps({"dataset": dataset, "rows": len(rows), "failures": sum(r["status"] != "COMPLETED" for r in rows)}, indent=2))


def _readout_batched(angles: np.ndarray, ring: bool, batch: int = 1000) -> np.ndarray:
    return np.concatenate([readout_48(product_or_ring(angles[start:start + batch], ring)) for start in range(0, len(angles), batch)])


def _validation_ids(dataset: str) -> np.ndarray:
    with (ROOT / "data_manifests/split_manifests" / f"{dataset}_splits.csv").open(newline="") as handle: return np.asarray([int(r["sample_id"]) for r in csv.DictReader(handle) if r["split"] == "validation"])


def failure(dataset: str, scale: str, method: str, config: dict[str, object], seed: int, error: Exception, seconds: float = 0) -> dict[str, object]:
    return {"dataset": dataset, "training_scale": scale, "method": method, "config_id": json.dumps(config, sort_keys=True, separators=(",", ":")), "model_seed": seed, "primary_metric": "", "brier": "", "temperature": "", "train_validation_seconds": seconds, "status": f"FAILED:{type(error).__name__}:{error}", "test_accessed": False}


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        with path.open(newline="") as handle: rows = list(csv.DictReader(handle)) + rows
        keyed = {(str(r["dataset"]), str(r["training_scale"]), str(r["method"]), str(r["config_id"]), str(r["model_seed"])): r for r in rows}; rows = list(keyed.values())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
