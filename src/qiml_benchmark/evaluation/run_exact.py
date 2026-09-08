"""Frozen exact primary test evaluation after the pre-test lock."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.svm import SVC

from qiml_benchmark.calibration.temperature import probabilities
from qiml_benchmark.circuits.statevector import apply_upload_sequence, product_or_ring
from qiml_benchmark.classical.cnn import CompactCNN
from qiml_benchmark.classical.models import build, trig_features
from qiml_benchmark.data.test_loaders import load_official_test, verify_lock
from qiml_benchmark.measurements.pauli import readout_48
from qiml_benchmark.metrics.core import classification_metrics
from qiml_benchmark.pdrqc.model import feature_matrix as pdr_features
from qiml_benchmark.quantum_kernels.iqp import exact_kernel
from qiml_benchmark.validation.run_primary_fixed import scores
from qiml_benchmark.validation.run_primary_trainable import image_tensor, quantum_predict
from qiml_benchmark.vqc.model import VQCDR, YomoMatched

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root(); SCALE = {"mnist": "D_5k", "fashion_mnist": "D_5k", "cifar10": "D_5k", "breastmnist": "D_100pct"}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("dataset", choices=tuple(SCALE)); args = parser.parse_args(); dataset = args.dataset; scale = SCALE[dataset]; verify_lock(); config = json.loads((ROOT / "configs/frozen/FINAL_CONFIGS.yaml").read_text())["datasets"][dataset]
    train = np.load(ROOT / "cache/pca_features" / f"{dataset}_{scale}.npz", allow_pickle=False); test = np.load(ROOT / "cache/pca_test_features" / f"{dataset}_{scale}.npz", allow_pickle=False); x, y, xt, yt = train["train_angles"], train["train_labels"], test["test_angles"], test["test_labels"]
    rows = []
    for method, item in config.items():
        started = time.perf_counter(); method_config = item["config"]; outputs = produce(dataset, scale, method, item, method_config, x, y, xt)
        for seed, raw in outputs.items():
            temperature = float(item["temperatures"][str(seed)]); prob = probabilities(raw, temperature); threshold = None if dataset != "breastmnist" else float(item["validation_selected_thresholds"][str(seed)]); metric = classification_metrics(yt, prob, threshold); prediction = prob.argmax(1) if threshold is None else (prob[:, 1] >= threshold).astype(np.int8)
            target = ROOT / "predictions/test/exact" / dataset; target.mkdir(parents=True, exist_ok=True); path = target / f"{method}_{seed}.npz"; np.savez_compressed(path, sample_id=np.arange(len(yt)), label=yt, raw_score=raw, calibrated_probability=prob, predicted_class=prediction, temperature=temperature, threshold=np.nan if threshold is None else threshold)
            row = {"dataset": dataset, "training_scale": scale, "method": method, "model_seed": seed, "config_id": json.dumps(method_config, sort_keys=True, separators=(",", ":")), "prediction_file": str(path.relative_to(ROOT)), "temperature": temperature, "threshold": "" if threshold is None else threshold, "wall_seconds": time.perf_counter() - started, "statevector_evaluations": state_count(method, len(x), len(xt)), "kernel_entries": kernel_count(method, len(x), len(xt)), "status": "COMPLETED"}
            row.update({f"metric_{key}": value for key, value in metric.items()}); rows.append(row)
    target = ROOT / "results/raw/exact_parts" / f"{dataset}.csv"; target.parent.mkdir(parents=True, exist_ok=True); write_rows(target, rows); print(json.dumps({"dataset": dataset, "runs": len(rows), "methods": len(config)}, indent=2))


def produce(dataset: str, scale: str, method: str, item: dict[str, object], config: dict[str, object], x: np.ndarray, y: np.ndarray, xt: np.ndarray) -> dict[int, np.ndarray]:
    seeds = [int(s) for s in item["model_seeds"]]
    if method in {"C-LR", "C-MLP", "C-Poly", "C-RBF", "C-RFF"}: return {seed: scores(build(method, 2 if dataset == "breastmnist" else 10, seed, **config).fit(x, y), xt) for seed in seeds}
    if method == "C-Trig":
        a, b = trig_features(x, config["level"]), trig_features(xt, config["level"]); return {42: scores(build(method, 2 if dataset == "breastmnist" else 10, 42, **config).fit(a, y), b)}
    if method == "C-ResNet":
        raw = np.load(ROOT / "cache/resnet_features" / f"{dataset}.npz", allow_pickle=False); raw_test = np.load(ROOT / "cache/resnet_test_features" / f"{dataset}.npz", allow_pickle=False); model = build(method, 2 if dataset == "breastmnist" else 10, 42, **config).fit(raw["train_features"][np.load(ROOT / "cache/pca_features" / f"{dataset}_{scale}.npz")["train_ids"]], y); return {42: scores(model, raw_test["test_features"])}
    if method in {"QF-Product", "QF-Ring"}:
        ring = method == "QF-Ring"; a, b = batched_readout(x, ring), batched_readout(xt, ring); return {42: scores(build("C-LR", 2 if dataset == "breastmnist" else 10, 42, **config).fit(a, y), b)}
    if method == "QK-IQP":
        gram, cross = exact_kernel(x), exact_kernel(xt, x); return {42: SVC(kernel="precomputed", C=config["C"]).fit(gram, y).decision_function(cross)}
    if method == "PdrQC-Matched":
        selection = json.loads((ROOT / "results/raw/pdr_selections" / config["selection_file"]).read_text()); a = batched_pdr(x, selection); b = batched_pdr(xt, selection); return {42: scores(build("C-LR", 2 if dataset == "breastmnist" else 10, 42, **config).fit(a, y), b)}
    if method in {"VQC-DR", "Yomo-Matched"}:
        result = {}
        for checkpoint in item["checkpoints"]:
            saved = torch.load(ROOT / checkpoint, map_location="cpu", weights_only=False); seed = int(saved["seed"]); model = VQCDR(2 if dataset == "breastmnist" else 10, seed) if method == "VQC-DR" else YomoMatched(2 if dataset == "breastmnist" else 10, seed); model.load_state_dict(saved["model"]); raw = quantum_predict(model, xt, torch.device("cpu")); result[seed] = np.log(np.clip(raw, 1e-12, 1)) if method == "Yomo-Matched" else raw
        return result
    if method == "C-CNN":
        images, _ = load_official_test(dataset); tensor = image_tensor(images); result = {}
        for checkpoint in item["checkpoints"]:
            saved = torch.load(ROOT / checkpoint, map_location="cpu", weights_only=False); seed = int(saved["seed"]); model = CompactCNN(1 if images.ndim == 3 else 3, 2 if dataset == "breastmnist" else 10); model.load_state_dict(saved["model"]); model.eval(); parts = []
            with torch.no_grad():
                for start in range(0, len(tensor), 512): parts.append(model(tensor[start:start + 512]).numpy())
            result[seed] = np.concatenate(parts)
        return result
    raise ValueError(method)


def batched_readout(values: np.ndarray, ring: bool) -> np.ndarray: return np.concatenate([readout_48(product_or_ring(values[i:i + 1000], ring)) for i in range(0, len(values), 1000)])
def batched_pdr(values: np.ndarray, selection: dict[str, object]) -> np.ndarray: return np.concatenate([pdr_features(apply_upload_sequence(values[i:i + 1000], selection["sequence"]), selection["paulis"]) for i in range(0, len(values), 1000)])
def state_count(method: str, train: int, test: int) -> int: return train + test if method.startswith("QF-") or method == "PdrQC-Matched" else test if method in {"VQC-DR", "Yomo-Matched"} else 0
def kernel_count(method: str, train: int, test: int) -> int: return train * train + train * test if method == "QK-IQP" else 0
def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
