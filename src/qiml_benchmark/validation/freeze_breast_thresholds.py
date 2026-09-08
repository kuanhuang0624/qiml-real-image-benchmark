"""Fit every frozen BreastMNIST decision threshold using validation data only."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from sklearn.svm import SVC

from qiml_benchmark.calibration.temperature import fit_binary_threshold, probabilities
from qiml_benchmark.circuits.statevector import apply_upload_sequence, product_or_ring
from qiml_benchmark.classical.cnn import CompactCNN
from qiml_benchmark.classical.models import build, trig_features
from qiml_benchmark.data.loaders import load_breast_validation, load_official_training
from qiml_benchmark.measurements.pauli import readout_48
from qiml_benchmark.pdrqc.model import feature_matrix as pdr_features
from qiml_benchmark.quantum_kernels.iqp import exact_kernel
from qiml_benchmark.validation.run_primary_fixed import scores
from qiml_benchmark.validation.run_primary_trainable import image_tensor, quantum_predict
from qiml_benchmark.vqc.model import VQCDR, YomoMatched

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()


def main() -> None:
    final_path = ROOT / "configs/frozen/FINAL_CONFIGS.yaml"; scale_path = ROOT / "configs/frozen/DATA_SCALE_CONFIGS.yaml"
    final, scaling = json.loads(final_path.read_text()), json.loads(scale_path.read_text())
    for scale in ("D_25pct", "D_50pct", "D_100pct"):
        selected = scaling["datasets"]["breastmnist"][scale]
        for method, item in selected.items():
            if item["status"] != "FROZEN": continue
            thresholds = thresholds_for(scale, method, item)
            item["validation_selected_thresholds"] = thresholds
            if scale == "D_100pct": final["datasets"]["breastmnist"][method]["validation_selected_thresholds"] = thresholds
    final_path.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n"); scale_path.write_text(json.dumps(scaling, indent=2, sort_keys=True) + "\n")
    targets = list((ROOT / "configs/frozen").glob("*.yaml")) + sorted((ROOT / "data_manifests").rglob("*.csv")) + [ROOT / "configs/method_registry.yaml"]
    (ROOT / "configs/frozen/PRE_TEST_HASHES.sha256").write_text("".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT)}\n" for path in sorted(targets)), encoding="utf-8")
    print("BREAST_VALIDATION_THRESHOLDS_FROZEN")


def thresholds_for(scale: str, method: str, item: dict[str, object]) -> dict[str, float]:
    archive = np.load(ROOT / "cache/pca_features" / f"breastmnist_{scale}.npz", allow_pickle=False); x, y, xv, yv = (archive[k] for k in ("train_angles", "train_labels", "val_angles", "val_labels")); config = item["config"]
    raw_by_seed: dict[str, np.ndarray] = {}
    if method in {"C-LR", "C-MLP", "C-Poly", "C-RBF", "C-RFF"}:
        for seed in item["model_seeds"]: raw_by_seed[str(seed)] = scores(build(method, 2, seed, **config).fit(x, y), xv)
    elif method == "C-Trig":
        tx, vx = trig_features(x, config["level"]), trig_features(xv, config["level"])
        raw_by_seed["42"] = scores(build(method, 2, 42, **config).fit(tx, y), vx)
    elif method == "C-ResNet":
        raw = np.load(ROOT / "cache/resnet_features/breastmnist.npz", allow_pickle=False); model = build(method, 2, 42, **config).fit(raw["train_features"][archive["train_ids"]], y); raw_by_seed["42"] = scores(model, raw["val_features"])
    elif method in {"QF-Product", "QF-Ring"}:
        ring = method == "QF-Ring"; tx, vx = readout_48(product_or_ring(x, ring)), readout_48(product_or_ring(xv, ring)); raw_by_seed["42"] = scores(build("C-LR", 2, 42, **config).fit(tx, y), vx)
    elif method == "QK-IQP":
        model = SVC(kernel="precomputed", C=config["C"]).fit(exact_kernel(x), y); raw_by_seed["42"] = model.decision_function(exact_kernel(xv, x))
    elif method == "PdrQC-Matched":
        selection = json.loads((ROOT / "results/raw/pdr_selections" / config["selection_file"]).read_text()); tx = pdr_features(apply_upload_sequence(x, selection["sequence"]), selection["paulis"]); vx = pdr_features(apply_upload_sequence(xv, selection["sequence"]), selection["paulis"]); raw_by_seed["42"] = scores(build("C-LR", 2, 42, **config).fit(tx, y), vx)
    elif method in {"VQC-DR", "Yomo-Matched"}:
        for checkpoint in item["checkpoints"]:
            saved = torch.load(ROOT / checkpoint, map_location="cpu", weights_only=False); seed = str(saved["seed"]); model = VQCDR(2, saved["seed"]) if method == "VQC-DR" else YomoMatched(2, saved["seed"]); model.load_state_dict(saved["model"]); raw = quantum_predict(model, xv, torch.device("cpu")); raw_by_seed[seed] = np.log(np.clip(raw, 1e-12, 1)) if method == "Yomo-Matched" else raw
    elif method == "C-CNN":
        val_images, _ = load_breast_validation(); tensor = image_tensor(val_images)
        for checkpoint in item["checkpoints"]:
            saved = torch.load(ROOT / checkpoint, map_location="cpu", weights_only=False); seed = str(saved["seed"]); model = CompactCNN(1, 2); model.load_state_dict(saved["model"]); model.eval()
            with torch.no_grad(): raw_by_seed[seed] = model(tensor).numpy()
    else: raise ValueError(method)
    result = {}
    for seed, raw in raw_by_seed.items():
        temperature = float(item["temperatures"][seed]); prob = probabilities(raw, temperature); result[seed] = fit_binary_threshold(prob[:, 1], yv, split="validation")
    return result


if __name__ == "__main__": main()
