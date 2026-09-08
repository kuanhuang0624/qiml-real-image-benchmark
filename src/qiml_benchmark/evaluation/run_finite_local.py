"""Finite-shot evaluation for local-readout and classifier quantum methods."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from qiml_benchmark.calibration.temperature import fit_binary_threshold, fit_temperature, probabilities
from qiml_benchmark.circuits.statevector import apply_upload_sequence, product_or_ring
from qiml_benchmark.classical.models import build
from qiml_benchmark.measurements.pauli import finite_readout_48
from qiml_benchmark.metrics.core import classification_metrics
from qiml_benchmark.pdrqc.model import grouped_finite_features
from qiml_benchmark.validation.run_primary_fixed import scores
from qiml_benchmark.vqc.model import VQCDR, YomoMatched

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root(); SCALE = {"mnist": "D_5k", "fashion_mnist": "D_5k", "cifar10": "D_5k", "breastmnist": "D_100pct"}; MODEL_SEEDS = (42, 2026, 3407); MEASUREMENT_SEEDS = (101, 211, 307, 401, 503)


def derived(seed: int, budget: int, tag: int) -> int: return int((seed * 1000003 + budget * 1009 + tag * 9176) % (2**32 - 1))


def main() -> None:
    from qiml_benchmark.data.test_loaders import verify_lock
    verify_lock()
    parser = argparse.ArgumentParser(); parser.add_argument("dataset", choices=tuple(SCALE)); args = parser.parse_args(); dataset = args.dataset; scale = SCALE[dataset]; classes = 2 if dataset == "breastmnist" else 10
    locked = json.loads((ROOT / "configs/frozen/FINAL_CONFIGS.yaml").read_text())["datasets"][dataset]; interface = np.load(ROOT / "cache/pca_features" / f"{dataset}_{scale}.npz", allow_pickle=False); test = np.load(ROOT / "cache/pca_test_features" / f"{dataset}_{scale}.npz", allow_pickle=False)
    x, y, xv, yv, xt, yt = interface["train_angles"], interface["train_labels"], interface["val_angles"], interface["val_labels"], test["test_angles"], test["test_labels"]
    rows = []
    for method, ring in (("QF-Product", False), ("QF-Ring", True)):
        item = locked[method]; train_state, val_state, test_state = product_or_ring(x, ring), product_or_ring(xv, ring), product_or_ring(xt, ring)
        for budget in (1, 8, 32, 128, 512):
            validation_features = {ms: finite_readout_48(val_state, budget, ms) for ms in MEASUREMENT_SEEDS}; test_features = {ms: finite_readout_48(test_state, budget, ms) for ms in MEASUREMENT_SEEDS}
            for model_seed in MODEL_SEEDS:
                train_feature = finite_readout_48(train_state, budget, derived(model_seed, budget, 1)); model = build("C-LR", classes, model_seed, **item["config"]).fit(train_feature, y)
                val_raw = {ms: scores(model, feature) for ms, feature in validation_features.items()}; temperature, threshold = calibrate(val_raw, yv, dataset)
                for ms, feature in test_features.items(): rows.append(save(dataset, method, budget, model_seed, ms, scores(model, feature), yt, temperature, threshold, settings=1 if budget == 1 else 3, groups=1 if budget == 1 else 3))
    for method in ("VQC-DR", "Yomo-Matched"):
        item = locked[method]
        for checkpoint in item["checkpoints"]:
            saved = torch.load(ROOT / checkpoint, map_location="cpu", weights_only=False); model_seed = int(saved["seed"]); model = VQCDR(classes, model_seed) if method == "VQC-DR" else YomoMatched(classes, model_seed); model.load_state_dict(saved["model"]); model.eval()
            val_state = quantum_states(model, xv); test_state = quantum_states(model, xt)
            for budget in (1, 8, 32, 128, 512):
                val_raw = {ms: sampled_model_scores(model, val_state, budget, ms, method) for ms in MEASUREMENT_SEEDS}; temperature, threshold = calibrate(val_raw, yv, dataset)
                for ms in MEASUREMENT_SEEDS: rows.append(save(dataset, method, budget, model_seed, ms, sampled_model_scores(model, test_state, budget, ms, method), yt, temperature, threshold, settings=1, groups=1))
    method = "PdrQC-Matched"; item = locked[method]; selection = json.loads((ROOT / "results/raw/pdr_selections" / item["config"]["selection_file"]).read_text()); groups = selection["groups"]; train_state, val_state, test_state = (apply_upload_sequence(values, selection["sequence"]) for values in (x, xv, xt))
    for budget in (8, 32, 128, 512):
        val_features = {ms: grouped_finite_features(val_state, selection["paulis"], groups, budget, ms) for ms in MEASUREMENT_SEEDS}; test_features = {ms: grouped_finite_features(test_state, selection["paulis"], groups, budget, ms) for ms in MEASUREMENT_SEEDS}
        for model_seed in MODEL_SEEDS:
            train_feature = grouped_finite_features(train_state, selection["paulis"], groups, budget, derived(model_seed, budget, 7)); model = build("C-LR", classes, model_seed, **item["config"]).fit(train_feature, y); val_raw = {ms: scores(model, f) for ms, f in val_features.items()}; temperature, threshold = calibrate(val_raw, yv, dataset)
            for ms, f in test_features.items(): rows.append(save(dataset, method, budget, model_seed, ms, scores(model, f), yt, temperature, threshold, settings=len(groups), groups=len(groups)))
    path = ROOT / "results/raw/finite_parts" / f"{dataset}.csv"; path.parent.mkdir(parents=True, exist_ok=True); fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"dataset": dataset, "finite_runs": len(rows)}, indent=2))


def quantum_states(model: torch.nn.Module, values: np.ndarray, batch: int = 512) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for i in range(0, len(values), batch): parts.append(model.quantum(torch.as_tensor(values[i:i + batch], dtype=torch.float32)).numpy())
    return np.concatenate(parts)


def sample_outcomes(state: np.ndarray, shots: int, seed: int) -> np.ndarray:
    prob = np.abs(state.astype(np.complex128)) ** 2; prob /= prob.sum(1, keepdims=True)
    if np.min(prob) < 0 or np.max(np.abs(prob.sum(1) - 1)) > 1e-12: raise RuntimeError("finite-shot probability normalization failed")
    rng = np.random.default_rng(seed); return np.stack([rng.multinomial(shots, p) for p in prob])


def sampled_model_scores(model: torch.nn.Module, state: np.ndarray, budget: int, seed: int, method: str) -> np.ndarray:
    counts = sample_outcomes(state, budget, seed)
    if method == "VQC-DR":
        index = np.arange(256); z = np.column_stack([1 - 2 * ((index >> (7 - q)) & 1) for q in range(8)]); feature = counts @ z / budget
        with torch.no_grad(): return model.head(torch.as_tensor(feature, dtype=torch.float32)).numpy()
    assignment = model.assignment.numpy(); grouped = np.column_stack([counts[:, assignment == k].sum(1) for k in range(model.classes)]) / model.group_sizes.numpy(); grouped /= np.maximum(grouped.sum(1, keepdims=True), 1e-12); return np.log(np.clip(grouped, 1e-12, 1))


def calibrate(validation_raw: dict[int, np.ndarray], labels: np.ndarray, dataset: str) -> tuple[float, float | None]:
    average_raw = np.mean(list(validation_raw.values()), axis=0); temperature = fit_temperature(average_raw, labels, split="validation"); threshold = None
    if dataset == "breastmnist": threshold = fit_binary_threshold(probabilities(average_raw, temperature)[:, 1], labels, split="validation")
    return temperature, threshold


def save(dataset: str, method: str, budget: int, model_seed: int, measurement_seed: int, raw: np.ndarray, labels: np.ndarray, temperature: float, threshold: float | None, settings: int, groups: int) -> dict[str, object]:
    prob = probabilities(raw, temperature); prediction = prob.argmax(1) if threshold is None else (prob[:, 1] >= threshold).astype(np.int8); metrics = classification_metrics(labels, prob, threshold); directory = ROOT / "predictions/test/finite_shot" / dataset; directory.mkdir(parents=True, exist_ok=True); path = directory / f"{method}_B{budget}_m{model_seed}_s{measurement_seed}.npz"; np.savez_compressed(path, sample_id=np.arange(len(labels)), label=labels, raw_score=raw, calibrated_probability=prob, predicted_class=prediction, temperature=temperature, threshold=np.nan if threshold is None else threshold)
    row = {"dataset": dataset, "method": method, "budget_definition": "total_state_preparations_per_image", "nominal_budget": budget, "actual_state_preparations_per_image": budget, "model_seed": model_seed, "measurement_seed": measurement_seed, "measurement_settings": settings, "measurement_groups": groups, "temperature": temperature, "threshold": "" if threshold is None else threshold, "prediction_file": str(path.relative_to(ROOT)), "offline_shot_draws": len(labels) * budget, "real_qpu_shots": 0, "status": "COMPLETED"}; row.update({f"metric_{k}": v for k, v in metrics.items()}); return row


if __name__ == "__main__": main()
