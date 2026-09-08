"""GPU validation for VQC-DR, Yomo-Matched, and the compact CNN; test-free."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from qiml_benchmark.calibration.temperature import fit_temperature, probabilities
from qiml_benchmark.classical.cnn import CompactCNN
from qiml_benchmark.data.loaders import load_official_training
from qiml_benchmark.metrics.core import classification_metrics
from qiml_benchmark.validation.run_primary_fixed import CLASSES, PRIMARY_SCALE, _validation_ids
from qiml_benchmark.vqc.model import VQCDR, YomoMatched, parameter_count

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root(); SEEDS = (42, 2026, 3407)


def quantum_predict(model: torch.nn.Module, values: np.ndarray, device: torch.device, batch: int = 512) -> np.ndarray:
    parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch): parts.append(model(torch.as_tensor(values[start:start + batch], dtype=torch.float32, device=device)).cpu().numpy())
    return np.concatenate(parts)


def train_quantum(method: str, x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray, classes: int, seed: int, learning_rate: float, device: torch.device, checkpoint: Path) -> tuple[np.ndarray, dict[str, object]]:
    torch.manual_seed(seed); np_rng = np.random.default_rng(seed); model = (VQCDR(classes, seed) if method == "VQC-DR" else YomoMatched(classes, seed)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4); batch_size = 128; best_metric = -np.inf; best_state = None; stale = 0; losses = []; gradient_norms = []; started = time.perf_counter(); initial_epoch = 0
    resume = checkpoint.with_suffix(".resume.pt")
    if resume.exists():
        saved = torch.load(resume, map_location=device, weights_only=False); model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); initial_epoch = int(saved["epoch"])
        best_metric, best_state, best_epoch, stale = saved["best_metric"], saved["best_validation_state"], saved["best_epoch"], saved["stale"]
        losses, gradient_norms = saved["losses"], saved["gradient_norms"]; np_rng.bit_generator.state = saved["numpy_bit_generator"]; random.setstate(saved["python_rng"]); torch.set_rng_state(saved["torch_cpu_rng"])
        if device.type == "cuda" and saved["torch_cuda_rng"] is not None: torch.cuda.set_rng_state_all(saved["torch_cuda_rng"])
    for epoch in range(initial_epoch, 20):
        model.train(); order = np_rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            index = order[start:start + batch_size]; xb = torch.as_tensor(x[index], dtype=torch.float32, device=device); yb = torch.as_tensor(y[index], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True); output = model(xb); loss = model.loss(output, yb) if method == "Yomo-Matched" else torch.nn.functional.cross_entropy(output, yb)
            loss.backward(); norm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None)); optimizer.step(); losses.append(float(loss.detach())); gradient_norms.append(float(norm.detach()))
        raw = quantum_predict(model, xv, device); score = np.log(np.clip(raw, 1e-12, 1)) if method == "Yomo-Matched" else raw
        temperature = fit_temperature(score, yv, split="validation"); prob = probabilities(score, temperature); metrics = classification_metrics(yv, prob); primary = metrics["auroc"] if classes == 2 else metrics["accuracy"]
        if primary > best_metric + 0.002:
            best_metric = primary; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; best_epoch = epoch + 1; stale = 0
        else: stale += 1
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": None, "epoch": epoch + 1, "seed": seed, "method": method, "best_metric": best_metric, "best_validation_state": best_state, "best_epoch": best_epoch, "stale": stale, "losses": losses, "gradient_norms": gradient_norms, "python_rng": random.getstate(), "numpy_bit_generator": np_rng.bit_generator.state, "torch_cpu_rng": torch.get_rng_state(), "torch_cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, resume)
        if epoch >= 5 and stale >= 4: break
    assert best_state is not None; model.load_state_dict(best_state); torch.save({"model": best_state, "seed": seed, "method": method, "best_epoch": best_epoch}, checkpoint)
    raw = quantum_predict(model, xv, device); raw_score = np.log(np.clip(raw, 1e-12, 1)) if method == "Yomo-Matched" else raw
    return raw_score, {"seconds": time.perf_counter() - started, "best_epoch": best_epoch, "loss_start": losses[0], "loss_end": losses[-1], "median_gradient": float(np.median(gradient_norms)), "parameters": parameter_count(model)}


def image_tensor(images: np.ndarray) -> torch.Tensor:
    if images.ndim == 3: images = images[:, None]
    else: images = images.transpose(0, 3, 1, 2)
    return torch.as_tensor(images, dtype=torch.float32) / 255


def train_cnn(images: np.ndarray, labels: np.ndarray, val_images: np.ndarray, val_labels: np.ndarray, classes: int, seed: int, learning_rate: float, device: torch.device, checkpoint: Path) -> tuple[np.ndarray, dict[str, object]]:
    torch.manual_seed(seed); rng = np.random.default_rng(seed); model = CompactCNN(1 if images.ndim == 3 else 3, classes).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4); train_images = image_tensor(images); validation_images = image_tensor(val_images); batch_size = 128; started = time.perf_counter(); best_metric = -np.inf; best_state = None; stale = 0; initial_epoch = 0
    resume = checkpoint.with_suffix(".resume.pt")
    if resume.exists():
        saved = torch.load(resume, map_location=device, weights_only=False); model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); initial_epoch = int(saved["epoch"])
        best_metric, best_state, best_epoch, stale = saved["best_metric"], saved["best_validation_state"], saved["best_epoch"], saved["stale"]
        rng.bit_generator.state = saved["numpy_bit_generator"]; random.setstate(saved["python_rng"]); torch.set_rng_state(saved["torch_cpu_rng"])
        if device.type == "cuda" and saved["torch_cuda_rng"] is not None: torch.cuda.set_rng_state_all(saved["torch_cuda_rng"])
    for epoch in range(initial_epoch, 12):
        model.train(); order = rng.permutation(len(images))
        for start in range(0, len(images), batch_size):
            index = order[start:start + batch_size]; xb = train_images[index].to(device); yb = torch.as_tensor(labels[index], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True); loss = torch.nn.functional.cross_entropy(model(xb), yb); loss.backward(); optimizer.step()
        model.eval(); outputs = []
        with torch.no_grad():
            for start in range(0, len(val_images), 512): outputs.append(model(validation_images[start:start + 512].to(device)).cpu().numpy())
        raw = np.concatenate(outputs); prob = probabilities(raw, fit_temperature(raw, val_labels, split="validation")); metric = classification_metrics(val_labels, prob); primary = metric["auroc"] if classes == 2 else metric["accuracy"]
        if primary > best_metric + .002: best_metric = primary; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; best_epoch = epoch + 1; stale = 0
        else: stale += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": None, "epoch": epoch + 1, "seed": seed, "method": "C-CNN", "best_metric": best_metric, "best_validation_state": best_state, "best_epoch": best_epoch, "stale": stale, "python_rng": random.getstate(), "numpy_bit_generator": rng.bit_generator.state, "torch_cpu_rng": torch.get_rng_state(), "torch_cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, resume)
        if epoch >= 4 and stale >= 3: break
    assert best_state is not None; model.load_state_dict(best_state); checkpoint.parent.mkdir(parents=True, exist_ok=True); torch.save({"model": best_state, "seed": seed, "method": "C-CNN", "best_epoch": best_epoch}, checkpoint)
    model.eval(); outputs = []
    with torch.no_grad():
        for start in range(0, len(val_images), 512): outputs.append(model(validation_images[start:start + 512].to(device)).cpu().numpy())
    return np.concatenate(outputs), {"seconds": time.perf_counter() - started, "best_epoch": best_epoch, "parameters": sum(p.numel() for p in model.parameters())}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("dataset", choices=tuple(PRIMARY_SCALE)); parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda"); parser.add_argument("--scale"); parser.add_argument("--methods", choices=("all", "cnn", "quantum"), default="all"); parser.add_argument("--mode", choices=("primary", "scaling"), default="primary"); args = parser.parse_args(); dataset = args.dataset; scale = args.scale or PRIMARY_SCALE[dataset]; classes = CLASSES[dataset]
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("allocated CUDA device is unavailable")
    torch.set_num_threads(2); device = torch.device(args.device); archive = np.load(ROOT / "cache/pca_features" / f"{dataset}_{scale}.npz", allow_pickle=False); x, y, xv, yv = (archive[k] for k in ("train_angles", "train_labels", "val_angles", "val_labels")); rows = []
    for method in (() if args.methods == "cnn" else ("VQC-DR", "Yomo-Matched")):
        for learning_rate in (.01, .03):
            for seed in SEEDS:
                config = {"learning_rate": learning_rate, "weight_decay": 1e-4, "epoch_cap": 20, "batch_size": 128}
                checkpoint = ROOT / "checkpoints/validation" / f"{dataset}_{scale}_{method}_{learning_rate}_{seed}.pt"
                try:
                    raw, resources = train_quantum(method, x, y, xv, yv, classes, seed, learning_rate, device, checkpoint); temperature = fit_temperature(raw, yv, split="validation"); metric = classification_metrics(yv, probabilities(raw, temperature)); primary = metric["auroc"] if classes == 2 else metric["accuracy"]
                    rows.append(row(dataset, scale, method, config, seed, primary, metric["brier"], temperature, resources, checkpoint))
                except Exception as error: rows.append(failure(dataset, scale, method, config, seed, error))
    if args.methods != "quantum":
        all_images, all_labels = load_official_training(dataset)
        if dataset == "breastmnist":
            from qiml_benchmark.data.loaders import load_breast_validation
            val_images, val_labels = load_breast_validation()
        else: val_images, val_labels = all_images[_validation_ids(dataset)], all_labels[_validation_ids(dataset)]
        train_images = all_images[archive["train_ids"]]
        for learning_rate in (3e-4, 1e-3):
            for seed in SEEDS:
                config = {"learning_rate": learning_rate, "weight_decay": 1e-4, "epoch_cap": 12, "batch_size": 128}
                checkpoint = ROOT / "checkpoints/validation" / f"{dataset}_{scale}_C-CNN_{learning_rate}_{seed}.pt"
                try:
                    raw, resources = train_cnn(train_images, y, val_images, val_labels, classes, seed, learning_rate, device, checkpoint); temperature = fit_temperature(raw, yv, split="validation"); metric = classification_metrics(yv, probabilities(raw, temperature)); primary = metric["auroc"] if classes == 2 else metric["accuracy"]
                    rows.append(row(dataset, scale, "C-CNN", config, seed, primary, metric["brier"], temperature, resources, checkpoint))
                except Exception as error: rows.append(failure(dataset, scale, "C-CNN", config, seed, error))
    prefix = "trainable" if args.mode == "primary" else "scaling_trainable"
    target = ROOT / "results/raw/validation_parts" / f"{prefix}_{dataset}_{scale}.csv"; target.parent.mkdir(parents=True, exist_ok=True); write_rows(target, rows); print(json.dumps({"dataset": dataset, "scale": scale, "rows": len(rows), "failures": sum(r["status"] != "COMPLETED" for r in rows)}, indent=2))


def row(dataset: str, scale: str, method: str, config: dict[str, object], seed: int, primary: float, brier: float, temperature: float, resources: dict[str, object], checkpoint: Path) -> dict[str, object]:
    return {"dataset": dataset, "training_scale": scale, "method": method, "config_id": json.dumps(config, sort_keys=True, separators=(",", ":")), "model_seed": seed, "primary_metric": primary, "brier": brier, "temperature": temperature, "train_validation_seconds": resources["seconds"], "best_epoch": resources["best_epoch"], "parameters": resources["parameters"], "checkpoint": str(checkpoint.relative_to(ROOT)), "status": "COMPLETED", "test_accessed": False}


def failure(dataset: str, scale: str, method: str, config: dict[str, object], seed: int, error: Exception) -> dict[str, object]:
    return {"dataset": dataset, "training_scale": scale, "method": method, "config_id": json.dumps(config, sort_keys=True, separators=(",", ":")), "model_seed": seed, "primary_metric": "", "brier": "", "temperature": "", "train_validation_seconds": "", "best_epoch": "", "parameters": "", "checkpoint": "", "status": f"FAILED:{type(error).__name__}:{error}", "test_accessed": False}


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        with path.open(newline="") as handle: rows = list(csv.DictReader(handle)) + rows
        keyed = {(str(r["dataset"]), str(r["training_scale"]), str(r["method"]), str(r["config_id"]), str(r["model_seed"])): r for r in rows}; rows = list(keyed.values())
    with path.open("w", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
