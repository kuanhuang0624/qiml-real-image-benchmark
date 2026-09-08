"""Finite-shot QK-IQP with frozen PSD/Nystrom protocol."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

from qiml_benchmark.calibration.temperature import fit_binary_threshold, fit_temperature, probabilities
from qiml_benchmark.metrics.core import classification_metrics
from qiml_benchmark.quantum_kernels.iqp import exact_kernel

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root(); SCALE = {"mnist": "D_5k", "fashion_mnist": "D_5k", "cifar10": "D_5k", "breastmnist": "D_100pct"}; SEEDS = (101, 211, 307, 401, 503)


def main() -> None:
    from qiml_benchmark.data.test_loaders import verify_lock
    verify_lock()
    parser = argparse.ArgumentParser(); parser.add_argument("dataset", choices=tuple(SCALE)); parser.add_argument("--benchmark-one", action="store_true"); args = parser.parse_args(); dataset = args.dataset; scale = SCALE[dataset]; item = json.loads((ROOT / "configs/frozen/FINAL_CONFIGS.yaml").read_text())["datasets"][dataset]["QK-IQP"]
    train = np.load(ROOT / "cache/pca_features" / f"{dataset}_{scale}.npz", allow_pickle=False); test = np.load(ROOT / "cache/pca_test_features" / f"{dataset}_{scale}.npz", allow_pickle=False); x, y, xv, yv, xt, yt = train["train_angles"], train["train_labels"], train["val_angles"], train["val_labels"], test["test_angles"], test["test_labels"]
    exact_train, exact_val, exact_test = exact_kernel(x), exact_kernel(xv, x), exact_kernel(xt, x); rows = []
    budgets = (32,) if args.benchmark_one else (32, 128, 512); seeds = (101,) if args.benchmark_one else SEEDS
    for budget in budgets:
        seed_outputs = {}; diagnostics = {}
        for seed in seeds:
            started = time.perf_counter(); rng = np.random.default_rng(seed); sampled = rng.binomial(budget, np.clip(exact_train, 0, 1)) / budget; sampled = (sampled + sampled.T) / 2; np.fill_diagonal(sampled, 1)
            eigenvalue, eigenvector = np.linalg.eigh(sampled); negative = eigenvalue < 0; positive = eigenvalue > 1e-10; basis = eigenvector[:, positive]
            corrected_train = (basis * eigenvalue[positive]) @ basis.T
            val_cross = rng.binomial(budget, np.clip(exact_val, 0, 1)) / budget; test_cross = rng.binomial(budget, np.clip(exact_test, 0, 1)) / budget
            corrected_val = (val_cross @ basis) @ basis.T; corrected_test = (test_cross @ basis) @ basis.T; classifier = SVC(kernel="precomputed", C=item["config"]["C"]).fit(corrected_train, y)
            seed_outputs[seed] = (classifier.decision_function(corrected_val), classifier.decision_function(corrected_test)); diagnostics[seed] = {"minimum_eigenvalue": float(eigenvalue.min()), "negative_eigenvalues": int(negative.sum()), "negative_mass": float(-eigenvalue[negative].sum()), "frobenius_correction": float(np.sqrt(np.square(eigenvalue[negative]).sum())), "positive_rank": int(positive.sum()), "support_vectors": int(classifier.n_support_.sum()), "seconds": time.perf_counter() - started}
            print(json.dumps({"dataset": dataset, "budget": budget, "seed": seed, **diagnostics[seed]}), flush=True)
        average_val = np.mean([v[0] for v in seed_outputs.values()], axis=0); temperature = fit_temperature(average_val, yv, split="validation"); threshold = fit_binary_threshold(probabilities(average_val, temperature)[:, 1], yv, split="validation") if dataset == "breastmnist" else None
        for seed, (_val, raw) in seed_outputs.items():
            prob = probabilities(raw, temperature); prediction = prob.argmax(1) if threshold is None else (prob[:, 1] >= threshold).astype(np.int8); metric = classification_metrics(yt, prob, threshold); directory = ROOT / "predictions/test/finite_shot" / dataset; directory.mkdir(parents=True, exist_ok=True); path = directory / f"QK-IQP_B{budget}_s{seed}.npz"; np.savez_compressed(path, sample_id=np.arange(len(yt)), label=yt, raw_score=raw, calibrated_probability=prob, predicted_class=prediction, temperature=temperature, threshold=np.nan if threshold is None else threshold)
            diag = diagnostics[seed]; row = {"dataset": dataset, "method": "QK-IQP", "budget_definition": "shots_per_kernel_entry", "nominal_budget": budget, "actual_state_preparations_per_image": budget * diag["support_vectors"], "model_seed": 42, "measurement_seed": seed, "measurement_settings": 1, "measurement_groups": 1, "temperature": temperature, "threshold": "" if threshold is None else threshold, "prediction_file": str(path.relative_to(ROOT)), "offline_shot_draws": budget * (len(x) ** 2 + len(xv) * len(x) + len(xt) * len(x)), "real_qpu_shots": 0, "kernel_entries": len(x) ** 2 + len(xv) * len(x) + len(xt) * len(x), "status": "BENCHMARK_ONLY" if args.benchmark_one else "COMPLETED", **diag}; row.update({f"metric_{k}": v for k, v in metric.items()}); rows.append(row)
    target = ROOT / "results/raw/finite_qk_benchmark.csv" if args.benchmark_one else ROOT / "results/raw/finite_parts" / f"qk_{dataset}.csv"; target.parent.mkdir(parents=True, exist_ok=True); fields = sorted({k for r in rows for k in r})
    with target.open("w", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
