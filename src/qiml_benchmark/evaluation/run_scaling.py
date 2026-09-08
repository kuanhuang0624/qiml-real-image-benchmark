"""Frozen exact data-scaling test evaluation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from qiml_benchmark.calibration.temperature import probabilities
from qiml_benchmark.evaluation.run_exact import produce, state_count, kernel_count
from qiml_benchmark.metrics.core import classification_metrics

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()
SCALES = {"mnist": ("D_1k", "D_5k", "D_10k", "D_full"), "fashion_mnist": ("D_1k", "D_5k", "D_10k", "D_full"), "cifar10": ("D_1k", "D_5k", "D_10k", "D_full"), "breastmnist": ("D_25pct", "D_50pct", "D_100pct")}
PRIMARY = {"mnist": "D_5k", "fashion_mnist": "D_5k", "cifar10": "D_5k", "breastmnist": "D_100pct"}


def main() -> None:
    from qiml_benchmark.data.test_loaders import verify_lock
    verify_lock()
    parser = argparse.ArgumentParser(); parser.add_argument("dataset", choices=tuple(SCALES)); parser.add_argument("scale", choices=sum(SCALES.values(), ())); args = parser.parse_args(); dataset, scale = args.dataset, args.scale
    if scale not in SCALES[dataset]: raise ValueError("invalid dataset/scale")
    target = ROOT / "results/raw/scaling_parts" / f"{dataset}_{scale}.csv"; target.parent.mkdir(parents=True, exist_ok=True)
    if scale == PRIMARY[dataset]:
        import pandas as pd
        raw = pd.read_csv(ROOT / "results/raw/exact_test_runs.csv"); raw = raw[(raw.dataset == dataset) & (raw.training_scale == scale)].copy(); raw["source"] = "phase5_primary_reuse"; raw.to_csv(target, index=False); print(dataset, scale, len(raw)); return
    locked = json.loads((ROOT / "configs/frozen/DATA_SCALE_CONFIGS.yaml").read_text())["datasets"][dataset][scale]; train = np.load(ROOT / "cache/pca_features" / f"{dataset}_{scale}.npz", allow_pickle=False); test = np.load(ROOT / "cache/pca_test_features" / f"{dataset}_{scale}.npz", allow_pickle=False); x, y, xt, yt = train["train_angles"], train["train_labels"], test["test_angles"], test["test_labels"]; rows = []
    for method, item in locked.items():
        if item["status"] != "FROZEN": rows.append({"dataset": dataset, "training_scale": scale, "method": method, "model_seed": "", "status": item["status"], "source": "frozen_boundary"}); continue
        outputs = produce(dataset, scale, method, item, item["config"], x, y, xt)
        for seed, raw in outputs.items():
            temperature = float(item["temperatures"][str(seed)]); prob = probabilities(raw, temperature); threshold = None if dataset != "breastmnist" else float(item["validation_selected_thresholds"][str(seed)]); prediction = prob.argmax(1) if threshold is None else (prob[:, 1] >= threshold).astype(np.int8); metric = classification_metrics(yt, prob, threshold); directory = ROOT / "predictions/data_scaling" / dataset / scale; directory.mkdir(parents=True, exist_ok=True); path = directory / f"{method}_{seed}.npz"; np.savez_compressed(path, sample_id=np.arange(len(yt)), label=yt, raw_score=raw, calibrated_probability=prob, predicted_class=prediction, temperature=temperature, threshold=np.nan if threshold is None else threshold)
            row = {"dataset": dataset, "training_scale": scale, "method": method, "model_seed": seed, "config_id": json.dumps(item["config"], sort_keys=True, separators=(",", ":")), "prediction_file": str(path.relative_to(ROOT)), "temperature": temperature, "threshold": "" if threshold is None else threshold, "statevector_evaluations": state_count(method, len(x), len(xt)), "kernel_entries": kernel_count(method, len(x), len(xt)), "status": "COMPLETED", "source": "phase7_scaling"}; row.update({f"metric_{k}": v for k, v in metric.items()}); rows.append(row)
    fields = sorted({k for row in rows for k in row})
    with target.open("w", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"dataset": dataset, "scale": scale, "rows": len(rows), "completed": sum(r["status"] == "COMPLETED" for r in rows)}, indent=2))


if __name__ == "__main__": main()
