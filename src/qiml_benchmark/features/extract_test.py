"""One-time guarded test feature extraction and frozen scale-specific transforms."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from qiml_benchmark.data.test_loaders import load_official_test, verify_lock
from qiml_benchmark.features.extract_resnet import build_model, extract
from qiml_benchmark.features.interface import transform_saved as transform

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()
SCALES = {"mnist": ("D_1k", "D_5k", "D_10k", "D_full"), "fashion_mnist": ("D_1k", "D_5k", "D_10k", "D_full"), "cifar10": ("D_1k", "D_5k", "D_10k", "D_full"), "breastmnist": ("D_25pct", "D_50pct", "D_100pct")}


def main() -> None:
    verify_lock(); first = datetime.now().astimezone().isoformat(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.set_num_threads(32 if device.type == "cpu" else 2); model = build_model(device); records = []
    feature_dir = ROOT / "cache/resnet_test_features"; angle_dir = ROOT / "cache/pca_test_features"; feature_dir.mkdir(parents=True, exist_ok=True); angle_dir.mkdir(parents=True, exist_ok=True)
    for dataset, scales in SCALES.items():
        images, labels = load_official_test(dataset); features = extract(model, images, device, 32 if device.type == "cpu" else 128); raw_path = feature_dir / f"{dataset}.npz"; np.savez_compressed(raw_path, test_features=features, test_labels=labels)
        for scale in scales:
            target = angle_dir / f"{dataset}_{scale}.npz"; angles = transform(features, ROOT / "cache/pca_features" / f"{dataset}_{scale}_preprocessor.npz"); np.savez_compressed(target, test_angles=angles, test_labels=labels)
            records.append({"dataset": dataset, "scale": scale, "count": len(labels), "angle_hash": hashlib.sha256(target.read_bytes()).hexdigest(), "raw_feature_hash": hashlib.sha256(raw_path.read_bytes()).hexdigest(), "device": str(device), "first_test_access": first})
        print(json.dumps(records[-1]), flush=True)
    manifest = ROOT / "data_manifests/feature_manifests/test_features.csv"
    with manifest.open("w", newline="") as handle: writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(records)
    from qiml_benchmark.protocol import digest
    lock_hash = digest(ROOT / "REPRODUCTION_LOCK.json")
    with (ROOT / "TEST_ACCESS_LOG.md").open("a", encoding="utf-8") as log:
        log.write(f"# New reproduction test extraction\n\nUTC: `{first}`. Lock SHA256: `{lock_hash}`. Hash guard passed before opening test arrays. This is not the original historical test access.\n\n")


if __name__ == "__main__": main()
