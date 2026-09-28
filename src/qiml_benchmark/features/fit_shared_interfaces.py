"""Fit train-only standardization/PCA/angle maps independently at every frozen scale."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()
RAW_FEATURES = ROOT / "cache" / "resnet_features"
OUT = ROOT / "cache" / "pca_features"
NATURAL_SCALES = ("D_1k", "D_5k", "D_10k", "D_full")
BREAST_SCALES = ("D_25pct", "D_50pct", "D_100pct")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ids(dataset: str, split: str) -> np.ndarray:
    path = ROOT / "data_manifests" / ("nested_training_manifests" if split != "validation" else "split_manifests") / (f"{dataset}_{split}.csv" if split != "validation" else f"{dataset}_splits.csv")
    with path.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == split]
    return np.asarray([int(r["sample_id"]) for r in rows], dtype=np.int64)


def angles(z: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.pi * np.clip(z / np.maximum(scale, 1e-8), -1, 1)).astype(np.float32)


def fit_one(dataset: str, scale_id: str, all_features: np.ndarray, labels: np.ndarray, val_features: np.ndarray, val_labels: np.ndarray, train_ids: np.ndarray) -> dict[str, object]:
    train = all_features[train_ids]
    scaler = StandardScaler().fit(train)
    train_standard = scaler.transform(train)
    pca = PCA(n_components=8, svd_solver="randomized", random_state=20260812).fit(train_standard)
    z_train = pca.transform(train_standard)
    z_val = pca.transform(scaler.transform(val_features))
    quantiles = np.quantile(np.abs(z_train), .99, axis=0)
    a_train, a_val = angles(z_train, quantiles), angles(z_val, quantiles)
    OUT.mkdir(parents=True, exist_ok=True)
    interface = OUT / f"{dataset}_{scale_id}.npz"
    np.savez_compressed(interface, train_angles=a_train, train_labels=labels[train_ids], train_ids=train_ids, val_angles=a_val, val_labels=val_labels)
    preprocessor = OUT / f"{dataset}_{scale_id}_preprocessor.npz"
    np.savez_compressed(preprocessor, mean=scaler.mean_, std=scaler.scale_, pca_mean=pca.mean_, components=pca.components_, explained_variance=pca.explained_variance_, explained_variance_ratio=pca.explained_variance_ratio_, quantiles=quantiles)
    clipping = np.mean(np.abs(z_train / np.maximum(quantiles, 1e-8)) > 1, axis=0)
    return {
        "dataset": dataset, "training_scale": scale_id, "backbone": "ResNet18_Weights.IMAGENET1K_V1",
        "pretraining_source": "ImageNet-1K external data", "raw_feature_dimension": 512, "PCA_dimension": 8,
        "explained_variance": float(pca.explained_variance_ratio_.sum()), "maximum_clipping_rate": float(clipping.max()),
        "preprocessing_hash": sha256(preprocessor), "interface_hash": sha256(interface), "test_accessed": False,
    }


def main() -> None:
    records = []
    for dataset in ("mnist", "fashion_mnist", "cifar10"):
        with np.load(RAW_FEATURES / f"{dataset}.npz", allow_pickle=False) as archive:
            features, labels = archive["train_features"], archive["train_labels"]
        val_ids = ids(dataset, "validation")
        for scale in NATURAL_SCALES:
            records.append(fit_one(dataset, scale, features, labels, features[val_ids], labels[val_ids], ids(dataset, scale)))
    with np.load(RAW_FEATURES / "breastmnist.npz", allow_pickle=False) as archive:
        features, labels = archive["train_features"], archive["train_labels"]
        val_features, val_labels = archive["val_features"], archive["val_labels"]
    for scale in BREAST_SCALES:
        records.append(fit_one("breastmnist", scale, features, labels, val_features, val_labels, ids("breastmnist", scale)))
    target = ROOT / "data_manifests" / "preprocessing_hashes.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(records)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
