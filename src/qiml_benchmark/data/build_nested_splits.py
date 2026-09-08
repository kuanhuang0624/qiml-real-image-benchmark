"""Build deterministic stratified train/validation/nested manifests; never read test arrays."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
import struct
import tarfile
from pathlib import Path

import numpy as np

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()
RAW = ROOT / "data" / "raw"
SEED = 20260812


def _idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected label magic {magic}")
        labels = np.frombuffer(handle.read(), dtype=np.uint8).copy()
    if len(labels) != count:
        raise ValueError("label count mismatch")
    return labels


def _cifar_train_labels(path: Path) -> np.ndarray:
    labels: list[int] = []
    with tarfile.open(path, "r:gz") as archive:
        for batch in range(1, 6):
            member = archive.getmember(f"cifar-10-batches-py/data_batch_{batch}")
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError("CIFAR member unavailable")
            record = pickle.load(handle, encoding="bytes")
            labels.extend(record[b"labels"])
    return np.asarray(labels, dtype=np.uint8)


def _breast_train_labels(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        return archive["train_labels"].reshape(-1).astype(np.uint8)


def _hash_rows(rows: list[dict[str, object]]) -> str:
    payload = "\n".join(f"{r['dataset']}|{r['split']}|{r['sample_id']}|{r['label']}|{r['order']}" for r in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def _natural_manifests(dataset: str, labels: np.ndarray) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len(labels) not in (50_000, 60_000) or set(np.unique(labels)) != set(range(10)):
        raise RuntimeError(f"unexpected {dataset} training labels")
    rng = np.random.default_rng(SEED)
    validation: list[int] = []
    nested: dict[str, list[int]] = {"D_1k": [], "D_5k": [], "D_10k": [], "D_full": []}
    for cls in range(10):
        ids = np.flatnonzero(labels == cls)
        ids = ids[rng.permutation(len(ids))]
        validation.extend(ids[:500].tolist())
        pool = ids[500:]
        nested["D_1k"].extend(pool[:100].tolist())
        nested["D_5k"].extend(pool[:500].tolist())
        nested["D_10k"].extend(pool[:1000].tolist())
        nested["D_full"].extend(pool.tolist())
    rows: list[dict[str, object]] = []
    for split, ids in [("validation", validation), *nested.items()]:
        ordered = sorted(ids)
        rows.extend({"dataset": dataset, "split": split, "sample_id": i, "label": int(labels[i]), "order": k} for k, i in enumerate(ordered))
    expected_full = 55_000 if len(labels) == 60_000 else 45_000
    sets = {name: set(ids) for name, ids in nested.items()}
    assert len(validation) == 5_000 and len(sets["D_full"]) == expected_full
    assert sets["D_1k"] <= sets["D_5k"] <= sets["D_10k"] <= sets["D_full"]
    assert not set(validation) & sets["D_full"]
    return rows, {
        "dataset": dataset, "version": "official_torchvision_archive", "official_training_count": len(labels),
        "validation_count": 5000, "D_1k_count": 1000, "D_5k_count": 5000,
        "D_10k_count": 10000, "D_full_count": expected_full, "test_count": 10000,
        "class_counts": json.dumps(np.bincount(labels, minlength=10).tolist()), "image_shape": "28x28" if dataset != "cifar10" else "32x32x3",
        "split_seed": SEED, "manifest_hash": _hash_rows(rows), "overlap_audit": "PASS",
    }


def _breast_manifests(labels: np.ndarray) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len(labels) != 546 or set(np.unique(labels)) != {0, 1}:
        raise RuntimeError("unexpected BreastMNIST training split")
    rng = np.random.default_rng(SEED)
    levels: dict[str, list[int]] = {"D_25pct": [], "D_50pct": [], "D_100pct": []}
    for cls in (0, 1):
        ids = np.flatnonzero(labels == cls)
        ids = ids[rng.permutation(len(ids))]
        levels["D_25pct"].extend(ids[:round(len(ids) * .25)].tolist())
        levels["D_50pct"].extend(ids[:round(len(ids) * .50)].tolist())
        levels["D_100pct"].extend(ids.tolist())
    rows = []
    for split, ids in levels.items():
        ordered = sorted(ids)
        rows.extend({"dataset": "breastmnist", "split": split, "sample_id": i, "label": int(labels[i]), "order": k} for k, i in enumerate(ordered))
    sets = {name: set(ids) for name, ids in levels.items()}
    assert sets["D_25pct"] <= sets["D_50pct"] <= sets["D_100pct"]
    return rows, {
        "dataset": "breastmnist", "version": "MedMNIST_v2_28", "official_training_count": 546,
        "validation_count": 78, "D_1k_count": "N/A", "D_5k_count": "N/A", "D_10k_count": "N/A",
        "D_full_count": 546, "test_count": 156, "class_counts": json.dumps(np.bincount(labels, minlength=2).tolist()),
        "image_shape": "28x28x1", "split_seed": SEED, "manifest_hash": _hash_rows(rows), "overlap_audit": "PASS_official_val_test_not_opened",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    sources = {
        "mnist": _idx_labels(RAW / "mnist_train_labels.gz"),
        "fashion_mnist": _idx_labels(RAW / "fashion_train_labels.gz"),
        "cifar10": _cifar_train_labels(RAW / "cifar-10-python.tar.gz"),
    }
    audit_rows = []
    for dataset, labels in sources.items():
        rows, audit = _natural_manifests(dataset, labels)
        _write_csv(ROOT / "data_manifests" / "split_manifests" / f"{dataset}_splits.csv", rows)
        for split in ("D_1k", "D_5k", "D_10k", "D_full"):
            _write_csv(ROOT / "data_manifests" / "nested_training_manifests" / f"{dataset}_{split}.csv", [r for r in rows if r["split"] == split])
        audit_rows.append(audit)
    breast_rows, breast_audit = _breast_manifests(_breast_train_labels(RAW / "breastmnist.npz"))
    _write_csv(ROOT / "data_manifests" / "split_manifests" / "breastmnist_splits.csv", breast_rows)
    for split in ("D_25pct", "D_50pct", "D_100pct"):
        _write_csv(ROOT / "data_manifests" / "nested_training_manifests" / f"breastmnist_{split}.csv", [r for r in breast_rows if r["split"] == split])
    audit_rows.append(breast_audit)
    _write_csv(ROOT / "data_manifests" / "dataset_versions.csv", audit_rows)
    print(json.dumps(audit_rows, indent=2))


if __name__ == "__main__":
    main()
