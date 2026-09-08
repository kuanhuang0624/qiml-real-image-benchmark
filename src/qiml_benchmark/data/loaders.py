"""Training/validation loaders. Test loading requires the separate guarded path."""
from __future__ import annotations

import gzip
import pickle
import struct
import tarfile
from pathlib import Path

import numpy as np

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()
RAW = ROOT / "data" / "raw"


def _idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"unexpected image magic {magic}")
        images = np.frombuffer(handle.read(), dtype=np.uint8).copy().reshape(count, rows, cols)
    return images


def _idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected label magic {magic}")
        labels = np.frombuffer(handle.read(), dtype=np.uint8).copy()
    if len(labels) != count:
        raise ValueError("label count mismatch")
    return labels


def load_official_training(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Load only official training content (and Breast validation when explicitly requested elsewhere)."""
    if dataset in {"mnist", "fashion_mnist"}:
        stem = "mnist" if dataset == "mnist" else "fashion"
        return _idx_images(RAW / f"{stem}_train_images.gz"), _idx_labels(RAW / f"{stem}_train_labels.gz")
    if dataset == "cifar10":
        images: list[np.ndarray] = []
        labels: list[int] = []
        with tarfile.open(RAW / "cifar-10-python.tar.gz", "r:gz") as archive:
            for batch in range(1, 6):
                handle = archive.extractfile(f"cifar-10-batches-py/data_batch_{batch}")
                if handle is None:
                    raise RuntimeError("missing CIFAR training batch")
                record = pickle.load(handle, encoding="bytes")
                images.append(record[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
                labels.extend(record[b"labels"])
        return np.concatenate(images), np.asarray(labels, dtype=np.uint8)
    if dataset == "breastmnist":
        with np.load(RAW / "breastmnist.npz", allow_pickle=False) as archive:
            return archive["train_images"].copy(), archive["train_labels"].reshape(-1).astype(np.uint8)
    raise ValueError(dataset)


def load_breast_validation() -> tuple[np.ndarray, np.ndarray]:
    with np.load(RAW / "breastmnist.npz", allow_pickle=False) as archive:
        return archive["val_images"].copy(), archive["val_labels"].reshape(-1).astype(np.uint8)
