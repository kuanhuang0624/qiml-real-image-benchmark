"""Official test loaders guarded by a new reproduction's immutable hash lock."""
from __future__ import annotations

import gzip
import hashlib
import pickle
import struct
import subprocess
import tarfile
from pathlib import Path

import numpy as np

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root(); RAW = ROOT / "data/raw"


def verify_lock() -> None:
    from qiml_benchmark.protocol import verify
    verify(ROOT)


def load_official_test(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    verify_lock()
    if dataset in {"mnist", "fashion_mnist"}:
        stem = "mnist" if dataset == "mnist" else "fashion"
        with gzip.open(RAW / f"{stem}_test_images.gz", "rb") as handle:
            magic, count, rows, columns = struct.unpack(">IIII", handle.read(16)); images = np.frombuffer(handle.read(), dtype=np.uint8).copy().reshape(count, rows, columns)
        with gzip.open(RAW / f"{stem}_test_labels.gz", "rb") as handle:
            label_magic, label_count = struct.unpack(">II", handle.read(8)); labels = np.frombuffer(handle.read(), dtype=np.uint8).copy()
        if magic != 2051 or label_magic != 2049 or count != label_count: raise RuntimeError("IDX test integrity failure")
        return images, labels
    if dataset == "cifar10":
        with tarfile.open(RAW / "cifar-10-python.tar.gz", "r:gz") as archive:
            handle = archive.extractfile("cifar-10-batches-py/test_batch");
            if handle is None: raise RuntimeError("CIFAR test batch missing")
            record = pickle.load(handle, encoding="bytes")
        return record[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.asarray(record[b"labels"], dtype=np.uint8)
    if dataset == "breastmnist":
        with np.load(RAW / "breastmnist.npz", allow_pickle=False) as archive: return archive["test_images"].copy(), archive["test_labels"].reshape(-1).astype(np.uint8)
    raise ValueError(dataset)
