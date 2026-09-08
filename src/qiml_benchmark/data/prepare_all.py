"""Download immutable official dataset archives without opening test arrays."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()
RAW = ROOT / "data" / "raw"

FILES = {
    "mnist_train_images.gz": ("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "mnist_train_labels.gz": ("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "mnist_test_images.gz": ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    "mnist_test_labels.gz": ("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    "fashion_train_images.gz": ("https://ossci-datasets.s3.amazonaws.com/fashion-mnist/train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    "fashion_train_labels.gz": ("https://ossci-datasets.s3.amazonaws.com/fashion-mnist/train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
    "fashion_test_images.gz": ("https://ossci-datasets.s3.amazonaws.com/fashion-mnist/t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
    "fashion_test_labels.gz": ("https://ossci-datasets.s3.amazonaws.com/fashion-mnist/t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    "cifar-10-python.tar.gz": ("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "c58f30108f718f92721af3b95e74349a"),
    "breastmnist.npz": ("https://zenodo.org/records/10519652/files/breastmnist.npz?download=1", "750601b1f35ba3300ea97c75c52ff8f6"),
}


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, (url, expected_md5) in FILES.items():
        path = RAW / name
        if not path.exists():
            print(f"downloading {url} -> {path}", flush=True)
            urllib.request.urlretrieve(url, path)
        observed_md5 = digest(path, "md5")
        if observed_md5 != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {name}: {observed_md5} != {expected_md5}")
        manifest.append({
            "filename": name,
            "url": url,
            "bytes": path.stat().st_size,
            "md5": observed_md5,
            "sha256": digest(path, "sha256"),
            "test_payload_opened": False,
        })
    (ROOT / "data_manifests" / "raw_file_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
