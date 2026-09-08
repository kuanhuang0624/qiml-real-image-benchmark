"""Extract deterministic frozen ResNet-18 features from training/validation images only."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18

from qiml_benchmark.data.loaders import load_breast_validation, load_official_training

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()
CACHE = ROOT / "cache" / "resnet_features"
WEIGHTS = Path(os.environ.get("QIML_RESNET_WEIGHTS", ROOT / "weights/resnet18-f37072fd.pth"))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_model(device: torch.device) -> torch.nn.Module:
    if not WEIGHTS.is_file():
        raise FileNotFoundError("Provide official ImageNet1K-V1 ResNet-18 weights via QIML_RESNET_WEIGHTS; see experiments/README.md")
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError("ResNet must remain in evaluation mode")
    return model


def preprocess(images: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(images)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1).repeat(1, 3, 1, 1)
    elif tensor.ndim == 4 and tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"unsupported image shape {tuple(tensor.shape)}")
    tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
    tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
    return (tensor - MEAN.to(device)) / STD.to(device)


def extract(model: torch.nn.Module, images: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    outputs = np.empty((len(images), 512), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            stop = min(start + batch_size, len(images))
            outputs[start:stop] = model(preprocess(images[start:stop], device)).cpu().numpy()
    return outputs


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128 if device.type == "cuda" else 16
    model = build_model(device)
    records = []
    for dataset in ("mnist", "fashion_mnist", "cifar10", "breastmnist"):
        started = time.perf_counter()
        train_images, train_labels = load_official_training(dataset)
        train_features = extract(model, train_images, device, batch_size)
        payload: dict[str, np.ndarray] = {"train_features": train_features, "train_labels": train_labels}
        if dataset == "breastmnist":
            val_images, val_labels = load_breast_validation()
            payload.update(val_features=extract(model, val_images, device, batch_size), val_labels=val_labels)
        path = CACHE / f"{dataset}.npz"
        np.savez_compressed(path, **payload)
        records.append({
            "dataset": dataset, "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
            "rows": len(train_features), "dimension": 512, "device": str(device),
            "seconds": time.perf_counter() - started, "weights_sha256": sha256(WEIGHTS),
            "test_accessed": False,
        })
        print(json.dumps(records[-1]), flush=True)
    target = ROOT / "data_manifests" / "feature_manifests" / "raw_resnet_features.csv"
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(records)


if __name__ == "__main__":
    main()
