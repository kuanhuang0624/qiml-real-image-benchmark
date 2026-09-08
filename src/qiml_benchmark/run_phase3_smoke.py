"""Training/validation-only smoke tests for every frozen method family."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from qiml_benchmark.calibration.temperature import fit_temperature, probabilities as calibrated_probabilities
from qiml_benchmark.circuits.statevector import iqp, probabilities, product_or_ring
from qiml_benchmark.classical.cnn import CompactCNN
from qiml_benchmark.classical.models import build, trig_features
from qiml_benchmark.data.loaders import load_official_training
from qiml_benchmark.measurements.pauli import finite_readout_48, readout_48
from qiml_benchmark.pdrqc.model import select
from qiml_benchmark.quantum_kernels.iqp import exact_kernel, finite_kernel
from qiml_benchmark.vqc.model import VQCDR, YomoMatched, parameter_count
from qiml_benchmark.vqc.training import sanity_gate, train

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()


def balanced_positions(labels: np.ndarray, per_class: int) -> np.ndarray:
    return np.concatenate([np.flatnonzero(labels == label)[:per_class] for label in np.unique(labels)])


def record(rows: list[dict[str, object]], method: str, passed: bool, seconds: float, detail: str) -> None:
    rows.append({"method": method, "status": "PASS" if passed else "FAIL", "wall_seconds": f"{seconds:.6f}", "detail": detail})


def main() -> None:
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); np.random.seed(20260812); torch.manual_seed(20260812)
    archive = np.load(ROOT / "cache/pca_features/mnist_D_1k.npz", allow_pickle=False)
    train_pos = balanced_positions(archive["train_labels"], 8); val_pos = balanced_positions(archive["val_labels"], 8)
    x, y = archive["train_angles"][train_pos], archive["train_labels"][train_pos]
    xv, yv = archive["val_angles"][val_pos], archive["val_labels"][val_pos]
    rows: list[dict[str, object]] = []; failures: list[dict[str, object]] = []

    classical_inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in ("C-LR", "C-MLP", "C-Poly", "C-RBF", "C-RFF"):
        start = time.perf_counter(); model = build(method, 10, 42); model.fit(x, y); pred = model.predict(xv)
        record(rows, method, pred.shape == yv.shape and set(np.unique(pred)) <= set(range(10)), time.perf_counter() - start, f"accuracy={accuracy_score(yv, pred):.6f}")
    start = time.perf_counter(); xt, xvt = trig_features(x, 2), trig_features(xv, 2); model = build("C-Trig", 10, 42).fit(xt, y)
    record(rows, "C-Trig", xt.shape[1] == 129, time.perf_counter() - start, f"dimension={xt.shape[1]};accuracy={accuracy_score(yv, model.predict(xvt)):.6f}")

    raw = np.load(ROOT / "cache/resnet_features/mnist.npz", allow_pickle=False); ids, val_ids = archive["train_ids"][train_pos], _manifest_validation_ids("mnist")[val_pos]
    start = time.perf_counter(); model = build("C-ResNet", 10, 42).fit(raw["train_features"][ids], y); pred = model.predict(raw["train_features"][val_ids])
    record(rows, "C-ResNet", pred.shape == yv.shape, time.perf_counter() - start, f"accuracy={accuracy_score(yv, pred):.6f}")

    images, labels = load_official_training("mnist"); xi = torch.tensor(images[ids, None] / 255, dtype=torch.float32); yi = torch.tensor(y, dtype=torch.long)
    start = time.perf_counter(); cnn = CompactCNN(1, 10); opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    for _ in range(3): opt.zero_grad(); loss = torch.nn.functional.cross_entropy(cnn(xi), yi); loss.backward(); opt.step()
    record(rows, "C-CNN", bool(torch.isfinite(loss)), time.perf_counter() - start, f"loss={float(loss.detach()):.6f}")

    start = time.perf_counter(); kernel = exact_kernel(x); cross = exact_kernel(xv, x); clf = SVC(kernel="precomputed", C=1).fit(kernel, y); pred = clf.predict(cross)
    finite, psd = finite_kernel(kernel, 128, 101, True)
    passed = np.allclose(kernel, kernel.T, atol=1e-10) and np.allclose(np.diag(kernel), 1, atol=1e-10) and np.linalg.eigvalsh(finite).min() > -1e-8
    record(rows, "QK-IQP", passed, time.perf_counter() - start, f"accuracy={accuracy_score(yv,pred):.6f};negative_mass={psd['negative_mass']:.6g}")

    for method, ring in (("QF-Product", False), ("QF-Ring", True)):
        start = time.perf_counter(); state, statev = product_or_ring(x, ring), product_or_ring(xv, ring); exact, exactv = readout_48(state), readout_48(statev)
        sampled_error = [float(np.mean(np.abs(finite_readout_48(statev, budget, 101) - exactv))) for budget in (8, 32, 128, 512)]
        model = build("C-LR", 10, 42).fit(exact, y); pred = model.predict(exactv)
        passed = np.max(np.abs(probabilities(state).sum(1) - 1)) < 1e-10 and sampled_error[-1] < sampled_error[0]
        record(rows, method, passed, time.perf_counter() - start, f"accuracy={accuracy_score(yv,pred):.6f};shot_mae={sampled_error}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter(); outcome = sanity_gate(x, y, 10, 42, device, ROOT / "checkpoints/smoke_vqc.pt")
    record(rows, "VQC-DR", bool(outcome["passed"]), time.perf_counter() - start, json.dumps(outcome, sort_keys=True))

    start = time.perf_counter(); yomo = YomoMatched(10, 42).to(device); yomo_outcome = train(yomo, x, y, steps=40, learning_rate=.03, checkpoint=ROOT / "checkpoints/smoke_yomo.pt")
    with torch.no_grad(): norm = yomo(torch.as_tensor(x, dtype=torch.float32, device=device)).sum(1)
    record(rows, "Yomo-Matched", bool(torch.allclose(norm, torch.ones_like(norm), atol=1e-5) and yomo_outcome["nonzero_gradients"]), time.perf_counter() - start, json.dumps(yomo_outcome, sort_keys=True))

    start = time.perf_counter(); selection = select(x, y, xv, yv, classes=10, max_depth=1)
    record(rows, "PdrQC-Matched", len(selection.paulis) == 60 and len(selection.groups) > 0, time.perf_counter() - start, f"sequence={selection.sequence};observables=60;groups={len(selection.groups)};validation_accuracy={selection.validation_score:.6f}")

    logits = np.column_stack([-(xv[:, 0] - center) ** 2 for center in np.linspace(-2, 2, 10)])
    temperature = fit_temperature(logits, yv, split="validation"); calibrated = calibrated_probabilities(logits, temperature)
    record(rows, "calibration", np.isfinite(temperature) and temperature > 0 and np.allclose(calibrated.sum(1), 1), 0, f"temperature={temperature:.8g}")
    record(rows, "resource_counters", True, 0, f"vqc_parameters={parameter_count(VQCDR(10,42))};kernel_entries={len(x)**2 + len(xv)*len(x)};real_qpu_shots=0")
    record(rows, "nested_manifests", set(ids).isdisjoint(set(val_ids)), 0, f"train={len(ids)};validation={len(val_ids)};test_accessed=false")

    failures = [{"phase": "phase3", "method": row["method"], "failure_code": "SMOKE_FAILED", "detail": row["detail"]} for row in rows if row["status"] != "PASS"]
    raw_dir = ROOT / "results/raw"; raw_dir.mkdir(parents=True, exist_ok=True); failure_dir = ROOT / "results/failures"; failure_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(raw_dir / "phase3_smoke_results.csv", rows, ["method", "status", "wall_seconds", "detail"])
    _write_csv(failure_dir / "phase3_failures.csv", failures, ["phase", "method", "failure_code", "detail"])
    status = "PASS" if not failures else "FAIL"
    summary = ROOT / "summary/PHASE3_SMOKE_STATUS.md"; summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(f"# Phase 3 smoke status\n\nStatus: **{status}**\n\nOfficial test access: **none**.\n\nAll fourteen method families were exercised on a balanced MNIST training/validation development subset. See `results/raw/phase3_smoke_results.csv`.\n", encoding="utf-8")
    print(json.dumps({"status": status, "rows": rows}, indent=2))


def _manifest_validation_ids(dataset: str) -> np.ndarray:
    path = ROOT / "data_manifests/split_manifests" / f"{dataset}_splits.csv"
    with path.open(newline="") as handle: return np.asarray([int(r["sample_id"]) for r in csv.DictReader(handle) if r["split"] == "validation"])


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
