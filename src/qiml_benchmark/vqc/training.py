"""Checkpointable deterministic training and the frozen VQC sanity gate."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from .model import VQCDR, YomoMatched


def train(model: torch.nn.Module, angles: np.ndarray, labels: np.ndarray, *, steps: int, learning_rate: float, checkpoint: Path | None = None, resume: Path | None = None) -> dict[str, object]:
    device = next(model.parameters()).device; x = torch.as_tensor(angles, dtype=torch.float32, device=device); y = torch.as_tensor(labels, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate); losses = []; gradients = []; initial_step = 0
    if resume is not None:
        saved = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); initial_step = int(saved["step"])
        np.random.set_state(saved["numpy_rng"]); torch.set_rng_state(saved["torch_cpu_rng"])
        if torch.cuda.is_available() and saved["torch_cuda_rng"] is not None: torch.cuda.set_rng_state_all(saved["torch_cuda_rng"])
    for step in range(initial_step, steps):
        optimizer.zero_grad(set_to_none=True); output = model(x)
        loss = model.loss(output, y) if isinstance(model, YomoMatched) else torch.nn.functional.cross_entropy(output, y)
        if not torch.isfinite(loss): raise FloatingPointError("non-finite training loss")
        loss.backward(); norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in model.parameters() if p.grad is not None)); gradients.append(float(norm.detach())); optimizer.step(); losses.append(float(loss.detach()))
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": steps, "numpy_rng": np.random.get_state(), "torch_cpu_rng": torch.get_rng_state(), "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, checkpoint)
    with torch.no_grad():
        output = model(x); prediction = output.argmax(1).cpu().numpy()
    first = losses[0] if losses else float("nan"); last = losses[-1] if losses else float("nan")
    return {"loss_start": first, "loss_end": last, "relative_loss_reduction": (first - last) / first if losses else float("nan"), "accuracy": float(np.mean(prediction == labels)), "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)), "median_gradient_norm": float(np.median(gradients)) if gradients else float("nan"), "nonzero_gradients": bool(np.all(np.asarray(gradients) > 0)) if gradients else True, "steps": steps, "initial_step": initial_step}


def sanity_gate(angles: np.ndarray, labels: np.ndarray, classes: int, seed: int, device: torch.device, checkpoint: Path) -> dict[str, object]:
    model = VQCDR(classes, seed).to(device); outcome = train(model, angles, labels, steps=300, learning_rate=.03, checkpoint=checkpoint)
    required_accuracy = .5 if classes == 10 else None; required_balanced = .75 if classes == 2 else None
    outcome["passed"] = bool(outcome["relative_loss_reduction"] >= .30 and outcome["median_gradient_norm"] > 1e-8 and outcome["nonzero_gradients"] and (required_accuracy is None or outcome["accuracy"] >= required_accuracy) and (required_balanced is None or outcome["balanced_accuracy"] >= required_balanced))
    return outcome
