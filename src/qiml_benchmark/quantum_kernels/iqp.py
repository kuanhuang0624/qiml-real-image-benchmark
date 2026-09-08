"""Exact and offline finite-shot IQP fidelity kernels."""
from __future__ import annotations

import numpy as np

from qiml_benchmark.circuits.statevector import iqp


def exact_kernel(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    sa = iqp(a); sb = sa if b is None else iqp(b)
    return np.abs(sa @ sb.conj().T) ** 2


def finite_kernel(exact: np.ndarray, shots: int, seed: int, training: bool) -> tuple[np.ndarray, dict[str, float | int]]:
    if shots not in (32, 128, 512): raise ValueError(shots)
    rng = np.random.default_rng(seed)
    sampled = rng.binomial(shots, np.clip(exact, 0, 1)) / shots
    if not training: return sampled, {}
    sampled = (sampled + sampled.T) / 2; np.fill_diagonal(sampled, 1)
    eigval, eigvec = np.linalg.eigh(sampled); negative = eigval < 0
    corrected = (eigvec * np.maximum(eigval, 0)) @ eigvec.T
    diagonal = np.sqrt(np.maximum(np.diag(corrected), 1e-12)); corrected /= diagonal[:, None] * diagonal[None, :]
    return corrected, {"minimum_eigenvalue": float(eigval.min()), "negative_eigenvalues": int(negative.sum()), "negative_mass": float(-eigval[negative].sum()), "frobenius_correction": float(np.linalg.norm(corrected - sampled))}
