"""Validation-only PdrQC-Matched Pauli selection and grouped finite-shot readout."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from qiml_benchmark.circuits.statevector import N_QUBITS, apply_single, apply_upload_sequence, pauli_strings, probabilities, H, SDG
from qiml_benchmark.measurements.pauli import expectation, qwc_groups


ATOMIC_BLOCKS = ("RX", "RY", "RZ", "CZ_RING", "CNOT_RING")


@dataclass(frozen=True)
class Selection:
    sequence: tuple[str, ...]
    paulis: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]
    validation_score: float


def feature_matrix(state: np.ndarray, paulis: Sequence[str]) -> np.ndarray:
    return np.column_stack([expectation(state, p) for p in paulis])


def discrimination_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Between-class variance divided by pooled within-class variance."""
    global_mean = features.mean(0); between = np.zeros(features.shape[1]); within = np.zeros(features.shape[1])
    for label in np.unique(labels):
        block = features[labels == label]; between += len(block) * (block.mean(0) - global_mean) ** 2
        within += ((block - block.mean(0)) ** 2).sum(0)
    return between / np.maximum(within, 1e-12)


def select(angles: np.ndarray, labels: np.ndarray, val_angles: np.ndarray, val_labels: np.ndarray, *, classes: int, max_depth: int = 3, seed: int = 42) -> Selection:
    pool = pauli_strings(2); keep = 60 if classes == 10 else 28
    winner: Selection | None = None; prefixes: list[tuple[str, ...]] = [()]
    for _depth in range(1, max_depth + 1):
        candidates: list[Selection] = []
        for prefix in prefixes:
            for block in ATOMIC_BLOCKS:
                sequence = prefix + (block,); train_state = apply_upload_sequence(angles, sequence)
                scores = discrimination_scores(feature_matrix(train_state, pool), labels)
                chosen = tuple(pool[i] for i in np.argsort(scores)[::-1][:keep])
                train_x = feature_matrix(train_state, chosen); val_x = feature_matrix(apply_upload_sequence(val_angles, sequence), chosen)
                clf = LogisticRegression(C=1, max_iter=1000, random_state=seed).fit(train_x, labels)
                score = float(np.mean(clf.predict(val_x) == val_labels))
                candidates.append(Selection(sequence, chosen, tuple(tuple(g) for g in qwc_groups(chosen)), score))
        current = max(candidates, key=lambda x: (x.validation_score, -len(x.groups), tuple(x.sequence)))
        if winner is None or current.validation_score > winner.validation_score + .002: winner = current
        prefixes = [current.sequence]
    assert winner is not None
    return winner


def _group_basis(group: Sequence[str]) -> str:
    basis = []
    for q in range(N_QUBITS):
        axes = {p[q] for p in group if p[q] != "I"}; basis.append(next(iter(axes)) if axes else "Z")
    return "".join(basis)


def _basis_probabilities(state: np.ndarray, basis: str) -> np.ndarray:
    rotated = state
    for q, axis in enumerate(basis):
        if axis == "X": rotated = apply_single(rotated, H, q)
        elif axis == "Y": rotated = apply_single(rotated, SDG, q); rotated = apply_single(rotated, H, q)
    return probabilities(rotated)


def grouped_finite_features(state: np.ndarray, paulis: Sequence[str], groups: Sequence[Sequence[str]], budget: int, seed: int) -> np.ndarray:
    if budget < len(groups): raise ValueError("budget must provide at least one shot per QWC group")
    rng = np.random.default_rng(seed); base, remainder = divmod(budget, len(groups)); allocation = [base] * len(groups)
    for offset in range(remainder): allocation[(seed + offset) % len(groups)] += 1
    out = np.empty((len(state), len(paulis))); column = {p: i for i, p in enumerate(paulis)}; indices = np.arange(1 << N_QUBITS)
    bits = np.column_stack([1 - 2 * ((indices >> (N_QUBITS - 1 - q)) & 1) for q in range(N_QUBITS)])
    for group, shots in zip(groups, allocation):
        prob = _basis_probabilities(state, _group_basis(group))
        observable = np.column_stack([np.prod(bits[:, [q for q, axis in enumerate(p) if axis != "I"]], axis=1) for p in group])
        for row, p in enumerate(prob): out[row, [column[x] for x in group]] = rng.multinomial(shots, p) @ observable / shots
    return out
