"""Frozen depth-one RY PdrQC selection with exact analytic expectations.

The parent selection is depth one with a product RY upload. For this state,
Pauli expectations factor into sin/cos terms, allowing exact 16-qubit
expectations without materializing large dense statevector batches.
"""
from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Selection:
    sequence: tuple[str, ...]
    paulis: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]
    candidate_count: int


def pauli_strings(qubits: int, max_weight: int = 2) -> list[str]:
    strings: list[str] = []
    for weight in range(1, max_weight + 1):
        for positions in itertools.combinations(range(qubits), weight):
            for axes in itertools.product("XYZ", repeat=weight):
                chars = ["I"] * qubits
                for position, axis in zip(positions, axes):
                    chars[position] = axis
                strings.append("".join(chars))
    return strings


def exact_ry_features(angles: np.ndarray, paulis: Sequence[str]) -> np.ndarray:
    angles = np.asarray(angles, dtype=float)
    out = np.empty((len(angles), len(paulis)), dtype=np.float64)
    sine, cosine = np.sin(angles), np.cos(angles)
    for column, pauli in enumerate(paulis):
        values = np.ones(len(angles), dtype=np.float64)
        for qubit, axis in enumerate(pauli):
            if axis == "X":
                values *= sine[:, qubit]
            elif axis == "Y":
                values.fill(0.0)
                break
            elif axis == "Z":
                values *= cosine[:, qubit]
            elif axis != "I":
                raise ValueError(axis)
        out[:, column] = values
    return out


def discrimination_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    global_mean = features.mean(0)
    between = np.zeros(features.shape[1])
    within = np.zeros(features.shape[1])
    for label in np.unique(labels):
        block = features[labels == label]
        between += len(block) * (block.mean(0) - global_mean) ** 2
        within += ((block - block.mean(0)) ** 2).sum(0)
    return between / np.maximum(within, 1e-12)


def qwc_groups(paulis: Sequence[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    for pauli in paulis:
        for group in groups:
            if all(a == "I" or b == "I" or a == b for existing in group for a, b in zip(pauli, existing)):
                group.append(pauli)
                break
        else:
            groups.append([pauli])
    return groups


def select(train_angles: np.ndarray, labels: np.ndarray, keep: int = 60) -> Selection:
    qubits = train_angles.shape[1]
    pool = pauli_strings(qubits, 2)
    scores = discrimination_scores(exact_ry_features(train_angles, pool), labels)
    chosen = tuple(pool[index] for index in np.argsort(scores)[::-1][:keep])
    groups = tuple(tuple(group) for group in qwc_groups(chosen))
    return Selection(("RY",), chosen, groups, len(pool))


def maximum_weight(paulis: Sequence[str]) -> int:
    return max(sum(axis != "I" for axis in pauli) for pauli in paulis)
