"""Pauli expectations, matched 48-vector, QWC grouping, and offline sampling."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from qiml_benchmark.circuits.statevector import H, SDG, N_QUBITS, RING, apply_single, probabilities


def expectation(state: np.ndarray, pauli: str) -> np.ndarray:
    if len(pauli) != N_QUBITS:
        raise ValueError("Pauli string must have length 8")
    dim = state.shape[1]; idx = np.arange(dim); flipped = idx.copy(); phase = np.ones(dim, complex)
    for q, axis in enumerate(pauli):
        bitpos = N_QUBITS - 1 - q; bit = (idx >> bitpos) & 1
        if axis == "X": flipped ^= 1 << bitpos
        elif axis == "Y":
            flipped ^= 1 << bitpos; phase *= -1j * (1 - 2 * bit)
        elif axis == "Z": phase *= 1 - 2 * bit
        elif axis != "I": raise ValueError(axis)
    return np.real(np.sum(np.conj(state) * (phase[None, :] * state[:, flipped]), axis=1))


def readout_48(state: np.ndarray) -> np.ndarray:
    strings = []
    for axis in "XYZ":
        for q in range(N_QUBITS):
            p = ["I"] * N_QUBITS; p[q] = axis; strings.append("".join(p))
    for axis in "XYZ":
        for q0, q1 in RING:
            p = ["I"] * N_QUBITS; p[q0] = axis; p[q1] = axis; strings.append("".join(p))
    return np.column_stack([expectation(state, p) for p in strings])


def qwc_groups(paulis: Sequence[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    for pauli in paulis:
        for group in groups:
            if all(a == "I" or b == "I" or a == b for existing in group for a, b in zip(pauli, existing)):
                group.append(pauli); break
        else:
            groups.append([pauli])
    return groups


def _basis_probabilities(state: np.ndarray, axis: str) -> np.ndarray:
    rotated = state
    if axis == "X":
        for q in range(N_QUBITS): rotated = apply_single(rotated, H, q)
    elif axis == "Y":
        for q in range(N_QUBITS):
            rotated = apply_single(rotated, SDG, q); rotated = apply_single(rotated, H, q)
    elif axis != "Z": raise ValueError(axis)
    return probabilities(rotated)


def finite_readout_48(state: np.ndarray, budget: int, seed: int) -> np.ndarray:
    if budget not in (1, 8, 32, 128, 512): raise ValueError(budget)
    allocation = {"X": 0, "Y": 0, "Z": 0}
    if budget == 1: allocation["Z"] = 1
    else:
        base, remainder = divmod(budget, 3); order = list("XYZ")
        shift = seed % 3; order = order[shift:] + order[:shift]
        allocation = {axis: base + int(axis in order[:remainder]) for axis in "XYZ"}
    rng = np.random.default_rng(seed); one_features = []; pair_features = []
    for axis in "XYZ":
        shots = allocation[axis]
        if shots == 0:
            one_features.extend([np.zeros(len(state)) for _ in range(8)])
            pair_features.extend([np.zeros(len(state)) for _ in range(8)])
            continue
        probs = _basis_probabilities(state, axis)
        values = np.empty((len(state), 16))
        indices = np.arange(1 << N_QUBITS)
        one = np.column_stack([1 - 2 * ((indices >> (N_QUBITS - 1 - q)) & 1) for q in range(N_QUBITS)])
        pair = np.column_stack([one[:, q0] * one[:, q1] for q0, q1 in RING])
        obs = np.column_stack([one, pair])
        for i, p in enumerate(probs): values[i] = rng.multinomial(shots, p) @ obs / shots
        one_features.extend(values[:, :8].T)
        pair_features.extend(values[:, 8:].T)
    return np.column_stack(one_features + pair_features)
