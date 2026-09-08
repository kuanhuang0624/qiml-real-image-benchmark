"""Independent small-batch dense statevector verifier for RY expectations."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def ry_state(angles: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles, dtype=float)
    states = []
    for row in angles:
        state = np.array([1.0 + 0j])
        for theta in row:
            state = np.kron(state, np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=np.complex128))
        states.append(state)
    return np.asarray(states)


def expectation(state: np.ndarray, pauli: str) -> np.ndarray:
    qubits = len(pauli)
    dim = state.shape[1]
    index = np.arange(dim)
    flipped = index.copy()
    phase = np.ones(dim, dtype=np.complex128)
    for qubit, axis in enumerate(pauli):
        bit_position = qubits - 1 - qubit
        bit = (index >> bit_position) & 1
        if axis == "X":
            flipped ^= 1 << bit_position
        elif axis == "Y":
            flipped ^= 1 << bit_position
            phase *= -1j * (1 - 2 * bit)
        elif axis == "Z":
            phase *= 1 - 2 * bit
        elif axis != "I":
            raise ValueError(axis)
    return np.real(np.sum(np.conj(state) * phase[None, :] * state[:, flipped], axis=1))


def feature_matrix(angles: np.ndarray, paulis: Sequence[str]) -> np.ndarray:
    state = ry_state(angles)
    return np.column_stack([expectation(state, pauli) for pauli in paulis])
