"""Vectorized NumPy statevector circuits using qubit axis 0 as the MSB."""
from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

N_QUBITS = 8
RING = tuple((q, (q + 1) % N_QUBITS) for q in range(N_QUBITS))
I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.diag([1, -1]).astype(np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
SDG = np.diag([1, -1j]).astype(np.complex128)


def ry(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta)
    out = np.empty(theta.shape + (2, 2), dtype=np.complex128)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    out[..., 0, 0] = c; out[..., 0, 1] = -s
    out[..., 1, 0] = s; out[..., 1, 1] = c
    return out


def rz(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta)
    out = np.zeros(theta.shape + (2, 2), dtype=np.complex128)
    out[..., 0, 0] = np.exp(-.5j * theta)
    out[..., 1, 1] = np.exp(.5j * theta)
    return out


def rx(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta)
    out = np.empty(theta.shape + (2, 2), dtype=np.complex128)
    c, s = np.cos(theta / 2), -1j * np.sin(theta / 2)
    out[..., 0, 0] = c; out[..., 0, 1] = s
    out[..., 1, 0] = s; out[..., 1, 1] = c
    return out


def zero_state(batch: int, qubits: int = N_QUBITS) -> np.ndarray:
    state = np.zeros((batch, 1 << qubits), dtype=np.complex128)
    state[:, 0] = 1
    return state


def apply_single(state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    """Apply constant (2,2) or per-sample (batch,2,2) gate."""
    batch, dim = state.shape
    qubits = int(np.log2(dim))
    shaped = state.reshape((batch,) + (2,) * qubits)
    moved = np.moveaxis(shaped, qubit + 1, -1)
    if gate.ndim == 2:
        moved = moved @ gate.T
    else:
        flat = moved.reshape(batch, -1, 2)
        moved = np.einsum("bni,bji->bnj", flat, gate).reshape(moved.shape)
    return np.moveaxis(moved, -1, qubit + 1).reshape(batch, dim)


def apply_cz(state: np.ndarray, control: int, target: int) -> np.ndarray:
    dim = state.shape[1]
    qubits = int(np.log2(dim))
    idx = np.arange(dim)
    mask = (((idx >> (qubits - 1 - control)) & 1) & ((idx >> (qubits - 1 - target)) & 1)).astype(bool)
    out = state.copy(); out[:, mask] *= -1
    return out


def apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    dim = state.shape[1]
    qubits = int(np.log2(dim))
    idx = np.arange(dim)
    cbit = (idx >> (qubits - 1 - control)) & 1
    perm = idx ^ (cbit << (qubits - 1 - target))
    return state[:, perm]


def apply_rzz(state: np.ndarray, theta: np.ndarray, q0: int, q1: int) -> np.ndarray:
    dim = state.shape[1]; qubits = int(np.log2(dim)); idx = np.arange(dim)
    z0 = 1 - 2 * ((idx >> (qubits - 1 - q0)) & 1)
    z1 = 1 - 2 * ((idx >> (qubits - 1 - q1)) & 1)
    return state * np.exp(-.5j * np.asarray(theta)[:, None] * (z0 * z1)[None, :])


def product_or_ring(angles: np.ndarray, ring: bool) -> np.ndarray:
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 2 or angles.shape[1] != N_QUBITS:
        raise ValueError("angles must have shape (n,8)")
    state = zero_state(len(angles))
    for _ in range(2):
        for q in range(N_QUBITS):
            state = apply_single(state, ry(angles[:, q]), q)
            state = apply_single(state, rz(angles[:, q]), q)
        if ring:
            for q0, q1 in RING:
                state = apply_cz(state, q0, q1)
    return state


def iqp(angles: np.ndarray) -> np.ndarray:
    """Two-repeat circular Havlíček-style H/RZ/RZZ feature map."""
    angles = np.asarray(angles, dtype=float)
    state = zero_state(len(angles))
    for _ in range(2):
        for q in range(N_QUBITS):
            state = apply_single(state, H, q)
            state = apply_single(state, rz(2 * angles[:, q]), q)
        for q0, q1 in RING:
            phase = 2 * (np.pi - angles[:, q0]) * (np.pi - angles[:, q1])
            state = apply_rzz(state, phase, q0, q1)
    return state


def apply_upload_sequence(angles: np.ndarray, sequence: Sequence[str]) -> np.ndarray:
    """PdrQC-Matched atomic upload block sequence."""
    angles = np.asarray(angles, dtype=float)
    state = zero_state(len(angles))
    for block in sequence:
        if block in {"RX", "RY", "RZ"}:
            fn = {"RX": rx, "RY": ry, "RZ": rz}[block]
            for q in range(N_QUBITS):
                state = apply_single(state, fn(angles[:, q]), q)
        elif block == "CZ_RING":
            for edge in RING: state = apply_cz(state, *edge)
        elif block == "CNOT_RING":
            for edge in RING: state = apply_cnot(state, *edge)
        else:
            raise ValueError(f"unsupported matched upload block {block}")
    return state


def probabilities(state: np.ndarray) -> np.ndarray:
    result = np.abs(state) ** 2
    residual = np.max(np.abs(result.sum(1) - 1))
    if residual > 1e-10:
        raise RuntimeError(f"state normalization residual {residual}")
    return result


def pauli_strings(max_weight: int = 2, qubits: int = N_QUBITS) -> list[str]:
    strings = []
    for weight in range(1, max_weight + 1):
        for positions in itertools.combinations(range(qubits), weight):
            for axes in itertools.product("XYZ", repeat=weight):
                chars = ["I"] * qubits
                for position, axis in zip(positions, axes): chars[position] = axis
                strings.append("".join(chars))
    return strings
