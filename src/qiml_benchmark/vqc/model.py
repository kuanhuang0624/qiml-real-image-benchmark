"""Differentiable two-block eight-qubit data-reuploading models."""
from __future__ import annotations

import torch
from torch import nn

QUBITS = 8
RING = tuple((q, (q + 1) % QUBITS) for q in range(QUBITS))


def _ry(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex64)


def _rz(theta: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(theta, dtype=torch.complex64)
    return torch.stack([torch.stack([torch.exp(-.5j * theta), z], -1), torch.stack([z, torch.exp(.5j * theta)], -1)], -2)


def _rx(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta / 2).to(torch.complex64); s = -1j * torch.sin(theta / 2)
    return torch.stack([torch.stack([c, s], -1), torch.stack([s, c], -1)], -2)


def _single(state: torch.Tensor, gate: torch.Tensor, qubit: int) -> torch.Tensor:
    batch, dim = state.shape; shaped = state.reshape((batch,) + (2,) * QUBITS); moved = torch.movedim(shaped, qubit + 1, -1)
    flat = moved.reshape(batch, -1, 2)
    if gate.ndim == 2: output = flat @ gate.T
    else: output = torch.einsum("bni,bji->bnj", flat, gate)
    return torch.movedim(output.reshape(moved.shape), -1, qubit + 1).reshape(batch, dim)


def _cnot(state: torch.Tensor, control: int, target: int) -> torch.Tensor:
    idx = torch.arange(1 << QUBITS, device=state.device); bit = (idx >> (QUBITS - 1 - control)) & 1
    perm = idx ^ (bit << (QUBITS - 1 - target)); return state[:, perm]


class ReuploadingState(nn.Module):
    def __init__(self, seed: int):
        super().__init__(); generator = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(.05 * torch.randn(2, QUBITS, 3, generator=generator))

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        angles = angles.to(dtype=torch.float32); state = torch.zeros((len(angles), 1 << QUBITS), dtype=torch.complex64, device=angles.device); state[:, 0] = 1
        for block in range(2):
            for q in range(QUBITS):
                state = _single(state, _ry(angles[:, q]), q); state = _single(state, _rz(angles[:, q]), q)
                state = _single(state, _rx(self.theta[block, q, 0]), q); state = _single(state, _ry(self.theta[block, q, 1]), q); state = _single(state, _rz(self.theta[block, q, 2]), q)
            for control, target in RING: state = _cnot(state, control, target)
        return state


def z_expectations(state: torch.Tensor) -> torch.Tensor:
    idx = torch.arange(1 << QUBITS, device=state.device); probs = torch.abs(state) ** 2
    signs = torch.stack([1 - 2 * ((idx >> (QUBITS - 1 - q)) & 1) for q in range(QUBITS)], 1).to(probs.dtype)
    return probs @ signs


class VQCDR(nn.Module):
    def __init__(self, classes: int, seed: int):
        super().__init__(); self.quantum = ReuploadingState(seed); self.head = nn.Linear(QUBITS, classes)
        torch.manual_seed(seed); nn.init.normal_(self.head.weight, std=.05); nn.init.zeros_(self.head.bias)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        return self.head(z_expectations(self.quantum(angles)))


class YomoMatched(nn.Module):
    def __init__(self, classes: int, seed: int):
        super().__init__(); self.quantum = ReuploadingState(seed); self.classes = classes
        sizes = [(1 << QUBITS) // classes + int(k < (1 << QUBITS) % classes) for k in range(classes)]
        assignment = torch.empty(1 << QUBITS, dtype=torch.long); start = 0
        for k, size in enumerate(sizes): assignment[start:start + size] = k; start += size
        self.register_buffer("assignment", assignment); self.register_buffer("group_sizes", torch.tensor(sizes, dtype=torch.float32))

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        probs = torch.abs(self.quantum(angles)) ** 2; aggregated = torch.zeros((len(angles), self.classes), device=angles.device)
        aggregated.scatter_add_(1, self.assignment.expand(len(angles), -1), probs)
        averaged = aggregated / self.group_sizes
        return averaged / averaged.sum(1, keepdim=True)

    def loss(self, probability: torch.Tensor, labels: torch.Tensor, gamma: float = .1, omega: float = .1, tau: float = .6) -> torch.Tensor:
        correct = probability[torch.arange(len(labels), device=labels.device), labels]
        ce = -torch.log(correct.clamp_min(1e-12)).mean(); confident = probability.max(1).values
        selected = confident[confident > tau]; sharpening = 1 - selected.mean() if len(selected) else torch.ones((), device=labels.device)
        entropy = -(probability * torch.log(probability.clamp_min(1e-12))).sum(1).mean()
        return ce + gamma * sharpening + omega * entropy


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
