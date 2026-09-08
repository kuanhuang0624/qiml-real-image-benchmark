import numpy as np
import pytest
from qiml_benchmark.circuits.statevector import zero_state
from qiml_benchmark.measurements.pauli import finite_readout_48
from qiml_benchmark.pdrqc.model import grouped_finite_features


class RecordingRNG:
    def __init__(self): self.shots = []
    def multinomial(self, n, p):
        self.shots.append(n)
        out = np.zeros_like(p, dtype=int); out[np.argmax(p)] = n
        return out


@pytest.mark.parametrize('budget', [1, 8, 32, 128, 512])
def test_qf_total_not_per_group(monkeypatch, budget):
    rng = RecordingRNG(); monkeypatch.setattr(np.random, 'default_rng', lambda seed: rng)
    values = finite_readout_48(zero_state(1), budget, 101)
    assert sum(rng.shots) == budget
    assert len(rng.shots) == (1 if budget == 1 else 3)
    assert values.shape == (1, 48)


@pytest.mark.parametrize('budget', [8, 128, 512])
def test_pdr_total_not_per_group(monkeypatch, budget):
    paulis = ['XIIIIIII', 'YIIIIIII', 'ZIIIIIII']; groups = [[p] for p in paulis]
    rng = RecordingRNG(); monkeypatch.setattr(np.random, 'default_rng', lambda seed: rng)
    values = grouped_finite_features(zero_state(1), paulis, groups, budget, 101)
    assert sum(rng.shots) == budget and len(rng.shots) == 3
    assert max(rng.shots) - min(rng.shots) <= 1
    assert values.shape == (1, 3)


def test_insufficient_pdr_budget_rejected():
    ps = ['XIIIIIII', 'YIIIIIII']
    with pytest.raises(ValueError): grouped_finite_features(zero_state(1), ps, [[p] for p in ps], 1, 101)
