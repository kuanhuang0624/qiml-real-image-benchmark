import numpy as np

from qiml_benchmark.circuits.statevector import iqp, pauli_strings, probabilities, product_or_ring
from qiml_benchmark.measurements.pauli import finite_readout_48, qwc_groups, readout_48


def test_states_and_kernel() -> None:
    angles = np.random.default_rng(7).uniform(-np.pi, np.pi, size=(5, 8))
    for state in (product_or_ring(angles, False), product_or_ring(angles, True), iqp(angles)):
        assert np.allclose(probabilities(state).sum(1), 1)
    states = iqp(angles); kernel = np.abs(states @ states.conj().T) ** 2
    assert np.allclose(kernel, kernel.T) and np.allclose(np.diag(kernel), 1)
    assert np.linalg.eigvalsh(kernel).min() > -1e-10


def test_product_ring_and_readout() -> None:
    angles = np.random.default_rng(9).normal(size=(4, 8))
    product, ring = product_or_ring(angles, False), product_or_ring(angles, True)
    assert not np.allclose(product, ring)
    assert readout_48(product).shape == (4, 48)
    sampled = finite_readout_48(ring, 512, 101)
    assert sampled.shape == (4, 48) and np.isfinite(sampled).all()


def test_pauli_pool_and_groups() -> None:
    pool = pauli_strings(2)
    assert len(pool) == 276
    groups = qwc_groups(pool[:60])
    assert groups and sum(map(len, groups)) == 60


def test_y_expectation_sign() -> None:
    from qiml_benchmark.measurements.pauli import expectation
    plus_i = np.zeros((1, 256), complex)
    plus_i[0, 0] = 1 / np.sqrt(2); plus_i[0, 128] = 1j / np.sqrt(2)
    assert np.allclose(expectation(plus_i, "YIIIIIII"), 1)
