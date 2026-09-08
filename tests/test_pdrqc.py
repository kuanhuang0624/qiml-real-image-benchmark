import numpy as np

from qiml_benchmark.circuits.statevector import apply_upload_sequence
from qiml_benchmark.pdrqc.model import grouped_finite_features, select


def test_pdr_selection_and_grouped_sampling() -> None:
    rng = np.random.default_rng(11); x = rng.normal(size=(12, 8)); y = np.arange(12) % 2
    selection = select(x, y, x, y, classes=2, max_depth=1)
    assert len(selection.paulis) == 28 and sum(map(len, selection.groups)) == 28
    sampled = grouped_finite_features(apply_upload_sequence(x, selection.sequence), selection.paulis, selection.groups, max(32, len(selection.groups)), 101)
    assert sampled.shape == (12, 28) and np.isfinite(sampled).all()
