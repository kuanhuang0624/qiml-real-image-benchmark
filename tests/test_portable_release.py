import json
from pathlib import Path
import numpy as np
import pytest
from qiml_benchmark.circuits.statevector import X, H, SDG, apply_single, probabilities, zero_state
from qiml_benchmark.measurements.pauli import expectation, _basis_probabilities
from qiml_benchmark.features.interface import AngleInterface
from qiml_benchmark.paths import initialize, workspace_root
from qiml_benchmark.protocol import digest, verify, source_hashes
from qiml_benchmark.examples import demo
from qiml_benchmark.width.model import exact_ry_features


def test_msb_qubit_order():
    assert probabilities(apply_single(zero_state(1), X, 0)).argmax() == 128
    assert probabilities(apply_single(zero_state(1), X, 7)).argmax() == 1


@pytest.mark.parametrize('axis', ['X', 'Y', 'Z'])
def test_basis_rotations_and_expectations(axis):
    rng = np.random.default_rng(71)
    state = rng.normal(size=(3, 256)) + 1j * rng.normal(size=(3, 256))
    state /= np.linalg.norm(state, axis=1, keepdims=True)
    prob = _basis_probabilities(state, axis)
    for q in range(8):
        signs = 1 - 2 * ((np.arange(256) >> (7 - q)) & 1)
        p = ['I'] * 8; p[q] = axis
        np.testing.assert_allclose(prob @ signs, expectation(state, ''.join(p)), atol=1e-12)


def test_train_only_pca():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(50, 512)); val = rng.normal(size=(12, 512))
    interface = AngleInterface().fit(x)
    before = {k: v.copy() for k, v in interface.arrays().items()}
    a = interface.transform(val)
    assert a.shape == (12, 8) and np.max(np.abs(a)) <= np.pi + 1e-6
    interface.transform(val * 1e5)
    for k, v in before.items(): np.testing.assert_array_equal(v, interface.arrays()[k])
    with pytest.raises(ValueError): interface.fit(val, split='test')


def test_width_analytic_dense_agreement():
    from qiml_benchmark.circuits.statevector import apply_upload_sequence, pauli_strings
    rng = np.random.default_rng(2); angles = rng.normal(size=(4, 8))
    ps = pauli_strings(2)
    state = apply_upload_sequence(angles, ('RY',))
    dense = np.column_stack([expectation(state, p) for p in ps])
    np.testing.assert_allclose(exact_ry_features(angles, ps), dense, atol=1e-12)


def test_workspace_refuses_original_directory(tmp_path, monkeypatch):
    monkeypatch.setenv('QIML_WORKDIR', str(tmp_path))
    with pytest.raises(RuntimeError): workspace_root()
    new = initialize(tmp_path / 'new')
    monkeypatch.setenv('QIML_WORKDIR', str(new)); assert workspace_root() == new
    with pytest.raises(FileExistsError): initialize(new)


def test_lock_tamper_fails_closed(tmp_path):
    root = initialize(tmp_path / 'locked'); (root / 'input.json').write_text('{}')
    lock = {'kind': 'new-reproduction-lock-v1', 'test_accessed': False,
            'files': {'input.json': digest(root / 'input.json')}, 'package_sources': source_hashes()}
    path = root / 'REPRODUCTION_LOCK.json'; path.write_text(json.dumps(lock))
    (root / 'REPRODUCTION_LOCK.sha256').write_text(digest(path) + '  REPRODUCTION_LOCK.json\n')
    verify(root)
    (root / 'input.json').write_text('{"changed": true}')
    with pytest.raises(RuntimeError, match='Locked input changed'): verify(root)


def test_synthetic_workflow(tmp_path):
    output = tmp_path / 'demo'; result = demo(output)
    assert result['status'] == 'DEMO_ONLY'
    lock = json.loads((output / 'validation_lock.json').read_text())
    assert lock['test_arrays_opened'] is False
    assert len(result['results']) == 3
    for path in output.glob('*predictions.npz'):
        with np.load(path) as a: np.testing.assert_allclose(a['probability'].sum(1), 1)
