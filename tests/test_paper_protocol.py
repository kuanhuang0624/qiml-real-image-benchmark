"""Regression checks for the paper's shared transform and fixed comparator."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from qiml_benchmark.features.interface import transform_saved
from qiml_benchmark.validation.freeze import build_contrasts


def test_saved_transform_applies_pca_center(tmp_path):
    rng = np.random.default_rng(7)
    values = rng.normal(size=(40, 12)) + np.arange(12)
    pca = PCA(8, svd_solver='full').fit(values)
    quantiles = np.quantile(np.abs(pca.transform(values)), .99, axis=0)
    path = tmp_path / 'preprocessor.npz'
    np.savez(path, mean=np.zeros(12), std=np.ones(12), pca_mean=pca.mean_,
             components=pca.components_, quantiles=quantiles)
    held_out = rng.normal(size=(10, 12)) + np.arange(12)
    expected = np.pi * np.clip(pca.transform(held_out) / quantiles, -1, 1)
    np.testing.assert_allclose(transform_saved(held_out, path), expected, atol=2e-7)


def test_cache_builder_saves_training_transform(tmp_path, monkeypatch):
    from qiml_benchmark.features import fit_shared_interfaces as builder
    monkeypatch.setattr(builder, 'OUT', tmp_path)
    rng = np.random.default_rng(19)
    train = rng.normal(size=(40, 16))
    val = rng.normal(size=(12, 16))
    labels = np.arange(40) % 2
    builder.fit_one('example', 'D_5k', train, labels, val,
                    np.arange(12) % 2, np.arange(40))
    path = tmp_path / 'example_D_5k_preprocessor.npz'
    standard = StandardScaler().fit_transform(train)
    pca = PCA(8, svd_solver='randomized', random_state=20260812).fit(standard)
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved['pca_mean'], pca.mean_)
    with np.load(tmp_path / 'example_D_5k.npz') as fitted:
        np.testing.assert_allclose(transform_saved(val, path), fitted['val_angles'], atol=2e-7)


def test_legacy_transform_retains_original_values(tmp_path):
    rng = np.random.default_rng(23)
    values = rng.normal(size=(4, 12))
    arrays = dict(mean=rng.normal(size=12), std=np.ones(12),
                  components=rng.normal(size=(8, 12)), quantiles=np.full(8, 10.0))
    path = tmp_path / 'legacy.npz'
    np.savez(path, **arrays)
    z = ((values - arrays['mean']) / arrays['std']) @ arrays['components'].T
    expected = (np.pi * np.clip(z / arrays['quantiles'], -1, 1)).astype(np.float32)
    np.testing.assert_array_equal(transform_saved(values, path), expected)


def reference_selection():
    root = Path(__file__).resolve().parents[1]
    path = root / 'experiments/reference_configs/benchmark/frozen/FINAL_CONFIGS.yaml'
    return json.loads(path.read_text())['datasets']


def test_primary_comparator_stays_rbf_when_other_control_wins():
    selected = reference_selection()
    for methods in selected.values():
        methods['C-Trig']['primary_metric_mean'] = 1.0
    contrasts = build_contrasts(selected)
    assert len(contrasts) == 12
    primary = [row for row in contrasts if row['id'].startswith('H1_')]
    assert len(primary) == 3
    assert all(row['method_B'] == 'C-RBF' for row in primary)
    assert all(row['method_A'] == 'PdrQC-Matched' for row in primary)
    changed = copy.deepcopy(selected)
    changed['fashion_mnist']['QF-Product']['primary_metric_mean'] = 1.0
    assert build_contrasts(changed)[0]['method_A'] == 'QF-Product'


def test_missing_rbf_comparator_fails_before_test_access():
    selected = reference_selection()
    selected['fashion_mnist']['C-RBF']['status'] = 'NOT_COMPLETED'
    with pytest.raises(RuntimeError, match='C-RBF comparator'):
        build_contrasts(selected)
