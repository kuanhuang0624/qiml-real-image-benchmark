"""Small, explicitly non-paper workflow using the released scientific core."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

from qiml_benchmark.calibration.temperature import fit_temperature, probabilities
from qiml_benchmark.circuits.statevector import apply_upload_sequence, product_or_ring
from qiml_benchmark.features.interface import AngleInterface
from qiml_benchmark.measurements.pauli import readout_48
from qiml_benchmark.pdrqc.model import select, feature_matrix
from qiml_benchmark.protocol import digest


def validate_arrays(x, y):
    if x.ndim != 2 or x.shape[1] != 512 or not np.isfinite(x).all():
        raise ValueError('features must be finite [N,512] arrays')
    if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('labels must be integer [N] arrays')


def feature_experiment(train_val: Path, test: Path, output: Path, *, demo: bool = False):
    output.mkdir(parents=True, exist_ok=False)
    with np.load(train_val, allow_pickle=False) as data:
        x, y, xv, yv = [data[key] for key in ('train_features', 'train_labels', 'val_features', 'val_labels')]
    validate_arrays(x, y); validate_arrays(xv, yv)
    classes = len(np.unique(y))
    if classes not in (2, 10) or set(y) != set(range(classes)) or not set(yv) <= set(y):
        raise ValueError('This benchmark supports contiguous 2- or 10-class labels')
    interface = AngleInterface().fit(x)
    a, av = interface.transform(x), interface.transform(xv)
    selection = select(a, y, av, yv, classes=classes, max_depth=1 if demo else 3)

    def representation(method, angles):
        if method == 'PdrQC-Matched':
            return feature_matrix(apply_upload_sequence(angles, selection.sequence), selection.paulis)
        return readout_48(product_or_ring(angles, method == 'QF-Ring'))

    models, records = {}, {}
    arrays = interface.arrays()
    for method in ('PdrQC-Matched', 'QF-Product', 'QF-Ring'):
        model = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(representation(method, a), y)
        scores = model.decision_function(representation(method, av))
        temperature = fit_temperature(scores, yv, split='validation')
        records[method] = {'C': 1, 'seed': 42, 'temperature': temperature,
                           'validation_accuracy': float(np.mean(probabilities(scores, temperature).argmax(1) == yv))}
        models[method] = model
        arrays[method + '_coef'] = model.coef_; arrays[method + '_intercept'] = model.intercept_
    np.savez_compressed(output / 'fitted_arrays.npz', **arrays)
    lock = {'kind': 'illustrative-workflow-not-paper-results', 'synthetic': demo,
            'training_input_sha256': digest(train_val), 'fitted_arrays_sha256': digest(output / 'fitted_arrays.npz'),
            'pdr_selection': asdict(selection), 'models': records, 'test_arrays_opened': False}
    (output / 'validation_lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    (output / 'validation_lock.sha256').write_text(digest(output / 'validation_lock.json') + '  validation_lock.json\n')
    # Test arrays are first opened AFTER fitting, calibration and the saved lock.
    with np.load(test, allow_pickle=False) as data:
        xt, yt = data['test_features'], data['test_labels']
    validate_arrays(xt, yt)
    if not set(yt) <= set(y): raise ValueError('Unseen test label')
    at = interface.transform(xt)
    results = {}
    for method, model in models.items():
        prob = probabilities(model.decision_function(representation(method, at)), records[method]['temperature'])
        results[method] = {'test_accuracy': float(np.mean(prob.argmax(1) == yt))}
        np.savez_compressed(output / (method + '_predictions.npz'), probability=prob, label=yt)
    result = {'status': 'DEMO_ONLY' if demo else 'NEW_FIXED_C_FEATURE_EXPERIMENT', 'results': results,
              'test_input_sha256': digest(test), 'warning': 'Not a rerun of the full frozen benchmark or a new manuscript result.'}
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def demo(output: Path):
    # A separate input folder and new output directory make repeated runs explicit.
    inputs = output.parent / (output.name + '_synthetic_inputs')
    inputs.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    direction = rng.normal(size=512).astype(np.float32)
    def sample(n):
        y = np.arange(n, dtype=np.int64) % 2
        return (rng.normal(size=(n, 512)) + (y[:, None] * 2 - 1) * direction * .3).astype(np.float32), y
    x, y = sample(64); xv, yv = sample(24); xt, yt = sample(24)
    np.savez_compressed(inputs / 'train_val.npz', train_features=x, train_labels=y, val_features=xv, val_labels=yv)
    np.savez_compressed(inputs / 'test.npz', test_features=xt, test_labels=yt)
    return feature_experiment(inputs / 'train_val.npz', inputs / 'test.npz', output, demo=True)
