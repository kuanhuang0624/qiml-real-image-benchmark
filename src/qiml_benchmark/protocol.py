"""Prospective lock for NEW full reproductions; never impersonates historical locks."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

METHODS = {'C-LR', 'C-MLP', 'C-Poly', 'C-RBF', 'C-RFF', 'C-Trig', 'C-ResNet',
           'C-CNN', 'QK-IQP', 'QF-Product', 'QF-Ring', 'VQC-DR', 'Yomo-Matched', 'PdrQC-Matched'}
STOCHASTIC = {'C-MLP', 'C-RFF', 'C-CNN', 'VQC-DR', 'Yomo-Matched'}
SCALES = {d: ('D_1k', 'D_5k', 'D_10k', 'D_full') for d in ('mnist', 'fashion_mnist', 'cifar10')}
SCALES['breastmnist'] = ('D_25pct', 'D_50pct', 'D_100pct')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*.py'))}


def check_selection(selected: dict, dataset: str, scale: str) -> None:
    if set(selected) != METHODS:
        raise RuntimeError(f'Incomplete method set: {dataset}/{scale}')
    for method, item in selected.items():
        expected_boundary = None
        if dataset != 'breastmnist' and scale == 'D_full':
            if method == 'QK-IQP': expected_boundary = 'QUADRATIC_SCALING_BOUNDARY'
            if method in ('C-Poly', 'C-RBF'): expected_boundary = 'CLASSICAL_KERNEL_SCALING_BOUNDARY'
        if expected_boundary:
            if item['status'] != expected_boundary:
                raise RuntimeError(f'Missing declared boundary: {method}')
            continue
        seeds = [42, 2026, 3407] if method in STOCHASTIC else [42]
        if item['status'] != 'FROZEN' or sorted(item['model_seeds']) != seeds:
            raise RuntimeError(f'Incomplete selection/seeds: {dataset}/{scale}/{method}')
        if set(item['temperatures']) != set(map(str, seeds)):
            raise RuntimeError('Incomplete validation temperatures')
        if dataset == 'breastmnist' and set(item.get('validation_selected_thresholds', {})) != set(map(str, seeds)):
            raise RuntimeError('BreastMNIST validation thresholds are not frozen')


def lock(root: Path) -> dict:
    if (root / 'REPRODUCTION_LOCK.json').exists() or (root / 'TEST_ACCESS_LOG.md').exists():
        raise RuntimeError('Refusing to overwrite a lock or relock after test access')
    if any((root / name).exists() for name in ('cache/pca_test_features', 'cache/resnet_test_features', 'predictions/test')):
        raise RuntimeError('Test artifacts already exist; use a new prospective workspace')
    folder = root / 'configs/frozen'
    final = json.loads((folder / 'FINAL_CONFIGS.yaml').read_text())
    scaling = json.loads((folder / 'DATA_SCALE_CONFIGS.yaml').read_text())
    if final.get('test_accessed') is not False or scaling.get('test_accessed') is not False:
        raise RuntimeError('Selections must declare no test access')
    for dataset, scales in SCALES.items():
        check_selection(final['datasets'][dataset], dataset, 'D_100pct' if dataset == 'breastmnist' else 'D_5k')
        for scale in scales:
            check_selection(scaling['datasets'][dataset][scale], dataset, scale)
    contrasts = json.loads((folder / 'PRIMARY_CONTRASTS.yaml').read_text())
    if len(contrasts['contrasts']) != 12:
        raise RuntimeError('Expected the declared twelve-contrast primary family')
    paths = list(folder.glob('*.yaml')) + [root / 'configs/method_registry.yaml']
    paths += sorted((root / 'data_manifests').rglob('*.csv'))
    for directory, suffix in (('cache/pca_features', '*.npz'), ('cache/resnet_features', '*.npz'),
                              ('results/raw/pdr_selections', '*.json'), ('checkpoints/validation', '*.pt')):
        found = sorted((root / directory).glob(suffix))
        if not found: raise RuntimeError(f'Required artifacts missing: {directory}')
        paths += found
    # Check every selected checkpoint, not just whether its directory is nonempty.
    for dataset, scales in scaling['datasets'].items():
        for scale, methods in scales.items():
            for method, item in methods.items():
                if item['status'] != 'FROZEN': continue
                if method in ('VQC-DR', 'Yomo-Matched', 'C-CNN'):
                    checkpoints = item.get('checkpoints', [])
                    if len(checkpoints) != 3 or any(not (root / p).is_file() for p in checkpoints):
                        raise RuntimeError(f'Missing selected checkpoints: {dataset}/{scale}/{method}')
    value = {'kind': 'new-reproduction-lock-v1', 'utc': datetime.now(timezone.utc).isoformat(),
             'files': {str(p.relative_to(root)): digest(p) for p in sorted(set(paths))},
             'package_sources': source_hashes(), 'test_accessed': False,
             'primary_family': 12, 'uncertainty_family': 6,
             'note': 'A new run lock, not an amendment or replacement of the historical locks.'}
    path = root / 'REPRODUCTION_LOCK.json'
    with path.open('x') as handle: json.dump(value, handle, indent=2, sort_keys=True); handle.write('\n')
    with (root / 'REPRODUCTION_LOCK.sha256').open('x') as handle: handle.write(digest(path) + '  REPRODUCTION_LOCK.json\n')
    return value


def verify(root: Path) -> None:
    path = root / 'REPRODUCTION_LOCK.json'
    expected = (root / 'REPRODUCTION_LOCK.sha256').read_text().split()[0]
    if digest(path) != expected: raise RuntimeError('Protocol lock hash mismatch')
    value = json.loads(path.read_text())
    if value['kind'] != 'new-reproduction-lock-v1' or value['test_accessed'] is not False:
        raise RuntimeError('Invalid protocol lock')
    for relative, expected in value['files'].items():
        target = root / relative
        if not target.resolve().is_relative_to(root.resolve()) or digest(target) != expected:
            raise RuntimeError(f'Locked input changed: {relative}')
    if value['package_sources'] != source_hashes():
        raise RuntimeError('Source changed after lock')
