"""Keep mutable experiment artifacts out of both installed code and references."""
import json
import os
from pathlib import Path

MARKER = '.qiml-workspace.json'


def workspace_root() -> Path:
    root = Path(os.environ.get('QIML_WORKDIR', Path.cwd() / 'runs/workspace')).expanduser().resolve()
    if root.exists():
        if not (root / MARKER).is_file():
            raise RuntimeError(f'Not an initialized release workspace: {root}. Use qiml init on a NEW directory.')
        if json.loads((root / MARKER).read_text()).get('format') != 'qiml-release-workspace-v1':
            raise RuntimeError('Unsupported workspace marker')
    return root


def initialize(root: Path) -> Path:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / MARKER).write_text(json.dumps({'format': 'qiml-release-workspace-v1',
        'purpose': 'new reproduction, not the original historical experiment'}, indent=2) + '\n')
    for relative in ('data/raw', 'data_manifests/feature_manifests', 'data_manifests/split_manifests',
                     'data_manifests/nested_training_manifests', 'cache/pca_features',
                     'configs/frozen', 'results/raw', 'results/aggregates', 'results/statistics',
                     'results/tables', 'results/resources', 'results/failures', 'summary',
                     'figures/png', 'figures/pdf', 'figures/svg', 'figures/source_data'):
        (root / relative).mkdir(parents=True, exist_ok=True)
    return root
