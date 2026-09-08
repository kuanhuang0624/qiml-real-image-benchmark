"""Recompute historical bootstrap/permutation results in a NEW workspace.

Requires the author's separately held prediction artifacts. Does not train,
perform fresh test inference, or write to the source artifact directory.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from qiml_benchmark.paths import initialize
from qiml_benchmark.protocol import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--workdir', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(); source = args.artifacts.resolve()
    metadata = ['results/raw/exact_test_runs.csv', 'results/raw/finite_shot_runs.csv',
                'results/raw/data_scaling_runs.csv', 'results/aggregates/data_scaling_performance.csv',
                'results/aggregates/ranking_stability.csv', 'configs/frozen/PRIMARY_CONTRASTS.yaml']
    files = set(metadata)
    labels_by_dataset = {}
    for name in metadata[:3]:
        frame = pd.read_csv(source / name)
        for row in frame[frame.status == 'COMPLETED'].itertuples():
            relative = row.prediction_file
            path = (source / relative).resolve()
            if not path.is_relative_to(source) or not path.is_file():
                raise RuntimeError(f'Missing/out-of-root prediction artifact: {relative}')
            with np.load(path, allow_pickle=False) as data:
                labels, ids, probability = data['label'], data['sample_id'], data['calibrated_probability']
                if row.dataset not in labels_by_dataset:
                    labels_by_dataset[row.dataset] = (labels.copy(), ids.copy())
                y, expected_ids = labels_by_dataset[row.dataset]
                if not np.array_equal(y, labels) or not np.array_equal(ids, expected_ids):
                    raise RuntimeError(f'Label/sample-order mismatch: {relative}')
                if not np.isfinite(probability).all() or np.min(probability) < 0 or not np.allclose(probability.sum(1), 1, atol=1e-6):
                    raise RuntimeError(f'Invalid probabilities: {relative}')
            files.add(relative)
    print(json.dumps({'status': 'INPUT_ALIGNMENT_PASS', 'files': len(files),
                      'datasets': sorted(labels_by_dataset)}, indent=2), flush=True)
    if args.check_only: return
    if args.workdir is None: parser.error('--workdir is required unless --check-only is used')
    if args.workdir.resolve().is_relative_to(source): parser.error('Use a workspace outside the source artifacts')
    work = initialize(args.workdir)
    for relative in sorted(files):
        target = work / relative; target.parent.mkdir(parents=True, exist_ok=True)
        # Independent copies avoid symlink-based writes into the original artifacts.
        shutil.copyfile(source / relative, target)
    (work / 'REANALYSIS_INPUTS.json').write_text(json.dumps({p: digest(source / p) for p in sorted(files)}, indent=2) + '\n')
    os.environ['QIML_WORKDIR'] = str(work)
    from qiml_benchmark.statistics import run_all
    run_all.main()  # Original B=P=10,000 and original family definitions.
    from qiml_benchmark.statistics import secondary_intervals
    secondary_intervals.main()
    print('Recomputed results: ' + str(work / 'results/statistics'))


if __name__ == '__main__': main()
