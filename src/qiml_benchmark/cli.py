"""Small, explicit commands; no network use unless a dataset module is invoked."""
import argparse
import json
import os
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description='QIML real-image benchmark utilities')
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='Offline synthetic installation check, not paper evidence')
    demo.add_argument('--output', type=Path, required=True)
    feature = sub.add_parser('feature-run', help='New fixed-C experiment on user-provided 512D features')
    feature.add_argument('--train-val', type=Path, required=True)
    feature.add_argument('--test', type=Path, required=True)
    feature.add_argument('--output', type=Path, required=True)
    init = sub.add_parser('init', help='Create an empty isolated full-reproduction workspace')
    init.add_argument('--workdir', type=Path, required=True)
    init.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[2])
    for name in ('lock', 'verify-lock'):
        command = sub.add_parser(name)
        command.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'demo':
        from qiml_benchmark.examples import demo as run
        result = run(args.output)
    elif args.command == 'feature-run':
        from qiml_benchmark.examples import feature_experiment
        result = feature_experiment(args.train_val, args.test, args.output)
    elif args.command == 'init':
        from qiml_benchmark.paths import initialize
        registry = args.repository / 'experiments/reference_configs/benchmark/method_registry.yaml'
        if not registry.is_file(): parser.error('Pass --repository pointing to a clone of this repository')
        root = initialize(args.workdir)
        shutil.copyfile(registry, root / 'configs/method_registry.yaml')
        result = {'workspace': str(root), 'next': 'export QIML_WORKDIR=' + str(root), 'test_access': 'locked'}
    else:
        os.environ['QIML_WORKDIR'] = str(args.workdir.resolve())
        from qiml_benchmark.paths import workspace_root
        from qiml_benchmark.protocol import lock, verify, digest
        root = workspace_root()
        if args.command == 'lock': lock(root)
        verify(root)
        result = {'status': 'LOCK_VERIFIED', 'sha256': digest(root / 'REPRODUCTION_LOCK.json')}
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
