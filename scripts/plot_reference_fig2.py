"""Render saved point estimates and CIs; never estimate intervals from aggregates."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('PdrQC-Matched', 'QF-Product', 'QF-Ring', 'VQC-DR', 'Yomo-Matched')
COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(ROOT / 'results/reference/benchmark/figure_data/fig2_finite_shot_performance.csv')
    with plt.rc_context({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
        for panel, (ax, dataset, title) in enumerate(zip(axes, ('fashion_mnist', 'cifar10', 'breastmnist'), ('Fashion-MNIST', 'CIFAR-10', 'BreastMNIST'))):
            for method, color, marker in zip(METHODS, COLORS, ('o', '^', 's', 'D', 'v')):
                rows = data[(data.dataset == dataset) & (data.method == method)].sort_values('nominal_budget')
                ax.plot(rows.nominal_budget, 100 * rows.point_estimate, label=method, color=color, marker=marker,
                        lw=2 if method == 'PdrQC-Matched' else 1.2, ms=5 if method == 'PdrQC-Matched' else 4)
                ax.fill_between(rows.nominal_budget, 100 * rows.ci_95_lower, 100 * rows.ci_95_upper, color=color, alpha=.12)
            ax.set_xscale('log', base=2); ax.set_xticks([1, 8, 32, 128, 512], labels=['1', '8', '32', '128', '512'])
            ax.set_title(f'({chr(97 + panel)}) {title}', fontweight='bold')
            ax.set_xlabel('State preparations per image')
            ax.set_ylabel('Test AUROC (%)' if dataset == 'breastmnist' else 'Test accuracy (%)')
            panel_data = data[data.dataset == dataset]
            bottom = max(0, 10 * np.floor(100 * panel_data.ci_95_lower.min() / 10)) if dataset == 'breastmnist' else 0
            top = min(100, max(90, 10 * np.ceil(100 * panel_data.ci_95_upper.max() / 10)))
            ax.set_ylim(bottom, top)
            ax.grid(axis='y', alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(.5, .01))
        fig.subplots_adjust(left=.06, right=.995, top=.89, bottom=.27, wspace=.30)
        for fmt in ('png', 'pdf', 'svg'):
            fig.savefig(args.output / ('fig2_finite_shot_performance.' + fmt), dpi=300)
        plt.close(fig)
    print('Rendered saved intervals without recomputation: ' + str(args.output))


if __name__ == '__main__': main()
