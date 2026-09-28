# Quantum Machine Learning for Image Classification: A Controlled Benchmark

Official code for **Quantum Machine Learning for Image Classification: A Controlled Benchmark** by **Kuan Huang, Meng Xu, and Yingfeng Wang**.

Accepted at the **2026 AAAI Fall Symposium Series**, [Second AAAI Symposium on Quantum Information & Machine Learning (QIML): Bridging Quantum Computing and Artificial Intelligence](https://aaai.org/conference/fall-symposia/2026-fall-symposium-series-2/#QIML).

**The paper will be available online soon.**

## Overview

The benchmark compares six quantum-derived methods, six matched classical controls, and two neural references on **MNIST, Fashion-MNIST, CIFAR-10, and BreastMNIST**.

```text
Image → frozen ResNet-18 → 512D features → train-only PCA8 → angles in [−π, π]
      → quantum or classical model → validation calibration → test evaluation
```

| Group | Methods |
| --- | --- |
| Quantum-derived methods | QK-IQP, QF-Product, QF-Ring, VQC-DR, Yomo-Matched, PdrQC-Matched |
| Matched classical controls | C-LR, C-MLP, C-Poly, C-RBF, C-RFF, C-Trig |
| Neural references | C-ResNet (full 512D features), C-CNN (original images) |

Experiments cover exact and finite-shot evaluation, calibration, selective risk, training-set size, and PCA8/12/16 interface widths. Quantum computations use CPU simulation and offline measurement sampling.

## Installation

Use Python 3.11 or 3.12.

```bash
git clone https://github.com/kuanhuang0624/qiml-real-image-benchmark.git
cd qiml-real-image-benchmark
python -m venv .venv
source .venv/bin/activate
python -m pip install -c environment/core-constraints.txt -e '.[test]'
```

For image feature extraction and VQC, Yomo, and CNN training, also install PyTorch and torchvision:

```bash
python -m pip install -c environment/core-constraints.txt -e '.[images,test]'
```

## Quick start

Run the tests, verify the saved results, and run a small synthetic example:

```bash
python -m pytest -q
python scripts/check_reference_results.py
qiml demo --output runs/demo
```

The demo runs PdrQC-Matched, QF-Product, and QF-Ring on synthetic features without dataset downloads or a GPU. It checks installation and produces example predictions. Use a new output directory for each run.

## Run the benchmark

Follow the [experiment guide](experiments/README.md) to download datasets, extract frozen ResNet-18 features, fit the shared interface, select models on validation data, and run exact, finite-shot, and training-scale experiments.

Raw datasets, pretrained weights, feature caches, trained checkpoints, and individual predictions must be obtained or generated separately. The PCA-width extension includes its analytic model, selected observables, and results; a complete width-experiment runner is not included. See [reproducibility details](docs/REPRODUCIBILITY.md).

### Use your own image features

Provide two NPZ files with separate training, validation, and test partitions:

| File | Arrays |
| --- | --- |
| `train_val.npz` | `train_features [N,512]`, `train_labels [N]`, `val_features [V,512]`, `val_labels [V]` |
| `test.npz` | `test_features [T,512]`, `test_labels [T]` |

Features must be finite floating-point values. Labels must be contiguous integers from `0` to `K-1`, with `K=2` or `K=10`.

```bash
qiml feature-run --train-val /path/to/train_val.npz \
  --test /path/to/test.npz --output runs/my_experiment
```

This command fits train-only PCA, selects PdrQC observables, calibrates on validation data, and evaluates PdrQC-Matched, QF-Product, and QF-Ring with a fixed `C=1` logistic readout. The full benchmark's hyperparameter search and finite-shot experiments use the drivers in the experiment guide.

## Paper results

| Paper item | Saved results |
| --- | --- |
| Table 1: exact performance | [Performance table](results/reference/benchmark/tables/table_05_exact_performance.md), [bootstrap intervals](results/reference/benchmark/statistics/bootstrap_intervals.csv) |
| Table 2A: primary comparisons | [Primary tests](results/reference/benchmark/statistics/primary_tests.csv) |
| Table 2B: uncertainty comparisons | [Uncertainty tests](results/reference/benchmark/statistics/uncertainty_tests.csv) |
| Figure 2: finite-shot performance | [Source data](results/reference/benchmark/figure_data/fig2_finite_shot_performance.csv) |
| Figure 3: interface width | [Source data](results/reference/width/figure_data/fig01_interface_width_sensitivity.csv), [exploratory tests](results/reference/width/statistics/exploratory_tests.csv) |

![Finite-shot performance](docs/figures/fig2_finite_shot_performance.png)

Generate Figure 2 from the saved estimates and confidence intervals:

```bash
python scripts/plot_reference_fig2.py --output runs/figure2
```

See [methods](docs/METHODS.md), [results](docs/RESULTS.md), and [statistical reanalysis](docs/STATISTICS.md) for details. Recomputing paired tests and confidence intervals requires individual prediction records.

## Citation

```bibtex
@inproceedings{huang2026controlled,
  title     = {Quantum Machine Learning for Image Classification: A Controlled Benchmark},
  author    = {Huang, Kuan and Xu, Meng and Wang, Yingfeng},
  booktitle = {Second AAAI Symposium on Quantum Information \& Machine Learning (QIML): Bridging Quantum Computing and Artificial Intelligence},
  year      = {2026},
  note      = {Accepted; publication forthcoming}
}
```

## License

The code is released under the [MIT License](LICENSE). External datasets, pretrained weights, and dependencies retain their respective licenses.
