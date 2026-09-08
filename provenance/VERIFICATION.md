# Code-release verification

This record concerns release packaging, not a new run of the paper experiments.
Commands were run from the new repository root. Original research directories
were only read, and no real quantum service was called.

## Clean core installation: Python 3.12.3

```bash
python -m venv .venv
.venv/bin/python -m pip install -c environment/core-constraints.txt -e '.[test]'
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
```

Observed outputs:

```text
Successfully installed qiml-real-image-benchmark-0.1.0 and core dependencies.
No broken requirements found.
23 passed, 1 skipped in 3.55s
SKIPPED: tests/test_training_resume.py — torch not installed in core-only environment.
```

## Optional PyTorch check: existing Python 3.11.10 environment

The existing environment supplied torch 2.11.0+cu128 and torchvision 0.26.0+cu128.
The tests used CPU tensors, not CUDA or QPU execution.

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
```

Observed output after all shot-accounting tests were added:

```text
24 passed in 25.50s
```

This was not a fresh installation of the large optional torch dependencies.
The clean core environment above was independently created and installed.

## Saved results, code provenance and release hygiene

```bash
python scripts/release_audit.py --original-root /path/to/original-projects
python scripts/check_reference_results.py
```

Observed checks:

```text
status: PASS
reference_files_byte_identical: 111
original_source_hashes_verified: 187
findings: []
primary_tests: 12; Holm verified; 8 below 0.05
uncertainty_tests: 6; Holm verified; 4 below 0.05
exploratory_width_tests: 2; separate Holm adjustment verified
reference_hashes_verified: 111
scientific_core_AST_equivalent_modules: 10
```

The release scanner checks file types, English text, personal server paths,
several credential patterns, Python syntax and original reference hashes. A
heuristic credential scan is not a guarantee that every possible secret format
has been detected. No API key or login token was read into repository files.
The AST check compared the original and exported circuit, measurement, PdrQC,
classical, calibration, metric, VQC and IQP core modules after namespace-only
substitution. Numerical operations in those ten modules are unchanged.

## README examples and plotting

```bash
qiml demo --output runs/clean_environment_demo
qiml init --workdir runs/readme_full
qiml verify-lock --workdir runs/readme_full
python scripts/plot_reference_fig2.py --output runs/readme_figure
```

The demo completed with `DEMO_ONLY` status, all three methods and normalized
prediction probabilities. Its synthetic accuracy values are not scientific
evidence. The new workspace initialized successfully. Verification of that
empty workspace correctly failed nonzero because no validation lock existed.
The plot command generated PNG/PDF/SVG from saved CIs. The PNG was visually
inspected for readable panel labels, confidence bands and bottom legend.

## External prediction-input check (no resampling)

```bash
python scripts/reanalyze_published.py --artifacts /path/to/original-benchmark --check-only
```

Observed output:

```text
status: INPUT_ALIGNMENT_PASS
files: 1857
datasets: breastmnist, cifar10, fashion_mnist, mnist
```

No source predictions were modified, no new held-out model inference was run,
and no 10,000-draw statistical recomputation was performed during packaging.
The full expensive reproduction entry points remain documented, not certified
as a newly completed scientific rerun. GitHub CI is a separate run of the
core-only checks and does not perform dataset downloads or training.
