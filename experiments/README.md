# Running experiments

## Execution levels

1. **Installation check:** `qiml demo`. Synthetic, seconds to minutes, no paper evidence.
2. **Feature experiment:** `qiml feature-run`. A small new three-method experiment,
   not the full original selection protocol.
3. **Full benchmark reproduction:** the drivers below. Real datasets, pretrained
   weights, and substantial CPU/RAM/time are required. This release did not rerun
   the complete scientific benchmark. Read the limitations before launching it.
4. **Historical statistical reanalysis:** requires external saved predictions;
   does not require model retraining. See [STATISTICS.md](../docs/STATISTICS.md).

The reference configurations in `reference_configs/` and selected observables in
`selections/` document completed runs. They are **not** pre-test locks for a new run.
Do not copy old temperatures or old validation results into a fresh reproduction.

## A. Create an isolated work directory

From the repository root, with the package and `[images]` extra installed:

```bash
qiml init --workdir runs/full
export QIML_WORKDIR="$PWD/runs/full"
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MPLBACKEND=Agg
```

Every historical driver was ported to `QIML_WORKDIR`; installed code and saved
reference results are not output destinations. The workspace marker prevents
accidentally pointing the drivers at an existing research archive.

## B. Obtain data and the frozen encoder

**Network activity is explicit in this step only.** The download module fetches
official archives and verifies known checksums. Check each dataset's current
terms before downloading. Archives can contain official test payloads; the
preparation stage does not open the test arrays.

```bash
python -m qiml_benchmark.data.prepare_all
python -m qiml_benchmark.data.build_nested_splits
```

Obtain the torchvision ImageNet1K-V1 ResNet-18 state dict from the official
[PyTorch weight URL](https://download.pytorch.org/models/resnet18-f37072fd.pth).
Store it outside Git, and point the extractor to it:

```bash
export QIML_RESNET_WEIGHTS=/absolute/path/to/resnet18-f37072fd.pth
python -m qiml_benchmark.features.extract_resnet
python -m qiml_benchmark.features.fit_shared_interfaces
```

The extractor never downloads weights implicitly. It uses deterministic
bilinear 224x224 resizing, ImageNet normalization, and a frozen encoder in eval
mode. PCA/scaling fit each training subset only. CPU extraction can take hours;
CUDA is optional. Do not substitute different encoder weights and call the run
an exact reproduction.

## C. Validation: primary and scaling

The primary training subset is `D_5k` for natural images and `D_100pct` for
BreastMNIST. The model seeds are `42,2026,3407` for stochastic methods;
deterministic fixed methods use seed `42`.

```bash
for dataset in mnist fashion_mnist cifar10 breastmnist; do
  python -m qiml_benchmark.validation.run_primary_fixed "$dataset"
  python -m qiml_benchmark.validation.run_primary_trainable "$dataset" --device cpu
done
```

Run validation for the additional training scales:

```bash
for dataset in mnist fashion_mnist cifar10; do
  for scale in D_1k D_10k D_full; do
    python -m qiml_benchmark.validation.run_primary_fixed "$dataset" --scale "$scale" --mode scaling
    python -m qiml_benchmark.validation.run_primary_trainable "$dataset" --scale "$scale" --mode scaling --device cpu
  done
done
for scale in D_25pct D_50pct; do
  python -m qiml_benchmark.validation.run_primary_fixed breastmnist --scale "$scale" --mode scaling
  python -m qiml_benchmark.validation.run_primary_trainable breastmnist --scale "$scale" --mode scaling --device cpu
done
```

Keep the declared full-natural-data kernel boundaries. The original driver
contains dataset-specific scaling-grid constants selected during the historical
primary stage; these are retained, not a new automatically adapting search.
Freeze after all validation stages complete. H1 compares the strongest
validation-selected quantum method with the fixed C-RBF control.

```bash
python -m qiml_benchmark.validation.freeze
python -m qiml_benchmark.validation.freeze_breast_thresholds
qiml lock --workdir "$QIML_WORKDIR"
qiml verify-lock --workdir "$QIML_WORKDIR"
```

The new lock rejects incomplete selected method/seed sets and missing checkpoints
or BreastMNIST thresholds. It hashes source code, configurations, feature caches,
observables and checkpoints. It is a new reproduction record; the original
research lock and its chronology remain unchanged. The drivers save failures in
their CSVs, so review those CSVs, not just shell exit codes.

## D. Held-out evaluation and finite-shot experiments

Only after the new lock passes:

```bash
python -m qiml_benchmark.features.extract_test
for dataset in mnist fashion_mnist cifar10 breastmnist; do
  python -m qiml_benchmark.evaluation.run_exact "$dataset"
done
python -m qiml_benchmark.evaluation.aggregate_exact

for dataset in mnist fashion_mnist cifar10 breastmnist; do
  python -m qiml_benchmark.evaluation.run_finite_local "$dataset"
  python -m qiml_benchmark.evaluation.run_finite_qk "$dataset"
done
python -m qiml_benchmark.evaluation.aggregate_finite
```

`run_finite_qk` can be expensive: it samples and corrects dense Gram matrices.
It is not a per-image-budget experiment and must not be pooled with local
measurement budgets. Finite-shot routines preserve the original algorithm,
including noisy training-feature readout refits and validation-only calibration.
They are not hardware execution or fixed-checkpoint inference-only noise tests.

Evaluate all declared training scales, then aggregate and analyze:

```bash
for dataset in mnist fashion_mnist cifar10; do
  for scale in D_1k D_5k D_10k D_full; do
    python -m qiml_benchmark.evaluation.run_scaling "$dataset" "$scale"
  done
done
for scale in D_25pct D_50pct D_100pct; do
  python -m qiml_benchmark.evaluation.run_scaling breastmnist "$scale"
done
python -m qiml_benchmark.evaluation.analyze_scaling
python -m qiml_benchmark.uncertainty.run_all
python -m qiml_benchmark.statistics.run_all
python -m qiml_benchmark.statistics.secondary_intervals
```

Each full statistical run retains 10,000 bootstrap and 10,000 permutation draws.
Do not reduce these counts and present the output as the historical inference.

## PCA-width extension

`qiml_benchmark.width.model` contains the executed analytic product-RY
expectation and selection implementation for PCA8/12/16. The saved extension
configurations, selected observables, source-data tables and exploratory tests
are included. PCA8 originally reused parent caches to avoid randomized-SVD
drift. The complete parent-cache-dependent width orchestration is not packaged
as an automatic fresh run. Historical replay requires the original parent caches.
Changing the width or rerunning selection is a new exploratory experiment.

## Provenance and expectations

Fresh training can produce different selections or results across dependency,
device, and numerical-library versions. The runtime tables in a new workspace
are separate from `results/reference/`. No command should overwrite or relabel
the completed benchmark's numerical evidence. Only bounded offline release
tests were run during packaging; the full expensive commands are documented
entry points, not an assertion that every full stage has been re-executed here.
