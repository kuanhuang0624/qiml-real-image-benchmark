# Reproducibility and limitations

## Available without external artifacts

- Scientific code for quantum/classical methods, measurement, calibration,
  validation, exact/finite-shot evaluation, scaling, uncertainty and inference.
- Frozen numerical configurations, selected Pauli strings/groups, result tables,
  significance tests and source-data CSVs for figures.
- Offline unit tests, gradient/resume checks when PyTorch is installed, a
  synthetic example, and consistency checks for the saved results.

## External inputs required for full reproduction

Raw public datasets, official ImageNet weights, extracted features, trained
checkpoints and prediction arrays are intentionally not distributed here.
Obtain datasets under their own terms or use the author's separately held
private archive. Training and resampling can be expensive. No full benchmark
training or full 10,000-draw statistical reanalysis was rerun during packaging.

The width extension's parent-cache-dependent orchestration is not an automatic
fresh workflow in this release. Its analytic core, frozen choices, results and
statistical evidence are included.

## Historical limitations carried forward

The implementation audit found that the historical trainable checkpoints were
not included in the old Git/frozen hash set, the successful CPU training
invocation was not preserved, and some final environment-audit tests had been
skipped because PyTorch was unavailable in that particular environment.
Finite-shot calibrators were materialized after exact test access, although
they were fitted only from validation predictions. This release does not
convert those facts into stronger prospective provenance claims.

Randomized PCA can drift between numerical-library versions; the width study
explicitly reused the frozen PCA8 parent arrays. Fresh runs should record
versions and hashes and should not promise bitwise historical reproduction.

## Scope and safety

The full-run drivers load locally trusted historical checkpoint/pickle formats.
Never load untrusted checkpoints or altered CIFAR pickle archives. The official
download stage verifies published archive checksums. Example NPZ files are
opened with `allow_pickle=False`.

The project is not a clinical risk-prediction system. BreastMNIST is a benchmark
dataset, not evidence of validated clinical utility. No quantum hardware results
or quantum advantage are claimed. Release verification output is recorded in
`provenance/VERIFICATION.md`.
