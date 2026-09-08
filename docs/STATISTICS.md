# Statistical workflow

The primary benchmark used 10,000 class-stratified image bootstrap replicates
and 10,000 paired randomization permutations. Model and measurement seeds are
retained in the resampling hierarchy. These are **image-level benchmark** tests,
not the patient-clustered EMBED/CSAW analyses from the unrelated ISBI project.

The twelve primary contrasts share one Holm family. The six uncertainty
contrasts use a separate Holm family. Two post-hoc width comparisons form a
separate exploratory family. Never combine them retrospectively to change
significance, or describe an interval containing zero as equivalence.

## Check the included results

```bash
python scripts/check_reference_results.py
```

This verifies saved hashes and recomputes Holm adjustments from saved raw
p-values. It does **not** regenerate p-values or intervals from aggregate tables.

## Recompute the original analyses

The private archive contains the prediction-level arrays omitted from this
code repository. If you have authorized access to that artifact directory:

```bash
python scripts/reanalyze_published.py --artifacts /path/to/real_image_qml_benchmark_2026 --check-only
python scripts/reanalyze_published.py --artifacts /path/to/real_image_qml_benchmark_2026 --workdir runs/reanalysis
```

Required inputs are the three run-index CSVs in `results/raw/`, the frozen
`PRIMARY_CONTRASTS.yaml`, two scaling aggregate tables, and every referenced
prediction NPZ. Each prediction provides `sample_id`, `label`,
`calibrated_probability`, and `threshold`. The checker verifies sample order,
labels and probability normalization before any resampling.

The script copies those inputs into a new workspace, hashes them, and invokes
the ported original statistics code with unchanged B=P=10,000. Source artifacts
are not changed. The full operation can use substantial memory, disk and CPU
time; run it deliberately. Prediction archives and resampling distributions
remain ignored by Git.

Without the external predictions, only aggregate consistency checks are
possible. The repository does not claim that aggregate means and standard
deviations are sufficient to reproduce paired inference.

## Preserved implementation details

The original metric definitions, threshold conventions, finite-shot hierarchy,
permutation RNG seeds and Holm families are retained. For example, the original
secondary `binary_ap` resampling routine uses stable score sorting and is not
silently replaced by a different tied-score estimator in this export. Any
scientific correction requires a separate versioned analysis, not a packaging
edit to the saved numbers.
