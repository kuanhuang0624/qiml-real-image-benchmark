# Porting notes

This is a code-oriented release derived from the completed QIML projects, not a
replacement history. `provenance/ORIGINS.json` records the source commits;
`provenance/SOURCE_FILES.csv` records source-file SHA256 values and destinations.

## Mechanical portability changes

- Python namespace `src` became `qiml_benchmark` for editable/package installation.
- Runtime roots now resolve from `QIML_WORKDIR`, with a marker for new workspaces,
  rather than from the installed source directory.
- The original author's ResNet weight path became `QIML_RESNET_WEIGHTS`.
- Historical report compilers with hard-coded commit claims, one-off recovery
  scripts, environment/GPU probes and the cancelled IonQ audit were excluded.
- Unrelated historical wrapper names are retained only where they delegate to
  their scientific driver; the experiment guide names the actual drivers.

The NumPy gate operations, QF readout, PdrQC selection/grouping, IQP kernel,
PyTorch quantum model/loss, CNN architecture, classical estimators, original
validation grids, sampling conventions and statistical formulas were retained.
Reference result CSVs and selected figure files are copied byte-for-byte.

## Corrections for new runs

- PCA cache files now save the fitted PCA mean, and test extraction subtracts
  it before projection. Older cache files without that field retain their
  original zero-center transform.
- H1 uses the paper's fixed C-RBF comparator. The original builder selected
  the highest-scoring classical method on validation data; the archived runs
  selected C-RBF on all three inferential datasets. New runs keep C-RBF even
  when another classical control has a higher validation score.

These changes affect newly generated experiments. Archived configurations,
predictions, result tables, figures, and reference checksums are unchanged.

## New release-only utilities

`cli.py`, `examples.py`, `features/interface.py`, `paths.py` and `protocol.py`
provide installation examples, isolated output directories and a prospective
lock for **new** reproductions. These utilities were not used to generate the
historical paper results. In particular:

- The example uses a fixed C=1 readout and a small synthetic split; it is not the
  original benchmark search. Its new PCA adapter explicitly preserves PCA
  centering; the original cache builder remains separately available.
- The original test loader required a specific old Git commit message. That
  cannot exist in a clean release history. New reproductions instead require
  a checked hash lock covering the new sources and artifacts.
- Explicit guards were added to the finite-shot and scaling test entry points.
  They do not change the historical prediction or statistical values.
- Test-extraction logs identify the new lock, not the original research commit.
- Documentation and example plotting were added without modifying the archived
  manuscript figures or their CSV source data.

The release does not retroactively repair original provenance limitations or
claim that its stronger new-run lock existed during the historical experiment.
