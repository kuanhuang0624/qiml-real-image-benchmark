# Failure summary

No method failed its frozen primary protocol. Nine predeclared full-data boundaries remain: QK-IQP has three quadratic-scaling boundaries, and C-Poly/C-RBF have six classical-kernel scaling boundaries across the natural-image datasets.

Operational events included unavailable/corrupt initial download attempts, three Slurm submissions that failed before script startup, an interrupted CPU-oversubscribed smoke attempt, a corrected finite-shot probability-normalization pilot, and output-free statistics optimization attempts. Phase 6 finite-shot calibration was materialized after first test access despite being specified for pre-test materialization; it used validation data only and did not use test outcomes. Details are retained under `logs/failures/`.

At final audit, the original PyTorch environment was no longer discoverable. Nineteen dependency-independent tests passed; three PyTorch-dependent modules were explicitly skipped. Their successful Phase 3 execution and outputs remain preserved. This is a reproducibility limitation, not a scientific configuration change.
