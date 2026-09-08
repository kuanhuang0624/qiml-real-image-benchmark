PARTIAL_SCALING_BOUNDARIES

# Final experiment status

1. Completed datasets: MNIST, Fashion-MNIST, CIFAR-10, BreastMNIST.
2. Completed primary methods: all 14 registered methods.
3. Completed training sizes: natural-image D_1k, D_5k, D_10k, and method-dependent D_full; BreastMNIST D_25pct, D_50pct, D_100pct.
4. Failed methods: none at the frozen primary scale.
5. Scaling boundaries: nine—three QK-IQP quadratic and six C-Poly/C-RBF classical-kernel full-data boundaries.
6. Validation runs: 978 (312 primary plus 666 scaling).
7. Exact test runs: 96.
8. Finite-shot runs: 1,500.
9. Data-scaling rows: 360; 351 completed and nine boundary rows.
10. Completed model seeds: 42, 2026, 3407 where frozen; deterministic methods use seed 42.
11. Completed measurement seeds: 101, 211, 307, 401, 503.
12. Primary metrics: accuracy for natural images and AUROC for BreastMNIST.
13. Uncertainty metrics: NLL, Brier, ECE_15, predictive entropy, model uncertainty, quantum-measurement uncertainty, error-detection AUROC, AURC, and risk at 100/90/80% coverage.
14. Confidence intervals: 701 saved interval rows from 10,000 hierarchical stratified bootstrap replicates, plus 21 rank-stability intervals.
15. Significance tests: 12 primary and six secondary uncertainty tests, each with 10,000 paired permutations.
16. Holm-adjusted results: eight primary and four uncertainty tests below 0.05.
17. Ranking stability: 21 scale-pair comparisons; observed Spearman rho range 0.800 to 1.000 over common completed methods.
18. Total statevector evaluations: 1,696,083 under recorded exact/scaling counters.
19. Total circuit evaluations: 1,010,970,722,160 offline simulated executions under the shot-draw convention.
20. Total kernel entries: 5,364,929,453.
21. Total offline shot draws: 1,010,970,722,160.
22. Total CPU-hours: 1.323011 instrumented plus additional uninstrumented work reported as unknown.
23. Total GPU-hours: at most 0.018611 allocated; zero model-training GPU-hours and zero real-QPU use.
24. First test access: 2026-08-12T20:06:32.448199-04:00, after lock commit `04d4776` and passing hash guard.
25. Protocol deviations: validation-only finite-shot calibration materialized after test access; corrected finite-shot probability renormalization; final PyTorch environment unavailable for clean rerun. No test outcome was used for retuning.
26. Table paths: `results/tables/table_01_dataset_splits.*` through `table_15_failures_integrity.*`.
27. Figure paths: `figures/{png,pdf,svg}/fig01_benchmark_pipeline.*` through `fig12_ranking_stability.*`.
28. Remaining experiments: no remaining in-boundary frozen runs; the nine declared full-data kernel boundaries remain unexecuted.
