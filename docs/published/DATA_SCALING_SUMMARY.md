# Data-scaling summary

All 15 frozen dataset-scale cells were evaluated, producing 351 completed seed-level runs. There are 9 predeclared boundary rows where full kernel evaluation was infeasible; no boundary was replaced with a surrogate result.

Rank stability is reported only over methods common to each scale pair in `results/aggregates/ranking_stability.csv`. All test evaluations reused validation-frozen configurations and calibration parameters.

Across scale pairs, Spearman rank correlations ranged from 0.836 to 1.000 for MNIST, 0.800 to 0.987 for Fashion-MNIST, 0.918 to 0.991 for CIFAR-10, and 0.843 to 0.882 for BreastMNIST. These correlations use only methods completed at both scales. The full natural-image QK-IQP, C-Poly, and C-RBF points are not imputed or connected through their recorded boundaries.
