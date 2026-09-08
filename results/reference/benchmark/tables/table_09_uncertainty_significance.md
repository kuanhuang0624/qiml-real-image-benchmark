# Table 9: Uncertainty Significance

| contrast | dataset | metric | method_A | method_B | effect_A_minus_B | ci_95_lower | ci_95_upper | raw_p | permutations | status | holm_p | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U_brier_fashion_mnist | fashion_mnist | brier | PdrQC-Matched | C-RBF | -0.0360352 | -0.0423116 | -0.0298173 | 9.999e-05 | 10000 | ESTIMATED | 0.00059994 | Statistically detectable paired difference. |
| U_aurc_fashion_mnist | fashion_mnist | aurc | PdrQC-Matched | C-RBF | -0.0459229 | -0.0525328 | -0.0392053 | 9.999e-05 | 10000 | ESTIMATED | 0.00059994 | Statistically detectable paired difference. |
| U_brier_cifar10 | cifar10 | brier | PdrQC-Matched | C-RBF | -0.0213457 | -0.0272533 | -0.0154243 | 9.999e-05 | 10000 | ESTIMATED | 0.00059994 | Statistically detectable paired difference. |
| U_aurc_cifar10 | cifar10 | aurc | PdrQC-Matched | C-RBF | -0.0579699 | -0.0678495 | -0.048238 | 9.999e-05 | 10000 | ESTIMATED | 0.00059994 | Statistically detectable paired difference. |
| U_brier_breastmnist | breastmnist | brier | PdrQC-Matched | C-RBF | 0.0106639 | -0.0564208 | 0.078533 | 0.776022 | 10000 | ESTIMATED | 1 | No statistically detectable difference was observed. |
| U_aurc_breastmnist | breastmnist | aurc | PdrQC-Matched | C-RBF | -0.0269986 | -0.106929 | 0.0515375 | 0.519248 | 10000 | ESTIMATED | 1 | No statistically detectable difference was observed. |
