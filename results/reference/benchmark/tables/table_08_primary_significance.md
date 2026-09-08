# Table 8: Primary Significance

| contrast | dataset | metric | method_A | method_B | setting | effect_A_minus_B | ci_95_lower | ci_95_upper | raw_p | permutations | status | holm_p | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1_fashion_mnist | fashion_mnist | accuracy | PdrQC-Matched | C-RBF | exact | -0.0219 | -0.0276 | -0.0159 | 9.999e-05 | 10000 | ESTIMATED | 0.00119988 | Statistically detectable paired difference. |
| H2_fashion_mnist | fashion_mnist | accuracy | QF-Ring | QF-Product | finite_shot_total_128 | -0.0241133 | -0.03154 | -0.016533 | 9.999e-05 | 10000 | ESTIMATED | 0.00119988 | Statistically detectable paired difference. |
| H3_fashion_mnist | fashion_mnist | accuracy | Yomo-Matched | VQC-DR | finite_shot_total_8 | -0.0670133 | -0.0730402 | -0.06092 | 9.999e-05 | 10000 | ESTIMATED | 0.00119988 | Statistically detectable paired difference. |
| H4_fashion_mnist | fashion_mnist | accuracy | PdrQC-Matched | QF-Ring | finite_shot_total_128 | 0.218127 | 0.210006 | 0.226173 | 9.999e-05 | 10000 | ESTIMATED | 0.00119988 | Statistically detectable paired difference. |
| H1_cifar10 | cifar10 | accuracy | PdrQC-Matched | C-RBF | exact | -0.035 | -0.0414 | -0.0287 | 9.999e-05 | 10000 | ESTIMATED | 0.00119988 | Statistically detectable paired difference. |
| H2_cifar10 | cifar10 | accuracy | QF-Ring | QF-Product | finite_shot_total_128 | -0.0893867 | -0.09768 | -0.0810532 | 9.999e-05 | 10000 | ESTIMATED | 0.00119988 | Statistically detectable paired difference. |
| H3_cifar10 | cifar10 | accuracy | Yomo-Matched | VQC-DR | finite_shot_total_8 | -0.0582067 | -0.0682473 | -0.0484198 | 9.999e-05 | 10000 | ESTIMATED | 0.00119988 | Statistically detectable paired difference. |
| H4_cifar10 | cifar10 | accuracy | PdrQC-Matched | QF-Ring | finite_shot_total_128 | 0.3387 | 0.329713 | 0.347747 | 9.999e-05 | 10000 | ESTIMATED | 0.00119988 | Statistically detectable paired difference. |
| H1_breastmnist | breastmnist | auroc | PdrQC-Matched | C-RBF | exact | 0.0106516 | -0.102146 | 0.122395 | 0.861914 | 10000 | ESTIMATED | 0.861914 | No statistically detectable difference was observed. |
| H2_breastmnist | breastmnist | auroc | QF-Ring | QF-Product | finite_shot_total_128 | -0.0489557 | -0.158466 | 0.0577008 | 0.348565 | 10000 | ESTIMATED | 0.69713 | No statistically detectable difference was observed. |
| H3_breastmnist | breastmnist | auroc | Yomo-Matched | VQC-DR | finite_shot_total_8 | -0.118992 | -0.199116 | -0.0388884 | 0.0191981 | 10000 | ESTIMATED | 0.0767923 | No statistically detectable difference was observed. |
| H4_breastmnist | breastmnist | auroc | PdrQC-Matched | QF-Ring | finite_shot_total_128 | 0.106307 | -0.00349137 | 0.211572 | 0.0432957 | 10000 | ESTIMATED | 0.129887 | No statistically detectable difference was observed. |
