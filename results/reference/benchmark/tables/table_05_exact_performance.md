# Table 5: Exact Performance

| dataset | training_scale | method | reference_group | primary_metric | primary_metric_mean | primary_metric_sd | primary_metric_95%_CI | ci_95_lower | ci_95_upper | metric_macro_f1_mean | metric_auprc_mean | metric_nll_mean | metric_brier_mean | metric_ece_15_mean | metric_aurc_mean | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| breastmnist | D_100pct | C-CNN | full-image reference | AUROC | 0.598719 | 0.00761488 | [0.497283, 0.697788] | 0.497283 | 0.697788 | 0.470646 | 0.813239 | 0.573257 | 0.386359 | 0.0880994 | 0.195659 | COMPLETED |
| breastmnist | D_100pct | C-LR | matched PCA8 input | AUROC | 0.760443 |  | [0.662698, 0.849206] | 0.662698 | 0.849206 | 0.711111 | 0.87618 | 0.511605 | 0.314578 | 0.0937091 | 0.131006 | COMPLETED |
| breastmnist | D_100pct | C-MLP | matched PCA8 input | AUROC | 0.688109 | 0.0872172 | [0.574760, 0.797623] | 0.57476 | 0.797623 | 0.607364 | 0.845209 | 0.591571 | 0.405361 | 0.161211 | 0.326622 | COMPLETED |
| breastmnist | D_100pct | C-Poly | matched PCA8 input | AUROC | 0.720343 |  | [0.623846, 0.809524] | 0.623846 | 0.809524 | 0.646792 | 0.86819 | 0.538445 | 0.349812 | 0.0963512 | 0.147899 | COMPLETED |
| breastmnist | D_100pct | C-RBF | matched PCA8 input | AUROC | 0.774018 |  | [0.677318, 0.862155] | 0.677318 | 0.862155 | 0.7166 | 0.877413 | 0.483899 | 0.298464 | 0.134495 | 0.143782 | COMPLETED |
| breastmnist | D_100pct | C-RFF | matched PCA8 input | AUROC | 0.78467 | 0.0291209 | [0.693190, 0.866889] | 0.69319 | 0.866889 | 0.702267 | 0.89428 | 0.478221 | 0.307033 | 0.0998777 | 0.115148 | COMPLETED |
| breastmnist | D_100pct | C-ResNet | full-image reference | AUROC | 0.817878 |  | [0.729741, 0.894110] | 0.729741 | 0.89411 | 0.768889 | 0.897696 | 0.510002 | 0.288365 | 0.104026 | 0.117796 | COMPLETED |
| breastmnist | D_100pct | C-Trig | matched PCA8 input | AUROC | 0.782999 |  | [0.684837, 0.870933] | 0.684837 | 0.870933 | 0.724398 | 0.878925 | 0.464876 | 0.290084 | 0.0869795 | 0.125788 | COMPLETED |
| breastmnist | D_100pct | PdrQC-Matched | matched PCA8 input | AUROC | 0.78467 |  | [0.695489, 0.863617] | 0.695489 | 0.863617 | 0.679352 | 0.893969 | 0.481934 | 0.309128 | 0.0993148 | 0.116783 | COMPLETED |
| breastmnist | D_100pct | QF-Product | matched PCA8 input | AUROC | 0.702172 |  | [0.597118, 0.801587] | 0.597118 | 0.801587 | 0.678946 | 0.841957 | 0.515282 | 0.334447 | 0.0990059 | 0.163168 | COMPLETED |
| breastmnist | D_100pct | QF-Ring | matched PCA8 input | AUROC | 0.684419 |  | [0.581245, 0.785923] | 0.581245 | 0.785923 | 0.635854 | 0.825634 | 0.533571 | 0.348765 | 0.0881717 | 0.175591 | COMPLETED |
| breastmnist | D_100pct | QK-IQP | matched PCA8 input | AUROC | 0.546575 |  | [0.439850, 0.655394] | 0.43985 | 0.655394 | 0.522977 | 0.742427 | 0.591713 | 0.399688 | 0.139216 | 0.285994 | COMPLETED |
| breastmnist | D_100pct | VQC-DR | matched PCA8 input | AUROC | 0.716583 | 0.0132816 | [0.626009, 0.802426] | 0.626009 | 0.802426 | 0.664702 | 0.856858 | 0.525496 | 0.342339 | 0.102288 | 0.149946 | COMPLETED |
| breastmnist | D_100pct | Yomo-Matched | matched PCA8 input | AUROC | 0.466653 | 0.0130603 | [0.366748, 0.569900] | 0.366748 | 0.5699 | 0.489855 | 0.700401 | 0.693891 | 0.500743 | 0.135164 | 0.566388 | COMPLETED |
| cifar10 | D_5k | C-CNN | full-image reference | Accuracy | 0.3532 | 0.00831925 | [0.342599, 0.364301] | 0.342599 | 0.364301 | 0.336803 |  | 1.72775 | 0.763093 | 0.0282442 | 0.525997 | COMPLETED |
| cifar10 | D_5k | C-LR | matched PCA8 input | Accuracy | 0.7089 |  | [0.700300, 0.717700] | 0.7003 | 0.7177 | 0.707735 |  | 0.781836 | 0.396931 | 0.0132503 | 0.12529 | COMPLETED |
| cifar10 | D_5k | C-MLP | matched PCA8 input | Accuracy | 0.675433 | 0.0281098 | [0.647800, 0.700833] | 0.6478 | 0.700833 | 0.672413 |  | 0.87803 | 0.440433 | 0.0160619 | 0.158045 | COMPLETED |
| cifar10 | D_5k | C-Poly | matched PCA8 input | Accuracy | 0.7075 |  | [0.698700, 0.716400] | 0.6987 | 0.7164 | 0.709969 |  | 0.968578 | 0.467302 | 0.0890198 | 0.241084 | COMPLETED |
| cifar10 | D_5k | C-RBF | matched PCA8 input | Accuracy | 0.716 |  | [0.707497, 0.724700] | 0.707497 | 0.7247 | 0.715052 |  | 0.936034 | 0.454644 | 0.112189 | 0.209594 | COMPLETED |
| cifar10 | D_5k | C-RFF | matched PCA8 input | Accuracy | 0.708467 | 0.00085049 | [0.699967, 0.717000] | 0.699967 | 0.717 | 0.707454 |  | 0.794947 | 0.397029 | 0.0219673 | 0.12567 | COMPLETED |
| cifar10 | D_5k | C-ResNet | full-image reference | Accuracy | 0.8471 |  | [0.840200, 0.854100] | 0.8402 | 0.8541 | 0.846987 |  | 0.437135 | 0.217216 | 0.0121201 | 0.0358573 | COMPLETED |
| cifar10 | D_5k | C-Trig | matched PCA8 input | Accuracy | 0.6981 |  | [0.689400, 0.706900] | 0.6894 | 0.7069 | 0.697005 |  | 0.836915 | 0.413516 | 0.0136273 | 0.136433 | COMPLETED |
| cifar10 | D_5k | PdrQC-Matched | matched PCA8 input | Accuracy | 0.681 |  | [0.672100, 0.689900] | 0.6721 | 0.6899 | 0.679735 |  | 0.875029 | 0.433298 | 0.0161291 | 0.151624 | COMPLETED |
| cifar10 | D_5k | QF-Product | matched PCA8 input | Accuracy | 0.4303 |  | [0.420900, 0.439500] | 0.4209 | 0.4395 | 0.425126 |  | 1.6493 | 0.713841 | 0.0252152 | 0.408445 | COMPLETED |
| cifar10 | D_5k | QF-Ring | matched PCA8 input | Accuracy | 0.3637 |  | [0.354800, 0.372800] | 0.3548 | 0.3728 | 0.355119 |  | 1.84964 | 0.774424 | 0.0320863 | 0.489047 | COMPLETED |
| cifar10 | D_5k | QK-IQP | matched PCA8 input | Accuracy | 0.1148 |  | [0.108900, 0.121000] | 0.1089 | 0.121 | 0.104047 |  | 2.30154 | 0.899788 | 0.0110234 | 0.878371 | COMPLETED |
| cifar10 | D_5k | VQC-DR | matched PCA8 input | Accuracy | 0.419833 | 0.0174291 | [0.403200, 0.438600] | 0.4032 | 0.4386 | 0.412108 |  | 1.53161 | 0.696691 | 0.0386158 | 0.404671 | COMPLETED |
| cifar10 | D_5k | Yomo-Matched | matched PCA8 input | Accuracy | 0.280867 | 0.0148433 | [0.266899, 0.296900] | 0.266899 | 0.2969 | 0.269175 |  | 2.06334 | 0.837228 | 0.0420758 | 0.646857 | COMPLETED |
| fashion_mnist | D_5k | C-CNN | full-image reference | Accuracy | 0.737933 | 0.00647328 | [0.728867, 0.747033] | 0.728867 | 0.747033 | 0.725477 |  | 0.719277 | 0.354624 | 0.0182952 | 0.087911 | COMPLETED |
| fashion_mnist | D_5k | C-LR | matched PCA8 input | Accuracy | 0.7652 |  | [0.757700, 0.773000] | 0.7577 | 0.773 | 0.762957 |  | 0.615555 | 0.316372 | 0.0111391 | 0.0708353 | COMPLETED |
| fashion_mnist | D_5k | C-MLP | matched PCA8 input | Accuracy | 0.745467 | 0.00984395 | [0.733333, 0.756133] | 0.733333 | 0.756133 | 0.738086 |  | 0.691624 | 0.344802 | 0.0177501 | 0.0833855 | COMPLETED |
| fashion_mnist | D_5k | C-Poly | matched PCA8 input | Accuracy | 0.7872 |  | [0.779700, 0.794600] | 0.7797 | 0.7946 | 0.786051 |  | 0.762068 | 0.356746 | 0.103462 | 0.12046 | COMPLETED |
| fashion_mnist | D_5k | C-RBF | matched PCA8 input | Accuracy | 0.7937 |  | [0.786300, 0.801100] | 0.7863 | 0.8011 | 0.792176 |  | 0.732594 | 0.350185 | 0.112368 | 0.115843 | COMPLETED |
| fashion_mnist | D_5k | C-RFF | matched PCA8 input | Accuracy | 0.790233 | 0.00202073 | [0.782900, 0.797801] | 0.7829 | 0.797801 | 0.789053 |  | 0.579232 | 0.294781 | 0.0170505 | 0.0621429 | COMPLETED |
| fashion_mnist | D_5k | C-ResNet | full-image reference | Accuracy | 0.8647 |  | [0.858200, 0.871000] | 0.8582 | 0.871 | 0.864545 |  | 0.386405 | 0.195979 | 0.00851747 | 0.0291123 | COMPLETED |
| fashion_mnist | D_5k | C-Trig | matched PCA8 input | Accuracy | 0.787 |  | [0.779498, 0.794500] | 0.779498 | 0.7945 | 0.785644 |  | 0.586437 | 0.298183 | 0.0110002 | 0.0631263 | COMPLETED |
| fashion_mnist | D_5k | PdrQC-Matched | matched PCA8 input | Accuracy | 0.7718 |  | [0.764100, 0.779500] | 0.7641 | 0.7795 | 0.769634 |  | 0.618533 | 0.31415 | 0.0130664 | 0.0699198 | COMPLETED |
| fashion_mnist | D_5k | QF-Product | matched PCA8 input | Accuracy | 0.5825 |  | [0.573600, 0.591300] | 0.5736 | 0.5913 | 0.573217 |  | 1.22926 | 0.544164 | 0.024059 | 0.204515 | COMPLETED |
| fashion_mnist | D_5k | QF-Ring | matched PCA8 input | Accuracy | 0.5891 |  | [0.580100, 0.597900] | 0.5801 | 0.5979 | 0.583134 |  | 1.23607 | 0.542913 | 0.0255719 | 0.204214 | COMPLETED |
| fashion_mnist | D_5k | QK-IQP | matched PCA8 input | Accuracy | 0.1347 |  | [0.128100, 0.141400] | 0.1281 | 0.1414 | 0.133288 |  | 2.29271 | 0.897935 | 0.0303992 | 0.822958 | COMPLETED |
| fashion_mnist | D_5k | VQC-DR | matched PCA8 input | Accuracy | 0.519567 | 0.0209538 | [0.499732, 0.540800] | 0.499732 | 0.5408 | 0.514514 |  | 1.38018 | 0.612939 | 0.0385585 | 0.274146 | COMPLETED |
| fashion_mnist | D_5k | Yomo-Matched | matched PCA8 input | Accuracy | 0.308367 | 0.0266335 | [0.281699, 0.331001] | 0.281699 | 0.331001 | 0.290829 |  | 2.00233 | 0.82272 | 0.0515065 | 0.583161 | COMPLETED |
| mnist | D_5k | C-CNN | full-image reference | Accuracy | 0.9072 | 0.00775951 | [0.898200, 0.915567] | 0.8982 | 0.915567 | 0.906849 |  | 0.311428 | 0.138612 | 0.0111741 | 0.0143652 | COMPLETED |
| mnist | D_5k | C-LR | matched PCA8 input | Accuracy | 0.8031 |  | [0.795800, 0.810500] | 0.7958 | 0.8105 | 0.799214 |  | 0.537826 | 0.272539 | 0.0101818 | 0.0551426 | COMPLETED |
| mnist | D_5k | C-MLP | matched PCA8 input | Accuracy | 0.7752 | 0.0184073 | [0.757199, 0.793002] | 0.757199 | 0.793002 | 0.770129 |  | 0.617578 | 0.306093 | 0.0147854 | 0.0686077 | COMPLETED |
| mnist | D_5k | C-Poly | matched PCA8 input | Accuracy | 0.8247 |  | [0.817600, 0.831900] | 0.8176 | 0.8319 | 0.82328 |  | 0.670875 | 0.302127 | 0.0685319 | 0.106543 | COMPLETED |
| mnist | D_5k | C-RBF | matched PCA8 input | Accuracy | 0.8443 |  | [0.837500, 0.851100] | 0.8375 | 0.8511 | 0.841063 |  | 0.606131 | 0.273993 | 0.121129 | 0.0694827 | COMPLETED |
| mnist | D_5k | C-RFF | matched PCA8 input | Accuracy | 0.8369 | 0.00150997 | [0.830033, 0.843633] | 0.830033 | 0.843633 | 0.833638 |  | 0.46841 | 0.232332 | 0.0104856 | 0.0400758 | COMPLETED |
| mnist | D_5k | C-ResNet | full-image reference | Accuracy | 0.9637 |  | [0.960000, 0.967300] | 0.96 | 0.9673 | 0.963211 |  | 0.117531 | 0.0550588 | 0.00303338 | 0.00233456 | COMPLETED |
| mnist | D_5k | C-Trig | matched PCA8 input | Accuracy | 0.8297 |  | [0.822700, 0.836700] | 0.8227 | 0.8367 | 0.825985 |  | 0.494362 | 0.241695 | 0.00732746 | 0.0429393 | COMPLETED |
| mnist | D_5k | PdrQC-Matched | matched PCA8 input | Accuracy | 0.7934 |  | [0.786000, 0.800900] | 0.786 | 0.8009 | 0.788874 |  | 0.575484 | 0.282016 | 0.0120001 | 0.0574017 | COMPLETED |
| mnist | D_5k | QF-Product | matched PCA8 input | Accuracy | 0.4969 |  | [0.487700, 0.506100] | 0.4877 | 0.5061 | 0.487341 |  | 1.44373 | 0.6273 | 0.0276401 | 0.281599 | COMPLETED |
| mnist | D_5k | QF-Ring | matched PCA8 input | Accuracy | 0.4221 |  | [0.413100, 0.431300] | 0.4131 | 0.4313 | 0.410442 |  | 1.68577 | 0.711961 | 0.0303865 | 0.379967 | COMPLETED |
| mnist | D_5k | QK-IQP | matched PCA8 input | Accuracy | 0.1085 |  | [0.102500, 0.114503] | 0.1025 | 0.114503 | 0.107949 |  | 2.30163 | 0.899807 | 0.010914 | 0.883792 | COMPLETED |
| mnist | D_5k | VQC-DR | matched PCA8 input | Accuracy | 0.413267 | 0.00846719 | [0.402732, 0.424434] | 0.402732 | 0.424434 | 0.401795 |  | 1.62172 | 0.696706 | 0.0372676 | 0.363593 | COMPLETED |
| mnist | D_5k | Yomo-Matched | matched PCA8 input | Accuracy | 0.2382 | 0.0104843 | [0.227000, 0.249833] | 0.227 | 0.249833 | 0.22294 |  | 2.13457 | 0.859189 | 0.0373465 | 0.661379 | COMPLETED |
