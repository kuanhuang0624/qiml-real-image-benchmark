# Table 12: Data Scaling

| dataset | training_scale | method | runs | primary_metric | primary_metric_mean | primary_metric_sd | statevector_evaluations | kernel_entries | status | ci_95_lower | ci_95_upper | rank | primary_metric_95%_CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| breastmnist | D_100pct | C-CNN | 3 | auroc | 0.598719 | 0.00761488 | 0 | 0 | COMPLETED | 0.498327 | 0.697022 | 12 | [0.498327, 0.697022] |
| breastmnist | D_100pct | C-LR | 1 | auroc | 0.760443 |  | 0 | 0 | COMPLETED | 0.662698 | 0.849206 | 6 | [0.662698, 0.849206] |
| breastmnist | D_100pct | C-MLP | 3 | auroc | 0.688109 | 0.0872172 | 0 | 0 | COMPLETED | 0.575184 | 0.797619 | 10 | [0.575184, 0.797619] |
| breastmnist | D_100pct | C-Poly | 1 | auroc | 0.720343 |  | 0 | 0 | COMPLETED | 0.623846 | 0.809524 | 7 | [0.623846, 0.809524] |
| breastmnist | D_100pct | C-RBF | 1 | auroc | 0.774018 |  | 0 | 0 | COMPLETED | 0.677318 | 0.862155 | 5 | [0.677318, 0.862155] |
| breastmnist | D_100pct | C-RFF | 3 | auroc | 0.78467 | 0.0291209 | 0 | 0 | COMPLETED | 0.69096 | 0.864874 | 2.5 | [0.690960, 0.864874] |
| breastmnist | D_100pct | C-ResNet | 1 | auroc | 0.817878 |  | 0 | 0 | COMPLETED | 0.729741 | 0.89411 | 1 | [0.729741, 0.894110] |
| breastmnist | D_100pct | C-Trig | 1 | auroc | 0.782999 |  | 0 | 0 | COMPLETED | 0.684837 | 0.870933 | 4 | [0.684837, 0.870933] |
| breastmnist | D_100pct | PdrQC-Matched | 1 | auroc | 0.78467 |  | 702 | 0 | COMPLETED | 0.695489 | 0.863617 | 2.5 | [0.695489, 0.863617] |
| breastmnist | D_100pct | QF-Product | 1 | auroc | 0.702172 |  | 702 | 0 | COMPLETED | 0.597118 | 0.801587 | 9 | [0.597118, 0.801587] |
| breastmnist | D_100pct | QF-Ring | 1 | auroc | 0.684419 |  | 702 | 0 | COMPLETED | 0.581245 | 0.785923 | 11 | [0.581245, 0.785923] |
| breastmnist | D_100pct | QK-IQP | 1 | auroc | 0.546575 |  | 0 | 383292 | COMPLETED | 0.43985 | 0.655394 | 13 | [0.439850, 0.655394] |
| breastmnist | D_100pct | VQC-DR | 3 | auroc | 0.716583 | 0.0132816 | 468 | 0 | COMPLETED | 0.626493 | 0.800893 | 8 | [0.626493, 0.800893] |
| breastmnist | D_100pct | Yomo-Matched | 3 | auroc | 0.466653 | 0.0130603 | 468 | 0 | COMPLETED | 0.36647 | 0.569411 | 14 | [0.366470, 0.569411] |
| breastmnist | D_25pct | C-CNN | 3 | auroc | 0.589599 | 0.00315365 | 0 | 0 | COMPLETED | 0.487881 | 0.688877 | 11 | [0.487881, 0.688877] |
| breastmnist | D_25pct | C-LR | 1 | auroc | 0.763158 |  | 0 | 0 | COMPLETED | 0.668338 | 0.850047 | 3 | [0.668338, 0.850047] |
| breastmnist | D_25pct | C-MLP | 3 | auroc | 0.577694 | 0.153386 | 0 | 0 | COMPLETED | 0.418405 | 0.743317 | 12 | [0.418405, 0.743317] |
| breastmnist | D_25pct | C-Poly | 1 | auroc | 0.772348 |  | 0 | 0 | COMPLETED | 0.688179 | 0.850042 | 2 | [0.688179, 0.850042] |
| breastmnist | D_25pct | C-RBF | 1 | auroc | 0.744779 |  | 0 | 0 | COMPLETED | 0.645572 | 0.836884 | 6 | [0.645572, 0.836884] |
| breastmnist | D_25pct | C-RFF | 3 | auroc | 0.772626 | 0.00580428 | 0 | 0 | COMPLETED | 0.678986 | 0.858885 | 1 | [0.678986, 0.858885] |
| breastmnist | D_25pct | C-ResNet | 1 | auroc | 0.751044 |  | 0 | 0 | COMPLETED | 0.64787 | 0.844199 | 4 | [0.647870, 0.844199] |
| breastmnist | D_25pct | C-Trig | 1 | auroc | 0.732665 |  | 0 | 0 | COMPLETED | 0.630326 | 0.824984 | 7 | [0.630326, 0.824984] |
| breastmnist | D_25pct | PdrQC-Matched | 1 | auroc | 0.750835 |  | 293 | 0 | COMPLETED | 0.651838 | 0.841688 | 5 | [0.651838, 0.841688] |
| breastmnist | D_25pct | QF-Product | 1 | auroc | 0.635338 |  | 293 | 0 | COMPLETED | 0.532159 | 0.736842 | 10 | [0.532159, 0.736842] |
| breastmnist | D_25pct | QF-Ring | 1 | auroc | 0.673977 |  | 293 | 0 | COMPLETED | 0.571011 | 0.775068 | 9 | [0.571011, 0.775068] |
| breastmnist | D_25pct | QK-IQP | 1 | auroc | 0.483292 |  | 0 | 40141 | COMPLETED | 0.380942 | 0.589599 | 13 | [0.380942, 0.589599] |
| breastmnist | D_25pct | VQC-DR | 3 | auroc | 0.725634 | 0.0290372 | 468 | 0 | COMPLETED | 0.630117 | 0.814397 | 8 | [0.630117, 0.814397] |
| breastmnist | D_25pct | Yomo-Matched | 3 | auroc | 0.475912 | 0.00777361 | 468 | 0 | COMPLETED | 0.380533 | 0.569063 | 14 | [0.380533, 0.569063] |
| breastmnist | D_50pct | C-CNN | 3 | auroc | 0.598719 | 0.00579299 | 0 | 0 | COMPLETED | 0.497144 | 0.697508 | 12 | [0.497144, 0.697508] |
| breastmnist | D_50pct | C-LR | 1 | auroc | 0.767753 |  | 0 | 0 | COMPLETED | 0.670635 | 0.855686 | 3.5 | [0.670635, 0.855686] |
| breastmnist | D_50pct | C-MLP | 3 | auroc | 0.652813 | 0.0892928 | 0 | 0 | COMPLETED | 0.54163 | 0.766917 | 11 | [0.541630, 0.766917] |
| breastmnist | D_50pct | C-Poly | 1 | auroc | 0.756892 |  | 0 | 0 | COMPLETED | 0.65873 | 0.846282 | 5 | [0.658730, 0.846282] |
| breastmnist | D_50pct | C-RBF | 1 | auroc | 0.796575 |  | 0 | 0 | COMPLETED | 0.707811 | 0.876149 | 2 | [0.707811, 0.876149] |
| breastmnist | D_50pct | C-RFF | 3 | auroc | 0.767753 | 0.0192136 | 0 | 0 | COMPLETED | 0.671329 | 0.853801 | 3.5 | [0.671329, 0.853801] |
| breastmnist | D_50pct | C-ResNet | 1 | auroc | 0.803467 |  | 0 | 0 | COMPLETED | 0.713659 | 0.883041 | 1 | [0.713659, 0.883041] |
| breastmnist | D_50pct | C-Trig | 1 | auroc | 0.756266 |  | 0 | 0 | COMPLETED | 0.663737 | 0.841902 | 6 | [0.663737, 0.841902] |
| breastmnist | D_50pct | PdrQC-Matched | 1 | auroc | 0.738513 |  | 430 | 0 | COMPLETED | 0.642648 | 0.827908 | 7 | [0.642648, 0.827908] |
| breastmnist | D_50pct | QF-Product | 1 | auroc | 0.69528 |  | 430 | 0 | COMPLETED | 0.590434 | 0.796157 | 9 | [0.590434, 0.796157] |
| breastmnist | D_50pct | QF-Ring | 1 | auroc | 0.713241 |  | 430 | 0 | COMPLETED | 0.608391 | 0.814745 | 8 | [0.608391, 0.814745] |
| breastmnist | D_50pct | QK-IQP | 1 | auroc | 0.553049 |  | 0 | 117820 | COMPLETED | 0.452799 | 0.650794 | 13 | [0.452799, 0.650794] |
| breastmnist | D_50pct | VQC-DR | 3 | auroc | 0.659287 | 0.0338045 | 468 | 0 | COMPLETED | 0.569269 | 0.749238 | 10 | [0.569269, 0.749238] |
| breastmnist | D_50pct | Yomo-Matched | 3 | auroc | 0.519006 | 0.0224255 | 468 | 0 | COMPLETED | 0.427804 | 0.612503 | 14 | [0.427804, 0.612503] |
| cifar10 | D_10k | C-CNN | 3 | accuracy | 0.398333 | 0.0213645 | 0 | 0 | COMPLETED | 0.3754 | 0.416933 | 11 | [0.375400, 0.416933] |
| cifar10 | D_10k | C-LR | 1 | accuracy | 0.715 |  | 0 | 0 | COMPLETED | 0.7065 | 0.7237 | 5 | [0.706500, 0.723700] |
| cifar10 | D_10k | C-MLP | 3 | accuracy | 0.705 | 0.00104403 | 0 | 0 | COMPLETED | 0.696733 | 0.7135 | 7 | [0.696733, 0.713500] |
| cifar10 | D_10k | C-Poly | 1 | accuracy | 0.7152 |  | 0 | 0 | COMPLETED | 0.7066 | 0.724002 | 4 | [0.706600, 0.724002] |
| cifar10 | D_10k | C-RBF | 1 | accuracy | 0.7237 |  | 0 | 0 | COMPLETED | 0.715 | 0.7323 | 2 | [0.715000, 0.732300] |
| cifar10 | D_10k | C-RFF | 3 | accuracy | 0.7173 | 0.00217945 | 0 | 0 | COMPLETED | 0.7087 | 0.726 | 3 | [0.708700, 0.726000] |
| cifar10 | D_10k | C-ResNet | 1 | accuracy | 0.8596 |  | 0 | 0 | COMPLETED | 0.8529 | 0.8663 | 1 | [0.852900, 0.866300] |
| cifar10 | D_10k | C-Trig | 1 | accuracy | 0.7076 |  | 0 | 0 | COMPLETED | 0.6989 | 0.7164 | 6 | [0.698900, 0.716400] |
| cifar10 | D_10k | PdrQC-Matched | 1 | accuracy | 0.685 |  | 20000 | 0 | COMPLETED | 0.6762 | 0.6939 | 8 | [0.676200, 0.693900] |
| cifar10 | D_10k | QF-Product | 1 | accuracy | 0.4449 |  | 20000 | 0 | COMPLETED | 0.4355 | 0.4545 | 9 | [0.435500, 0.454500] |
| cifar10 | D_10k | QF-Ring | 1 | accuracy | 0.3744 |  | 20000 | 0 | COMPLETED | 0.3654 | 0.3838 | 12 | [0.365400, 0.383800] |
| cifar10 | D_10k | QK-IQP | 1 | accuracy | 0.1185 |  | 0 | 2e+08 | COMPLETED | 0.112 | 0.1247 | 14 | [0.112000, 0.124700] |
| cifar10 | D_10k | VQC-DR | 3 | accuracy | 0.4313 | 0.0255869 | 30000 | 0 | COMPLETED | 0.4049 | 0.453933 | 10 | [0.404900, 0.453933] |
| cifar10 | D_10k | Yomo-Matched | 3 | accuracy | 0.283167 | 0.017178 | 30000 | 0 | COMPLETED | 0.267766 | 0.301901 | 13 | [0.267766, 0.301901] |
| cifar10 | D_1k | C-CNN | 3 | accuracy | 0.259467 | 0.00994703 | 0 | 0 | COMPLETED | 0.2489 | 0.270801 | 12 | [0.248900, 0.270801] |
| cifar10 | D_1k | C-LR | 1 | accuracy | 0.6961 |  | 0 | 0 | COMPLETED | 0.6874 | 0.7048 | 2 | [0.687400, 0.704800] |
| cifar10 | D_1k | C-MLP | 3 | accuracy | 0.4737 | 0.0988168 | 0 | 0 | COMPLETED | 0.3956 | 0.582303 | 8 | [0.395600, 0.582303] |
| cifar10 | D_1k | C-Poly | 1 | accuracy | 0.6612 |  | 0 | 0 | COMPLETED | 0.6521 | 0.6701 | 6 | [0.652100, 0.670100] |
| cifar10 | D_1k | C-RBF | 1 | accuracy | 0.693 |  | 0 | 0 | COMPLETED | 0.6842 | 0.7018 | 3 | [0.684200, 0.701800] |
| cifar10 | D_1k | C-RFF | 3 | accuracy | 0.684867 | 0.000493288 | 0 | 0 | COMPLETED | 0.6761 | 0.693433 | 4 | [0.676100, 0.693433] |
| cifar10 | D_1k | C-ResNet | 1 | accuracy | 0.8163 |  | 0 | 0 | COMPLETED | 0.8089 | 0.8237 | 1 | [0.808900, 0.823700] |
| cifar10 | D_1k | C-Trig | 1 | accuracy | 0.6685 |  | 0 | 0 | COMPLETED | 0.6594 | 0.677302 | 5 | [0.659400, 0.677302] |
| cifar10 | D_1k | PdrQC-Matched | 1 | accuracy | 0.6587 |  | 11000 | 0 | COMPLETED | 0.6499 | 0.6679 | 7 | [0.649900, 0.667900] |
| cifar10 | D_1k | QF-Product | 1 | accuracy | 0.3926 |  | 11000 | 0 | COMPLETED | 0.3833 | 0.402 | 9 | [0.383300, 0.402000] |
| cifar10 | D_1k | QF-Ring | 1 | accuracy | 0.3331 |  | 11000 | 0 | COMPLETED | 0.3244 | 0.3422 | 11 | [0.324400, 0.342200] |
| cifar10 | D_1k | QK-IQP | 1 | accuracy | 0.1081 |  | 0 | 1.1e+07 | COMPLETED | 0.102198 | 0.1143 | 14 | [0.102198, 0.114300] |
| cifar10 | D_1k | VQC-DR | 3 | accuracy | 0.365267 | 0.0219778 | 30000 | 0 | COMPLETED | 0.343233 | 0.3862 | 10 | [0.343233, 0.386200] |
| cifar10 | D_1k | Yomo-Matched | 3 | accuracy | 0.221367 | 0.00742989 | 30000 | 0 | COMPLETED | 0.2116 | 0.230733 | 13 | [0.211600, 0.230733] |
| cifar10 | D_5k | C-CNN | 3 | accuracy | 0.3532 | 0.00831925 | 0 | 0 | COMPLETED | 0.342667 | 0.3643 | 12 | [0.342667, 0.364300] |
| cifar10 | D_5k | C-LR | 1 | accuracy | 0.7089 |  | 0 | 0 | COMPLETED | 0.7003 | 0.7177 | 3 | [0.700300, 0.717700] |
| cifar10 | D_5k | C-MLP | 3 | accuracy | 0.675433 | 0.0281098 | 0 | 0 | COMPLETED | 0.647097 | 0.7012 | 8 | [0.647097, 0.701200] |
| cifar10 | D_5k | C-Poly | 1 | accuracy | 0.7075 |  | 0 | 0 | COMPLETED | 0.6987 | 0.7164 | 5 | [0.698700, 0.716400] |
| cifar10 | D_5k | C-RBF | 1 | accuracy | 0.716 |  | 0 | 0 | COMPLETED | 0.707497 | 0.7247 | 2 | [0.707497, 0.724700] |
| cifar10 | D_5k | C-RFF | 3 | accuracy | 0.708467 | 0.00085049 | 0 | 0 | COMPLETED | 0.6999 | 0.717 | 4 | [0.699900, 0.717000] |
| cifar10 | D_5k | C-ResNet | 1 | accuracy | 0.8471 |  | 0 | 0 | COMPLETED | 0.8402 | 0.8541 | 1 | [0.840200, 0.854100] |
| cifar10 | D_5k | C-Trig | 1 | accuracy | 0.6981 |  | 0 | 0 | COMPLETED | 0.6894 | 0.7069 | 6 | [0.689400, 0.706900] |
| cifar10 | D_5k | PdrQC-Matched | 1 | accuracy | 0.681 |  | 15000 | 0 | COMPLETED | 0.6721 | 0.6899 | 7 | [0.672100, 0.689900] |
| cifar10 | D_5k | QF-Product | 1 | accuracy | 0.4303 |  | 15000 | 0 | COMPLETED | 0.4209 | 0.4395 | 9 | [0.420900, 0.439500] |
| cifar10 | D_5k | QF-Ring | 1 | accuracy | 0.3637 |  | 15000 | 0 | COMPLETED | 0.3548 | 0.3728 | 11 | [0.354800, 0.372800] |
| cifar10 | D_5k | QK-IQP | 1 | accuracy | 0.1148 |  | 0 | 7.5e+07 | COMPLETED | 0.1089 | 0.121 | 14 | [0.108900, 0.121000] |
| cifar10 | D_5k | VQC-DR | 3 | accuracy | 0.419833 | 0.0174291 | 30000 | 0 | COMPLETED | 0.4034 | 0.438834 | 10 | [0.403400, 0.438834] |
| cifar10 | D_5k | Yomo-Matched | 3 | accuracy | 0.280867 | 0.0148433 | 30000 | 0 | COMPLETED | 0.266867 | 0.2968 | 13 | [0.266867, 0.296800] |
| cifar10 | D_full | C-CNN | 3 | accuracy | 0.572467 | 0.00540031 | 0 | 0 | COMPLETED | 0.562699 | 0.582467 | 7 | [0.562699, 0.582467] |
| cifar10 | D_full | C-LR | 1 | accuracy | 0.7157 |  | 0 | 0 | COMPLETED | 0.7072 | 0.7245 | 3 | [0.707200, 0.724500] |
| cifar10 | D_full | C-MLP | 3 | accuracy | 0.709367 | 0.00120554 | 0 | 0 | COMPLETED | 0.701033 | 0.717967 | 5 | [0.701033, 0.717967] |
| cifar10 | D_full | C-RFF | 3 | accuracy | 0.724633 | 0.00183394 | 0 | 0 | COMPLETED | 0.716067 | 0.733367 | 2 | [0.716067, 0.733367] |
| cifar10 | D_full | C-ResNet | 1 | accuracy | 0.8748 |  | 0 | 0 | COMPLETED | 0.8684 | 0.8812 | 1 | [0.868400, 0.881200] |
| cifar10 | D_full | C-Trig | 1 | accuracy | 0.7127 |  | 0 | 0 | COMPLETED | 0.704 | 0.7213 | 4 | [0.704000, 0.721300] |
| cifar10 | D_full | PdrQC-Matched | 1 | accuracy | 0.6923 |  | 55000 | 0 | COMPLETED | 0.6838 | 0.7012 | 6 | [0.683800, 0.701200] |
| cifar10 | D_full | QF-Product | 1 | accuracy | 0.4422 |  | 55000 | 0 | COMPLETED | 0.4329 | 0.4515 | 9 | [0.432900, 0.451500] |
| cifar10 | D_full | QF-Ring | 1 | accuracy | 0.378 |  | 55000 | 0 | COMPLETED | 0.369 | 0.3875 | 10 | [0.369000, 0.387500] |
| cifar10 | D_full | VQC-DR | 3 | accuracy | 0.445833 | 0.00394631 | 30000 | 0 | COMPLETED | 0.436833 | 0.454767 | 8 | [0.436833, 0.454767] |
| cifar10 | D_full | Yomo-Matched | 3 | accuracy | 0.289733 | 0.00205994 | 30000 | 0 | COMPLETED | 0.281567 | 0.298033 | 11 | [0.281567, 0.298033] |
| fashion_mnist | D_10k | C-CNN | 3 | accuracy | 0.765967 | 0.013512 | 0 | 0 | COMPLETED | 0.751098 | 0.7792 | 8 | [0.751098, 0.779200] |
| fashion_mnist | D_10k | C-LR | 1 | accuracy | 0.7707 |  | 0 | 0 | COMPLETED | 0.7633 | 0.778202 | 7 | [0.763300, 0.778202] |
| fashion_mnist | D_10k | C-MLP | 3 | accuracy | 0.763767 | 0.00402782 | 0 | 0 | COMPLETED | 0.755933 | 0.7717 | 9 | [0.755933, 0.771700] |
| fashion_mnist | D_10k | C-Poly | 1 | accuracy | 0.7929 |  | 0 | 0 | COMPLETED | 0.7855 | 0.8004 | 5 | [0.785500, 0.800400] |
| fashion_mnist | D_10k | C-RBF | 1 | accuracy | 0.7964 |  | 0 | 0 | COMPLETED | 0.7891 | 0.8036 | 3 | [0.789100, 0.803600] |
| fashion_mnist | D_10k | C-RFF | 3 | accuracy | 0.796867 | 0.000635085 | 0 | 0 | COMPLETED | 0.7897 | 0.803967 | 2 | [0.789700, 0.803967] |
| fashion_mnist | D_10k | C-ResNet | 1 | accuracy | 0.8756 |  | 0 | 0 | COMPLETED | 0.869497 | 0.8818 | 1 | [0.869497, 0.881800] |
| fashion_mnist | D_10k | C-Trig | 1 | accuracy | 0.7932 |  | 0 | 0 | COMPLETED | 0.785797 | 0.8006 | 4 | [0.785797, 0.800600] |
| fashion_mnist | D_10k | PdrQC-Matched | 1 | accuracy | 0.7807 |  | 20000 | 0 | COMPLETED | 0.7731 | 0.7882 | 6 | [0.773100, 0.788200] |
| fashion_mnist | D_10k | QF-Product | 1 | accuracy | 0.5876 |  | 20000 | 0 | COMPLETED | 0.5785 | 0.5967 | 11 | [0.578500, 0.596700] |
| fashion_mnist | D_10k | QF-Ring | 1 | accuracy | 0.5984 |  | 20000 | 0 | COMPLETED | 0.5894 | 0.6074 | 10 | [0.589400, 0.607400] |
| fashion_mnist | D_10k | QK-IQP | 1 | accuracy | 0.138 |  | 0 | 2e+08 | COMPLETED | 0.1314 | 0.1448 | 14 | [0.131400, 0.144800] |
| fashion_mnist | D_10k | VQC-DR | 3 | accuracy | 0.518833 | 0.00858856 | 30000 | 0 | COMPLETED | 0.507399 | 0.530101 | 12 | [0.507399, 0.530101] |
| fashion_mnist | D_10k | Yomo-Matched | 3 | accuracy | 0.313367 | 0.0245321 | 30000 | 0 | COMPLETED | 0.2881 | 0.3345 | 13 | [0.288100, 0.334500] |
| fashion_mnist | D_1k | C-CNN | 3 | accuracy | 0.590333 | 0.0301888 | 0 | 0 | COMPLETED | 0.561 | 0.618 | 8 | [0.561000, 0.618000] |
| fashion_mnist | D_1k | C-LR | 1 | accuracy | 0.7577 |  | 0 | 0 | COMPLETED | 0.7503 | 0.7653 | 6 | [0.750300, 0.765300] |
| fashion_mnist | D_1k | C-MLP | 3 | accuracy | 0.292333 | 0.232532 | 0 | 0 | COMPLETED | 0.0829 | 0.540502 | 12 | [0.082900, 0.540502] |
| fashion_mnist | D_1k | C-Poly | 1 | accuracy | 0.7493 |  | 0 | 0 | COMPLETED | 0.7415 | 0.7571 | 7 | [0.741500, 0.757100] |
| fashion_mnist | D_1k | C-RBF | 1 | accuracy | 0.7737 |  | 0 | 0 | COMPLETED | 0.7661 | 0.7813 | 2 | [0.766100, 0.781300] |
| fashion_mnist | D_1k | C-RFF | 3 | accuracy | 0.771967 | 0.000665833 | 0 | 0 | COMPLETED | 0.764533 | 0.779333 | 4 | [0.764533, 0.779333] |
| fashion_mnist | D_1k | C-ResNet | 1 | accuracy | 0.8308 |  | 0 | 0 | COMPLETED | 0.8238 | 0.8379 | 1 | [0.823800, 0.837900] |
| fashion_mnist | D_1k | C-Trig | 1 | accuracy | 0.7734 |  | 0 | 0 | COMPLETED | 0.7659 | 0.781 | 3 | [0.765900, 0.781000] |
| fashion_mnist | D_1k | PdrQC-Matched | 1 | accuracy | 0.7581 |  | 11000 | 0 | COMPLETED | 0.7503 | 0.7658 | 5 | [0.750300, 0.765800] |
| fashion_mnist | D_1k | QF-Product | 1 | accuracy | 0.5546 |  | 11000 | 0 | COMPLETED | 0.5452 | 0.5638 | 10 | [0.545200, 0.563800] |
| fashion_mnist | D_1k | QF-Ring | 1 | accuracy | 0.5844 |  | 11000 | 0 | COMPLETED | 0.5752 | 0.593602 | 9 | [0.575200, 0.593602] |
| fashion_mnist | D_1k | QK-IQP | 1 | accuracy | 0.1162 |  | 0 | 1.1e+07 | COMPLETED | 0.1099 | 0.1224 | 14 | [0.109900, 0.122400] |
| fashion_mnist | D_1k | VQC-DR | 3 | accuracy | 0.496767 | 0.0248116 | 30000 | 0 | COMPLETED | 0.4715 | 0.519201 | 11 | [0.471500, 0.519201] |
| fashion_mnist | D_1k | Yomo-Matched | 3 | accuracy | 0.280067 | 0.0124263 | 30000 | 0 | COMPLETED | 0.266266 | 0.2931 | 13 | [0.266266, 0.293100] |
| fashion_mnist | D_5k | C-CNN | 3 | accuracy | 0.737933 | 0.00647328 | 0 | 0 | COMPLETED | 0.729067 | 0.747068 | 9 | [0.729067, 0.747068] |
| fashion_mnist | D_5k | C-LR | 1 | accuracy | 0.7652 |  | 0 | 0 | COMPLETED | 0.7577 | 0.773 | 7 | [0.757700, 0.773000] |
| fashion_mnist | D_5k | C-MLP | 3 | accuracy | 0.745467 | 0.00984395 | 0 | 0 | COMPLETED | 0.7337 | 0.7562 | 8 | [0.733700, 0.756200] |
| fashion_mnist | D_5k | C-Poly | 1 | accuracy | 0.7872 |  | 0 | 0 | COMPLETED | 0.7797 | 0.7946 | 4 | [0.779700, 0.794600] |
| fashion_mnist | D_5k | C-RBF | 1 | accuracy | 0.7937 |  | 0 | 0 | COMPLETED | 0.7863 | 0.8011 | 2 | [0.786300, 0.801100] |
| fashion_mnist | D_5k | C-RFF | 3 | accuracy | 0.790233 | 0.00202073 | 0 | 0 | COMPLETED | 0.782867 | 0.7978 | 3 | [0.782867, 0.797800] |
| fashion_mnist | D_5k | C-ResNet | 1 | accuracy | 0.8647 |  | 0 | 0 | COMPLETED | 0.8582 | 0.871 | 1 | [0.858200, 0.871000] |
| fashion_mnist | D_5k | C-Trig | 1 | accuracy | 0.787 |  | 0 | 0 | COMPLETED | 0.779498 | 0.7945 | 5 | [0.779498, 0.794500] |
| fashion_mnist | D_5k | PdrQC-Matched | 1 | accuracy | 0.7718 |  | 15000 | 0 | COMPLETED | 0.7641 | 0.7795 | 6 | [0.764100, 0.779500] |
| fashion_mnist | D_5k | QF-Product | 1 | accuracy | 0.5825 |  | 15000 | 0 | COMPLETED | 0.5736 | 0.5913 | 11 | [0.573600, 0.591300] |
| fashion_mnist | D_5k | QF-Ring | 1 | accuracy | 0.5891 |  | 15000 | 0 | COMPLETED | 0.5801 | 0.5979 | 10 | [0.580100, 0.597900] |
| fashion_mnist | D_5k | QK-IQP | 1 | accuracy | 0.1347 |  | 0 | 7.5e+07 | COMPLETED | 0.1281 | 0.1414 | 14 | [0.128100, 0.141400] |
| fashion_mnist | D_5k | VQC-DR | 3 | accuracy | 0.519567 | 0.0209538 | 30000 | 0 | COMPLETED | 0.499698 | 0.5412 | 12 | [0.499698, 0.541200] |
| fashion_mnist | D_5k | Yomo-Matched | 3 | accuracy | 0.308367 | 0.0266335 | 30000 | 0 | COMPLETED | 0.282198 | 0.330967 | 13 | [0.282198, 0.330967] |
| fashion_mnist | D_full | C-CNN | 3 | accuracy | 0.866567 | 0.00240069 | 0 | 0 | COMPLETED | 0.8601 | 0.872901 | 2 | [0.860100, 0.872901] |
| fashion_mnist | D_full | C-LR | 1 | accuracy | 0.7716 |  | 0 | 0 | COMPLETED | 0.7641 | 0.7792 | 7 | [0.764100, 0.779200] |
| fashion_mnist | D_full | C-MLP | 3 | accuracy | 0.773067 | 0.0022053 | 0 | 0 | COMPLETED | 0.765633 | 0.780533 | 6 | [0.765633, 0.780533] |
| fashion_mnist | D_full | C-RFF | 3 | accuracy | 0.804067 | 0.00011547 | 0 | 0 | COMPLETED | 0.796967 | 0.811033 | 3 | [0.796967, 0.811033] |
| fashion_mnist | D_full | C-ResNet | 1 | accuracy | 0.8951 |  | 0 | 0 | COMPLETED | 0.8894 | 0.9009 | 1 | [0.889400, 0.900900] |
| fashion_mnist | D_full | C-Trig | 1 | accuracy | 0.7992 |  | 0 | 0 | COMPLETED | 0.792 | 0.8066 | 4 | [0.792000, 0.806600] |
| fashion_mnist | D_full | PdrQC-Matched | 1 | accuracy | 0.786 |  | 65000 | 0 | COMPLETED | 0.7786 | 0.7934 | 5 | [0.778600, 0.793400] |
| fashion_mnist | D_full | QF-Product | 1 | accuracy | 0.5891 |  | 65000 | 0 | COMPLETED | 0.5801 | 0.598 | 9 | [0.580100, 0.598000] |
| fashion_mnist | D_full | QF-Ring | 1 | accuracy | 0.5984 |  | 65000 | 0 | COMPLETED | 0.5894 | 0.6076 | 8 | [0.589400, 0.607600] |
| fashion_mnist | D_full | VQC-DR | 3 | accuracy | 0.555667 | 0.00673003 | 30000 | 0 | COMPLETED | 0.545666 | 0.565 | 10 | [0.545666, 0.565000] |
| fashion_mnist | D_full | Yomo-Matched | 3 | accuracy | 0.313333 | 0.0132085 | 30000 | 0 | COMPLETED | 0.2987 | 0.3269 | 11 | [0.298700, 0.326900] |
| mnist | D_10k | C-CNN | 3 | accuracy | 0.941133 | 0.00303699 | 0 | 0 | COMPLETED | 0.9361 | 0.946233 | 2 | [0.936100, 0.946233] |
| mnist | D_10k | C-LR | 1 | accuracy | 0.8053 |  | 0 | 0 | COMPLETED | 0.7979 | 0.8127 | 7 | [0.797900, 0.812700] |
| mnist | D_10k | C-MLP | 3 | accuracy | 0.801367 | 0.00458294 | 0 | 0 | COMPLETED | 0.793132 | 0.809367 | 9 | [0.793132, 0.809367] |
| mnist | D_10k | C-Poly | 1 | accuracy | 0.8328 |  | 0 | 0 | COMPLETED | 0.8258 | 0.84 | 6 | [0.825800, 0.840000] |
| mnist | D_10k | C-RBF | 1 | accuracy | 0.8547 |  | 0 | 0 | COMPLETED | 0.848 | 0.8612 | 3 | [0.848000, 0.861200] |
| mnist | D_10k | C-RFF | 3 | accuracy | 0.848667 | 0.00172434 | 0 | 0 | COMPLETED | 0.8421 | 0.8553 | 4 | [0.842100, 0.855300] |
| mnist | D_10k | C-ResNet | 1 | accuracy | 0.9699 |  | 0 | 0 | COMPLETED | 0.9665 | 0.9732 | 1 | [0.966500, 0.973200] |
| mnist | D_10k | C-Trig | 1 | accuracy | 0.8363 |  | 0 | 0 | COMPLETED | 0.8292 | 0.8431 | 5 | [0.829200, 0.843100] |
| mnist | D_10k | PdrQC-Matched | 1 | accuracy | 0.8039 |  | 20000 | 0 | COMPLETED | 0.7965 | 0.8112 | 8 | [0.796500, 0.811200] |
| mnist | D_10k | QF-Product | 1 | accuracy | 0.5084 |  | 20000 | 0 | COMPLETED | 0.4992 | 0.5177 | 10 | [0.499200, 0.517700] |
| mnist | D_10k | QF-Ring | 1 | accuracy | 0.4231 |  | 20000 | 0 | COMPLETED | 0.4144 | 0.4321 | 11 | [0.414400, 0.432100] |
| mnist | D_10k | QK-IQP | 1 | accuracy | 0.1098 |  | 0 | 2e+08 | COMPLETED | 0.1039 | 0.1157 | 14 | [0.103900, 0.115700] |
| mnist | D_10k | VQC-DR | 3 | accuracy | 0.420433 | 0.0147555 | 30000 | 0 | COMPLETED | 0.405499 | 0.4365 | 12 | [0.405499, 0.436500] |
| mnist | D_10k | Yomo-Matched | 3 | accuracy | 0.241267 | 0.00591974 | 30000 | 0 | COMPLETED | 0.233333 | 0.249901 | 13 | [0.233333, 0.249901] |
| mnist | D_1k | C-CNN | 3 | accuracy | 0.555933 | 0.0625077 | 0 | 0 | COMPLETED | 0.510232 | 0.6257 | 9 | [0.510232, 0.625700] |
| mnist | D_1k | C-LR | 1 | accuracy | 0.7908 |  | 0 | 0 | COMPLETED | 0.7833 | 0.7981 | 5 | [0.783300, 0.798100] |
| mnist | D_1k | C-MLP | 3 | accuracy | 0.614967 | 0.0848143 | 0 | 0 | COMPLETED | 0.527 | 0.6909 | 8 | [0.527000, 0.690900] |
| mnist | D_1k | C-Poly | 1 | accuracy | 0.7821 |  | 0 | 0 | COMPLETED | 0.7743 | 0.7898 | 6 | [0.774300, 0.789800] |
| mnist | D_1k | C-RBF | 1 | accuracy | 0.8193 |  | 0 | 0 | COMPLETED | 0.8123 | 0.8262 | 2 | [0.812300, 0.826200] |
| mnist | D_1k | C-RFF | 3 | accuracy | 0.808767 | 0.0010504 | 0 | 0 | COMPLETED | 0.8018 | 0.815767 | 3 | [0.801800, 0.815767] |
| mnist | D_1k | C-ResNet | 1 | accuracy | 0.9355 |  | 0 | 0 | COMPLETED | 0.9309 | 0.9402 | 1 | [0.930900, 0.940200] |
| mnist | D_1k | C-Trig | 1 | accuracy | 0.8025 |  | 0 | 0 | COMPLETED | 0.7952 | 0.8098 | 4 | [0.795200, 0.809800] |
| mnist | D_1k | PdrQC-Matched | 1 | accuracy | 0.7565 |  | 11000 | 0 | COMPLETED | 0.7486 | 0.7642 | 7 | [0.748600, 0.764200] |
| mnist | D_1k | QF-Product | 1 | accuracy | 0.4511 |  | 11000 | 0 | COMPLETED | 0.4419 | 0.4605 | 10 | [0.441900, 0.460500] |
| mnist | D_1k | QF-Ring | 1 | accuracy | 0.4054 |  | 11000 | 0 | COMPLETED | 0.396297 | 0.4146 | 12 | [0.396297, 0.414600] |
| mnist | D_1k | QK-IQP | 1 | accuracy | 0.1129 |  | 0 | 1.1e+07 | COMPLETED | 0.1069 | 0.1191 | 14 | [0.106900, 0.119100] |
| mnist | D_1k | VQC-DR | 3 | accuracy | 0.419833 | 0.0209218 | 30000 | 0 | COMPLETED | 0.401767 | 0.442233 | 11 | [0.401767, 0.442233] |
| mnist | D_1k | Yomo-Matched | 3 | accuracy | 0.1794 | 0.0139775 | 30000 | 0 | COMPLETED | 0.164566 | 0.192901 | 13 | [0.164566, 0.192901] |
| mnist | D_5k | C-CNN | 3 | accuracy | 0.9072 | 0.00775951 | 0 | 0 | COMPLETED | 0.898167 | 0.915533 | 2 | [0.898167, 0.915533] |
| mnist | D_5k | C-LR | 1 | accuracy | 0.8031 |  | 0 | 0 | COMPLETED | 0.7958 | 0.8105 | 7 | [0.795800, 0.810500] |
| mnist | D_5k | C-MLP | 3 | accuracy | 0.7752 | 0.0184073 | 0 | 0 | COMPLETED | 0.7572 | 0.7935 | 9 | [0.757200, 0.793500] |
| mnist | D_5k | C-Poly | 1 | accuracy | 0.8247 |  | 0 | 0 | COMPLETED | 0.8176 | 0.8319 | 6 | [0.817600, 0.831900] |
| mnist | D_5k | C-RBF | 1 | accuracy | 0.8443 |  | 0 | 0 | COMPLETED | 0.8375 | 0.8511 | 3 | [0.837500, 0.851100] |
| mnist | D_5k | C-RFF | 3 | accuracy | 0.8369 | 0.00150997 | 0 | 0 | COMPLETED | 0.830033 | 0.843701 | 4 | [0.830033, 0.843701] |
| mnist | D_5k | C-ResNet | 1 | accuracy | 0.9637 |  | 0 | 0 | COMPLETED | 0.96 | 0.9673 | 1 | [0.960000, 0.967300] |
| mnist | D_5k | C-Trig | 1 | accuracy | 0.8297 |  | 0 | 0 | COMPLETED | 0.8227 | 0.8367 | 5 | [0.822700, 0.836700] |
| mnist | D_5k | PdrQC-Matched | 1 | accuracy | 0.7934 |  | 15000 | 0 | COMPLETED | 0.786 | 0.8009 | 8 | [0.786000, 0.800900] |
| mnist | D_5k | QF-Product | 1 | accuracy | 0.4969 |  | 15000 | 0 | COMPLETED | 0.4877 | 0.5061 | 10 | [0.487700, 0.506100] |
| mnist | D_5k | QF-Ring | 1 | accuracy | 0.4221 |  | 15000 | 0 | COMPLETED | 0.4131 | 0.4313 | 11 | [0.413100, 0.431300] |
| mnist | D_5k | QK-IQP | 1 | accuracy | 0.1085 |  | 0 | 7.5e+07 | COMPLETED | 0.1025 | 0.114503 | 14 | [0.102500, 0.114503] |
| mnist | D_5k | VQC-DR | 3 | accuracy | 0.413267 | 0.00846719 | 30000 | 0 | COMPLETED | 0.402932 | 0.424633 | 12 | [0.402932, 0.424633] |
| mnist | D_5k | Yomo-Matched | 3 | accuracy | 0.2382 | 0.0104843 | 30000 | 0 | COMPLETED | 0.227067 | 0.249834 | 13 | [0.227067, 0.249834] |
| mnist | D_full | C-CNN | 3 | accuracy | 0.975767 | 0.00660631 | 0 | 0 | COMPLETED | 0.969033 | 0.9822 | 2 | [0.969033, 0.982200] |
| mnist | D_full | C-LR | 1 | accuracy | 0.8056 |  | 0 | 0 | COMPLETED | 0.7983 | 0.8127 | 6 | [0.798300, 0.812700] |
| mnist | D_full | C-MLP | 3 | accuracy | 0.813533 | 0.00591974 | 0 | 0 | COMPLETED | 0.804666 | 0.8221 | 5 | [0.804666, 0.822100] |
| mnist | D_full | C-RFF | 3 | accuracy | 0.8596 | 0.000655744 | 0 | 0 | COMPLETED | 0.8533 | 0.865933 | 3 | [0.853300, 0.865933] |
| mnist | D_full | C-ResNet | 1 | accuracy | 0.9782 |  | 0 | 0 | COMPLETED | 0.9754 | 0.981003 | 1 | [0.975400, 0.981003] |
| mnist | D_full | C-Trig | 1 | accuracy | 0.8411 |  | 0 | 0 | COMPLETED | 0.8341 | 0.848 | 4 | [0.834100, 0.848000] |
| mnist | D_full | PdrQC-Matched | 1 | accuracy | 0.8048 |  | 65000 | 0 | COMPLETED | 0.7974 | 0.8119 | 7 | [0.797400, 0.811900] |
| mnist | D_full | QF-Product | 1 | accuracy | 0.5195 |  | 65000 | 0 | COMPLETED | 0.51 | 0.5287 | 8 | [0.510000, 0.528700] |
| mnist | D_full | QF-Ring | 1 | accuracy | 0.4393 |  | 65000 | 0 | COMPLETED | 0.4304 | 0.4483 | 10 | [0.430400, 0.448300] |
| mnist | D_full | VQC-DR | 3 | accuracy | 0.462067 | 0.0247763 | 30000 | 0 | COMPLETED | 0.4361 | 0.482467 | 9 | [0.436100, 0.482467] |
| mnist | D_full | Yomo-Matched | 3 | accuracy | 0.262533 | 0.00201329 | 30000 | 0 | COMPLETED | 0.2553 | 0.269434 | 11 | [0.255300, 0.269434] |
