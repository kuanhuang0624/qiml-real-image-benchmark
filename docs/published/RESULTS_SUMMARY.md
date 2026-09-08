# Results summary

At the controlled primary scale, validation-selected PdrQC-Matched was the strongest quantum-derived method on all four datasets. Its exact primary metric was 0.7934 accuracy on MNIST, 0.7718 on Fashion-MNIST, 0.6810 on CIFAR-10, and 0.78467 AUROC on BreastMNIST.

The frozen H1 matched control, C-RBF, was higher by 0.0219 accuracy on Fashion-MNIST and 0.0350 on CIFAR-10; both differences survived the global twelve-test Holm correction. The BreastMNIST frozen H1 difference was 0.01065 AUROC with a 95% interval spanning zero and Holm p=0.8619. No statistically detectable difference was observed there; this is not an equivalence result.

All eight natural-image contrasts survived global Holm correction. In both natural datasets, Ring was lower than Product at 128 preparations, Yomo-Matched was lower than VQC-DR at eight preparations, and PdrQC-Matched was higher than Ring at 128 preparations. None of the four BreastMNIST primary contrasts survived Holm correction.

The full-image C-ResNet reference was highest on all datasets. These results do not establish quantum advantage, medical superiority, causal benefit from entanglement, or hardware efficiency.
