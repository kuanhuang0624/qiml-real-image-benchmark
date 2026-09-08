# Method provenance

Frozen source-audit date: 2026-08-12.

| Method | Source/version/status | Reproduction status | Preserved mechanism | Declared departure |
|---|---|---|---|---|
| QK-IQP | Havlíček et al., Nature 567 (2019), DOI 10.1038/s41586-019-0980-2, peer reviewed | matched gate-level adaptation | two-repeat data-dependent IQP/ZZ map and fidelity kernel | eight shared ResNet/PCA angles and circular nearest-neighbor graph replace the source problem |
| VQC-DR | Pérez-Salinas et al., Quantum 4, 226 (2020), peer reviewed | matched adaptation | repeated data upload, trainable rotations, shallow entangling layers | fixed eight-qubit/two-block multiclass Z-readout with shared benchmark input |
| Yomo-Matched | Liu et al., arXiv:2509.20090v1 (2025), preprint | matched adaptation | computational-basis probability aggregation, class partition, CE + sharpening + entropy loss, single/low-shot voting | frozen ResNet/PCA interface, eight qubits and two blocks replace source CNN and five-block experiment |
| PdrQC-Matched | Tu et al., arXiv:2604.16877v2 (revised 2026-08-08), preprint | matched adaptation | progressive upload selection, all weight≤2 Pauli candidates, discriminative B/(W+eps) selection, shared sparse multiclass representation, greedy QWC grouping | compact frozen candidate-depth budget and shared angle map; no official code was found |
| QF-Product | benchmark specification | exact local implementation | two RY/RZ upload blocks, no entanglers, full 48-feature readout | none |
| QF-Ring | benchmark specification | exact local implementation | QF-Product plus matched circular CZ layers, same 48 readout | none |

The visual benchmark design follows Innan, Rehman, and Shafique, arXiv:2605.19417v1: frozen ResNet-18 extraction, shared preprocessing and resource-aware cross-family reporting. Their preprint evaluates DQN-QTL, QPIE-inspired QTL, AE-CQTL, PVCQTL, and ED-QTL mainly on Fashion-MNIST and Ants-vs-Bees, with selected CIFAR-10 evidence. This project does not reproduce those families. It instead adds nested scaling, calibration, hierarchical uncertainty, paired randomization, and four explicitly controlled representation families. Its source reports metrics and resources but has limited uncertainty/significance analysis and incomplete repeated-run coverage.

The six conceptual papers are used only to separate preparation, measurements, structured representations, local correlations, snapshot budgets, and classical surrogates. They are not image-classification baselines.

## Prior-code provenance

| Prior file | Original SHA-256 | Use |
|---|---|---|
| `post_a4.../src/circuits/statevector.py` | `5665f5aba429d14c857bad36e58e2e851a6e23c097c46c14563728f73f018083` | inspected for qubit ordering and vectorized gate application; new implementation will add required circuits/readouts |
| `post_a4.../src/evaluation/metrics.py` | `ac9ff0fa78cb700c994a293faff0221bf2459955c7d339b305fa5aea276149c1` | metric naming lesson only; calibration/uncertainty code is new |
| `post_a4.../src/data/fashion.py` | `2d319223466f67ff31182a2ee7755d39855a068d8777f30303126ded48fb8d01` | IDX parsing lesson; new generic loaders and split manifests are required |

No file has yet been copied verbatim. Every eventual modification is represented by the new repository history.
