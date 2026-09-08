# What was run

The benchmark evaluated MNIST, Fashion-MNIST, CIFAR-10, and BreastMNIST through the shared frozen ResNet-18, train-only standardization/PCA8, and eight-angle interface. Fourteen methods were validation-selected: six quantum-derived families, six matched PCA8 classical controls, and two full-image classical references.

There were 312 primary validation trials, 666 scale-specific validation trials, 96 exact primary seed runs, 1,500 finite-shot model/measurement-seed runs, and 360 data-scaling rows (351 completed and nine frozen scaling boundaries). Exact, finite-shot, calibration, predictive-entropy, model-uncertainty, quantum-measurement-uncertainty, risk-coverage, bootstrap, permutation, and rank-stability analyses were executed.

All quantum results are statevector or offline finite-shot simulations. No real QPU was used.
