# PCA16 Resource Estimate

The required small-batch 16-qubit dense-statevector audit completed successfully for both datasets and agreed with the exact analytic RY expectation calculation to numerical tolerance.

- Dense audit examples: 8 total (4 per dataset)
- Dense audit observables per example: 12
- Dense audit wall time: 0.114554 seconds
- Statevector dimension: 65,536
- Full experiment backend: exact analytic product-state expectations, scientifically identical for the frozen depth-1 RY upload
- Observed process peak memory during all smoke tests: 0.184 GiB
- Estimated full extension CPU-hours: less than 2
- Estimated full extension GPU-hours: 0
- Resource-review threshold exceeded: no

The analytic evaluator is not an approximate simulator. It directly computes the exact Pauli expectations of the frozen product RY state and is independently checked against dense statevectors.
