# Contributing

Use a virtual environment, install `.[test]`, and run `python -m pytest -q`.
Install `.[images,test]` to include optional PyTorch gradient/resume tests.

Keep generated data and artifacts under ignored `runs/` directories. Never
commit credentials, raw datasets, checkpoints, features or individual predictions.
Do not modify `results/reference/` as a side effect of a code refactor. A
scientific correction requires a separately named analysis, documented protocol
change, source hashes and comparison with the preserved original outputs.

Code and documentation should be in English. Explain numerical conventions,
measurement-budget accounting and statistical families in changes that affect
them. Do not make quantum-advantage or clinical-performance claims from a demo.

Contributions are distributed under the [MIT License](LICENSE).
