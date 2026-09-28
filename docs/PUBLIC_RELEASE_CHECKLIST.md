# Release checks

Run from the repository root after installing `.[test]`:

```bash
python -m pip check
python -m pytest -q
python scripts/check_reference_results.py
python scripts/release_audit.py
```

- Keep the paper title, authors, acceptance status, and citation synchronized in
  `README.md` and `CITATION.cff`. Add the paper URL and DOI when available.
- Include `LICENSE` in source and package distributions.
- Keep datasets, weights, checkpoints, feature caches, and individual predictions
  outside version control.
- Review staged files and Git history for credentials and unintended artifacts.
- Run the optional PyTorch tests with `.[images,test]` installed when changing
  training or differentiable quantum models.
- Preserve reference results and their checksums. Record numerical changes for
  new experiments in `PORTING_NOTES.md`.
