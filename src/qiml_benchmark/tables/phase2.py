"""Generate the shared-interface audit table."""
from __future__ import annotations

import csv
from pathlib import Path

from .common import csv_to_markdown

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()


def main() -> None:
    source = ROOT / "data_manifests" / "preprocessing_hashes.csv"
    target = ROOT / "results" / "tables" / "table_02_feature_interface.csv"
    keep = ["dataset", "training_scale", "backbone", "pretraining_source", "raw_feature_dimension", "PCA_dimension", "explained_variance", "maximum_clipping_rate", "preprocessing_hash"]
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keep, lineterminator="\n")
        writer.writeheader(); writer.writerows({key: row[key] for key in keep} for row in rows)
    csv_to_markdown(target, target.with_suffix(".md"))


if __name__ == "__main__":
    main()
