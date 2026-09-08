"""Generate the dataset/split audit table from immutable manifests."""
from __future__ import annotations

import shutil
from pathlib import Path

from .common import csv_to_markdown

from qiml_benchmark.paths import workspace_root
ROOT = workspace_root()


def main() -> None:
    source = ROOT / "data_manifests" / "dataset_versions.csv"
    target = ROOT / "results" / "tables" / "table_01_dataset_splits.csv"
    shutil.copyfile(source, target)
    csv_to_markdown(target, target.with_suffix(".md"))


if __name__ == "__main__":
    main()
