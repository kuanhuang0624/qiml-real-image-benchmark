"""Minimal deterministic CSV-to-Markdown utilities."""
from __future__ import annotations

import csv
from pathlib import Path


def csv_to_markdown(source: Path, destination: Path) -> None:
    with source.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"empty table: {source}")
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    def render(row: list[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)) + " |"
    lines = [render(rows[0]), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(render(row) for row in rows[1:])
    destination.write_text("\n".join(lines) + "\n")
