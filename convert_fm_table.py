"""Convert a Football Manager "print-to-screen" pipe-table export into the same
comma-delimited CSV format the ScoutRAG project uses (fmdata24llm.csv).

The pretty-printed FM export looks like:

    "| Name              | Inf | Age | ... |"
    "| ----------------------------------- |"
    "| Rade Krunic - Bosnian | nEU | 29 | ... |"
    "| ----------------------------------- |"

i.e. every row is one quoted line whose columns are separated by '|', with
dashed separator lines between rows and many blank lines. This script flattens
that into a normal CSV with the identical 58-column schema, so the row can be
fed straight into the ScoutRAG engine (load_and_clean / build_profiles).

Usage:
    python convert_fm_table.py INPUT.csv OUTPUT.csv

Example:
    python convert_fm_table.py fenerbahce_only.csv fenerbahce_clean.csv
"""

from __future__ import annotations

import csv
import sys


def _split_row(line: str) -> list[str] | None:
    """Return the cell values of one pipe-table line, or None if it's not a data
    row (blank line, BOM line, or a dashed separator)."""
    s = line.strip()
    if not s or s in ('"﻿"', '﻿'):
        return None
    s = s.strip('"').strip()
    if not s.startswith("|"):
        return None
    # a separator line is only dashes/pipes/spaces
    if set(s) <= set("-| "):
        return None
    # drop the leading/trailing empty cell created by the outer pipes
    cells = [c.strip() for c in s.split("|")]
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def convert(in_path: str, out_path: str) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    header = None
    rows = []
    for line in lines:
        cells = _split_row(line)
        if cells is None:
            continue
        if header is None:
            # first real row is the column header (starts with "Name")
            if cells and cells[0].lower() == "name":
                header = cells
            continue
        # skip the FM placeholder row "- -" that carries no real player
        if cells and cells[0] in ("- -", "-"):
            continue
        # pad/trim to the header width so csv stays rectangular
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        elif len(cells) > len(header):
            cells = cells[: len(header)]
        rows.append(cells)

    if header is None:
        raise SystemExit("Could not find a header row (a '| Name | ... |' line).")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"Converted {in_path} -> {out_path}")
    print(f"  {len(rows)} player rows, {len(header)} columns")


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python convert_fm_table.py INPUT.csv OUTPUT.csv")
        raise SystemExit(1)
    convert(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
