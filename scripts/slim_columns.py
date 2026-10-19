"""scripts/slim_columns.py - keep only the columns the project needs.

It does NOTHING else: no filtering, no cleaning, no dedup, no reordering of
rows. Row i of the output is row i of the input, with fewer columns.
The cleaning/filtering is done in NB01.

The JSON metadata line (line 1 of OAS files: species, study, ...) is copied
unchanged to line 1 of the output, so no information is lost.
The original file is NEVER modified or overwritten.

Usage (Colab):
    python scripts/slim_columns.py --inputs /content/drive/MyDrive/AI_Project_Shared/data/raw/*.csv \
        --out-dir /content/drive/MyDrive/AI_Project_Shared/data/processed/slim
"""
import argparse
import gzip
from pathlib import Path

import pandas as pd

REQUIRED = "sequence_alignment_aa"
KEEP = [
    REQUIRED,
    "fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa", "cdr3_aa", "fwr4_aa",
    "junction_aa",
    "v_call", "d_call", "j_call",
    "productive", "locus",
]


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.name.lower().endswith(".gz") \
        else open(path, "r", encoding="utf-8")


def slim_one(path: Path, out_dir: Path, keep: list) -> None:
    with open_text(path) as f:
        first = f.readline().rstrip("\n")
    has_meta = REQUIRED not in first.split(",")
    skip = 1 if has_meta else 0
    header = list(pd.read_csv(path, skiprows=skip, nrows=0).columns)

    if REQUIRED not in header:
        raise ValueError(f"{path.name}: column '{REQUIRED}' not found. First columns: {header[:8]}")
    cols = [c for c in keep if c in header]
    missing = [c for c in keep if c not in header]
    if missing:
        print(f"  WARNING {path.name}: columns not found (skipped): {missing}")

    name = path.name[:-3] if path.name.lower().endswith(".gz") else path.name
    out = out_dir / name
    if out.resolve() == path.resolve():
        raise ValueError("output would overwrite the input file")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with open(out, "w", encoding="utf-8", newline="") as fo:
        if has_meta:
            fo.write(first + "\n")
        for i, chunk in enumerate(pd.read_csv(path, skiprows=skip, usecols=cols, dtype=str,
                                              keep_default_na=False, na_filter=False,
                                              chunksize=100_000)):
            chunk = chunk[cols]                      # fixed column order
            chunk.to_csv(fo, index=False, header=(i == 0))
            n_rows += len(chunk)

    n_in = sum(len(c) for c in pd.read_csv(path, skiprows=skip, usecols=[REQUIRED], dtype=str,
                                           chunksize=200_000))
    assert n_in == n_rows, f"row count changed: {n_in} -> {n_rows}"
    mb = lambda p: p.stat().st_size / 1e6
    print(f"{path.name}: {len(header)} -> {len(cols)} columns | {n_rows:,} rows (unchanged) | "
          f"{mb(path):.1f} MB -> {mb(out):.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep", nargs="+", default=KEEP, help="override the column list")
    args = ap.parse_args()
    for p in args.inputs:
        slim_one(Path(p), Path(args.out_dir), args.keep)


if __name__ == "__main__":
    main()