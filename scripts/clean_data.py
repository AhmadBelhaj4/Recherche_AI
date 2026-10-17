"""
scripts/clean_data.py
=====================
Cleans OAS antibody sequences and splits them into Train / Val / Test.

Pipeline:
1. Load CSV, TXT, or FASTA files.
2. Remove "-" gaps and whitespace.
3. Keep only the 20 standard amino acids (^[ACDEFGHIKLMNPQRSTVWY]+$).
4. Keep length within [MIN_LEN, MAX_LEN] (100-140).
5. Remove exact duplicates.
6. Random 90/5/5 split with seed 42 (professor's plan).
7. Save vhh_clean.csv, train.csv, val.csv, test.csv.
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

COL = "sequence_alignment_aa"
AA_REGEX = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def load_sequences_from_file(path: Path) -> list:
    """Load sequences from CSV, TSV, FASTA, or plain TXT file."""
    path = Path(path)
    if not path.exists():
        return []

    if path.suffix.lower() == ".csv":
        try:
            d = pd.read_csv(path, low_memory=False)
            if COL not in d.columns and "sequence" not in d.columns:
                d = pd.read_csv(path, skiprows=1, low_memory=False)
            col_to_use = COL if COL in d.columns else ("sequence" if "sequence" in d.columns else None)
            if col_to_use:
                return d[col_to_use].dropna().astype(str).tolist()
        except Exception:
            pass

    # Fallback / TXT / FASTA handling: read line-by-line
    seqs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            # If CSV row, take first or appropriate field
            if "," in line:
                parts = line.split(",")
                line = parts[0].strip()
            seqs.append(line)
    return seqs


def clean_and_split(
    input_paths,
    out_dir: Path,
    min_len: int = 100,
    max_len: int = 140,
    seed: int = 42,
    split_ratios=(0.90, 0.05, 0.05),
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_raw = []
    for p in input_paths:
        seqs = load_sequences_from_file(p)
        all_raw.extend(seqs)

    if not all_raw:
        raise ValueError(f"No sequences found in input paths: {input_paths}")

    df = pd.DataFrame({"raw_sequence": all_raw})
    print(f"1. Raw input rows: {len(df):,}")

    # 2. Remove gaps "-" and clean
    df["sequence"] = (
        df["raw_sequence"].astype(str).str.replace("-", "", regex=False).str.upper().str.strip()
    )
    df = df[df["sequence"] != ""]
    print(f"2. Gaps removed, non-empty: {len(df):,}")

    # 3. Keep only standard 20 amino acids
    mask_20 = df["sequence"].apply(lambda s: bool(AA_REGEX.match(s)))
    df = df[mask_20]
    print(f"3. Valid 20 amino acids: {len(df):,}")

    # 4. Length filtering
    lengths = df["sequence"].str.len()
    df = df[lengths.between(min_len, max_len)]
    print(f"4. Length {min_len}-{max_len}: {len(df):,}")

    # 5. Exact deduplication
    df = df.drop_duplicates(subset="sequence").reset_index(drop=True)
    print(f"5. Exact deduplication: {len(df):,}")

    # Save full clean
    clean_csv = out_dir / "vhh_clean.csv"
    df[["sequence"]].to_csv(clean_csv, index=False)
    print(f"Saved: {clean_csv}")

    # 6. Random 90/5/5 split (seed 42)
    np.random.seed(seed)
    n = len(df)
    indices = np.random.permutation(n)
    n_train = int(split_ratios[0] * n)
    n_val = int(split_ratios[1] * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_df = df.iloc[train_idx][["sequence"]]
    val_df = df.iloc[val_idx][["sequence"]]
    test_df = df.iloc[test_idx][["sequence"]]

    # Save splits
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print(f"\nSplits (seed={seed}):")
    print(f"  Train : {len(train_df):,} rows -> {out_dir / 'train.csv'}")
    print(f"  Val   : {len(val_df):,} rows -> {out_dir / 'val.csv'}")
    print(f"  Test  : {len(test_df):,} rows -> {out_dir / 'test.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and split OAS antibody data")
    parser.add_argument("--inputs", nargs="+", help="Input CSV/TXT files", default=["data/toy/vhh_unique.txt"])
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    clean_and_split(args.inputs, args.out_dir, seed=args.seed)
