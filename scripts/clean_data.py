"""scripts/clean_data.py - clean OAS heavy-chain sequences, then split train/val/test.

Look at the lengths first (writes only a picture, no data):
    python scripts/clean_data.py --inputs a.csv b.csv --out-dir data/processed/explore --hist-only
Real run:
    python scripts/clean_data.py --inputs a.csv b.csv --out-dir <processed/v1> --min-len 100 --max-len 140

Inputs: OAS .csv (column 'sequence_alignment_aa' or 'sequence') or .txt (one sequence per line).
Output: vhh_clean.csv, vhh_clean_meta.csv, train.csv, val.csv, test.csv,
        cleaning_report.csv, length_histogram.png, manifest.json
"""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

COL = "sequence_alignment_aa"
AA_PATTERN = r"[ACDEFGHIKLMNPQRSTVWY]+"  # same alphabet as CONTRACT.md and utils.py


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def load_sequences(path: Path) -> pd.Series:
    """Strict loader: fails loudly instead of guessing."""
    name = path.name.lower()
    if name.endswith((".csv", ".csv.gz")):
        for skip in (0, 1):  # OAS files may start with one JSON metadata line
            header = pd.read_csv(path, skiprows=skip, nrows=0)
            for col in (COL, "sequence"):
                if col in header.columns:
                    return pd.read_csv(path, skiprows=skip, usecols=[col], low_memory=False)[col]
        raise ValueError(f"{path.name}: no column '{COL}' or 'sequence'. First columns: {list(header.columns)[:8]}")
    if name.endswith(".txt"):
        with open(path, encoding="utf-8") as f:
            return pd.Series([ln.strip() for ln in f if ln.strip()], dtype=str)
    raise ValueError(f"Unsupported file type: {path.name} (use .csv, .csv.gz or .txt)")


def save_histogram(lengths: pd.Series, min_len: int, max_len: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.hist(lengths, bins=np.arange(lengths.min(), lengths.max() + 2) - 0.5, color="steelblue")
    plt.axvline(min_len, color="red", linestyle="--", label=f"MIN={min_len}")
    plt.axvline(max_len, color="orange", linestyle="--", label=f"MAX={max_len}")
    plt.xlabel("Sequence length"); plt.ylabel("Count"); plt.title("Length distribution")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-len", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=140)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hist-only", action="store_true", help="only print length stats + save the histogram")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = []

    def log(step: str, n: int) -> None:
        before = report[-1]["rows_after"] if report else n
        report.append({"step": step, "rows_before": before, "rows_after": n, "removed": before - n})
        print(f"{step:<34} {n:>12,}")

    # 1. load
    frames = []
    for p in inputs:
        s = load_sequences(p).dropna().astype(str)
        frames.append(pd.DataFrame({"raw": s.values, "source_file": p.name}))
    df = pd.concat(frames, ignore_index=True)
    log("1. raw rows", len(df))

    # 2. remove gaps, uppercase, strip
    df["sequence"] = df["raw"].str.replace("-", "", regex=False).str.upper().str.strip()
    df = df.drop(columns="raw")
    df = df[df["sequence"] != ""]
    log("2. gaps removed, non-empty", len(df))

    # 3. exactly the 20 amino acids
    ok = df["sequence"].str.fullmatch(AA_PATTERN)
    print("   examples rejected:", df.loc[~ok, "sequence"].head(3).tolist())
    df = df[ok]
    log("3. only the 20 amino acids", len(df))

    # 4. lengths: look first, then filter
    lengths = df["sequence"].str.len()
    print(f"   length min / median / max: {lengths.min()} / {int(lengths.median())} / {lengths.max()}")
    save_histogram(lengths, args.min_len, args.max_len, out_dir / "length_histogram.png")
    print(f"   histogram saved: {out_dir / 'length_histogram.png'}")
    if args.hist_only:
        return
    df = df[lengths.between(args.min_len, args.max_len)]
    log(f"4. length {args.min_len}-{args.max_len}", len(df))

    # 5. exact duplicates (across all files)
    df = df.drop_duplicates(subset="sequence").reset_index(drop=True)
    log("5. exact duplicates removed", len(df))

    # 6. save the clean set
    df[["sequence"]].to_csv(out_dir / "vhh_clean.csv", index=False)
    df[["sequence", "source_file"]].to_csv(out_dir / "vhh_clean_meta.csv", index=False)

    # 7. random 90/5/5 split, strictly disjoint
    n = len(df)
    n_train, n_val = int(0.90 * n), int(0.05 * n)
    if n_val == 0:
        raise ValueError(f"Only {n} sequences after cleaning: too few to split")
    perm = np.random.default_rng(args.seed).permutation(n)
    parts = {
        "train.csv": perm[:n_train],
        "val.csv": perm[n_train:n_train + n_val],
        "test.csv": perm[n_train + n_val:],
    }
    split_seqs = {}
    for fname, idx in parts.items():
        split_seqs[fname] = df["sequence"].iloc[idx]
        split_seqs[fname].to_frame().to_csv(out_dir / fname, index=False)
    assert pd.concat(split_seqs.values()).is_unique, "splits are not disjoint"
    print("\nSplit:", {k: len(v) for k, v in split_seqs.items()})
    print("Per source file kept:\n", df["source_file"].value_counts().to_string())

    # 8. report + manifest
    pd.DataFrame(report).to_csv(out_dir / "cleaning_report.csv", index=False)
    out_files = ["vhh_clean.csv", "train.csv", "val.csv", "test.csv"]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "parameters": {"min_len": args.min_len, "max_len": args.max_len, "seed": args.seed,
                       "split": "90/5/5 random after exact deduplication"},
        "inputs": [{"file": p.name, "sha256": sha256_of(p)} for p in inputs],
        "steps": report,
        "outputs": {f: {"rows": int(len(pd.read_csv(out_dir / f))), "sha256": sha256_of(out_dir / f)}
                    for f in out_files},
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. Everything is in: {out_dir}")


if __name__ == "__main__":
    main()
