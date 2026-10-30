"""
prepare_vhh.py
---------------
Prepare OAS / OAS-nano VHH sequences using ONLY six filtering layers:

L1  Remove alignment gaps and empty sequences.
L2  Keep only the 20 standard amino acids.
L3  Apply available OAS quality flags.
L4  Apply the overall sequence-length limits from utils.py.
L5  Check that the seven annotated regions reconstruct the full sequence.
L6  Check the professor's scaffold-specific cysteine rules.

Important notes about L5 and L6
--------------------------------
- L5 is skipped (with a note) unless ALL seven region columns are present.
- L6 is skipped (with a note) unless the required region columns are present.
- L6 assumes the supplied region strings use the professor's IMGT region layout:
    FWR1: 25 residues; IMGT position 10 is a gap.
    CDR1: 8 residues; IMGT positions 31-34 are gaps.
    FWR2: 17 residues.
    FWR3: 38 residues; IMGT position 73 is a gap.
  Therefore:
    Cys23 = FWR1 residue index 21 (zero-based)
    Cys38 = final residue of CDR1
    Cys55 = final residue of FWR2
    Cys104 = final residue of FWR3
- This is a sequence-based scaffold consistency check, NOT proof that a
  disulfide bond physically forms. Confirm the region convention/numbering
  on real OAS rows before relying on L6.
- Optional CDR3 disulfide configurations accepted:
    Cys38 + exactly one CDR3 cysteine, OR
    Cys55 + exactly one CDR3 cysteine.
  Cys38 and Cys55 are not both accepted together in this first version.
  Any other/unpaired cysteine pattern is rejected.
- Missing OAS flag values are treated as UNKNOWN, not as a pass or failure.
  Available flag columns are applied; missing columns/values are reported.

After L6, an OUTPUT STAGE (not a filter layer) removes exact duplicate
sequences: the first copy is kept and `n_occurrences` records how many copies
existed. NB02 requires zero exact duplicates. Use --no-dedup to disable it.

Outputs in <processed>/<DATA_VERSION>/:
    vhh_clean.csv       one column: sequence, exact-unique (for NB02)
    vhh_clean_meta.csv  sequence + n_occurrences + metadata/regions (for NB01)
    cleaning_report.csv rows before/after/removed at each of L1-L6
    length_histogram_before_length_filter.csv
    manifest.json       parameters, input/output hashes, report and notes

Examples:
    python scripts/prepare_vhh.py --inspect
    python scripts/prepare_vhh.py --max-rows 50000
    python scripts/prepare_vhh.py
    python scripts/prepare_vhh.py --inputs data/raw/a.csv data/raw/b.csv.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import MAX_LEN, MIN_LEN, STANDARD_AA, get_data_version, get_paths  # noqa: E402

REQUIRED = "sequence_alignment_aa"

REGIONS = [
    "fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa",
    "fwr3_aa", "cdr3_aa", "fwr4_aa",
]

EXTRA = [
    "junction_aa", "v_call", "d_call", "j_call", "productive",
    "locus", "stop_codon", "vj_in_frame", "v_frameshift",
    "complete_vdj", "v_identity",
]

WANTED = {REQUIRED, *REGIONS, *EXTRA}

BAD_FLAG_VALUES = {
    "productive": {"F", "FALSE"},
    "stop_codon": {"T", "TRUE"},
    "vj_in_frame": {"F", "FALSE"},
    "v_frameshift": {"T", "TRUE"},
}

STEP_NAMES = [
    "L1 gaps removed, empty dropped",
    "L2 alphabet (20 standard amino acids)",
    "L3 available OAS quality flags",
    "L4 overall length window",
    "L5 region integrity",
    "L6 scaffold-specific cysteine rules",
]


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def read_meta_line(path: Path) -> tuple[dict, int]:
    """Read OAS's optional first-line JSON metadata."""
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        first = handle.readline().rstrip("\n")

    if REQUIRED in first.replace('"', "").split(","):
        return {}, 0

    for candidate in (first, first.strip('"').replace('""', '"')):
        try:
            parsed = json.loads(candidate)
            return (parsed if isinstance(parsed, dict) else {}), 1
        except json.JSONDecodeError:
            continue

    # OAS files commonly have a metadata line. If it is not JSON, skip it
    # and let the header inspection below give a useful error if necessary.
    return {}, 1


def clean_region(series: pd.Series) -> pd.Series:
    """Normalize region strings in the same way as the full sequence."""
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("-", "", regex=False)
    )


def clean_chunk(
    chunk: pd.DataFrame,
    args: argparse.Namespace,
    removed: Counter,
    notes: set,
    length_histogram: Counter,
    stats: Counter,
) -> pd.DataFrame:
    df = chunk.copy()

    def apply_filter(step_name: str, mask: pd.Series) -> None:
        nonlocal df
        mask = mask.fillna(False).astype(bool)
        removed[step_name] += int((~mask).sum())
        df = df.loc[mask].copy()

    # L1 — remove gaps, normalize case, remove empty sequences.
    df["sequence"] = (
        df[REQUIRED].fillna("").astype(str).str.strip().str.upper()
        .str.replace("-", "", regex=False)
    )
    df["had_gaps"] = (
        chunk[REQUIRED].fillna("").astype(str).str.contains("-", regex=False)
    )

    for region in REGIONS:
        if region in df.columns:
            df[region] = clean_region(df[region])

    apply_filter(STEP_NAMES[0], df["sequence"].ne(""))
    if df.empty:
        return df

    # L2 — only the 20 standard amino acids.
    valid_alphabet = df["sequence"].str.fullmatch(f"[{STANDARD_AA}]+")
    apply_filter(STEP_NAMES[1], valid_alphabet)
    if df.empty:
        return df

    # L3 — apply each available OAS flag. Missing columns/values are UNKNOWN.
    oas_ok = pd.Series(True, index=df.index, dtype=bool)

    for column, bad_values in BAD_FLAG_VALUES.items():
        if column not in df.columns:
            notes.add(f"L3: column '{column}' missing; this check was not available.")
            continue

        values = df[column].fillna("").astype(str).str.strip().str.upper()
        missing = values.eq("")
        if missing.any():
            notes.add(
                f"L3: {int(missing.sum())} rows have missing '{column}' values; "
                "those rows were not rejected by this flag."
            )

        oas_ok &= ~values.isin(bad_values)

    apply_filter(STEP_NAMES[2], oas_ok)
    if df.empty:
        return df

    # L4 — overall sequence length. Record distribution before applying bounds.
    df["length"] = df["sequence"].str.len()
    length_histogram.update(df["length"].value_counts().to_dict())
    apply_filter(
        STEP_NAMES[3],
        df["length"].between(args.min_len, args.max_len, inclusive="both"),
    )
    if df.empty:
        return df

    df["n_cys"] = df["sequence"].str.count("C")

    # L5 — all seven region columns must exist; otherwise skip without error.
    missing_regions = [region for region in REGIONS if region not in df.columns]
    if missing_regions:
        notes.add(
            "L5 skipped: missing region column(s): " + ", ".join(missing_regions)
        )
    else:
        rebuilt = df[REGIONS[0]]
        for region in REGIONS[1:]:
            rebuilt = rebuilt + df[region]
        apply_filter(STEP_NAMES[4], rebuilt.eq(df["sequence"]))
        if df.empty:
            return df

    # L6 — scaffold-specific cysteine checks.
    # We require these regions to interpret the professor's positions.
    cys_required_regions = ["fwr1_aa", "cdr1_aa", "fwr2_aa", "fwr3_aa", "cdr3_aa"]
    missing_cys_regions = [
        region for region in cys_required_regions if region not in df.columns
    ]

    if missing_cys_regions:
        notes.add(
            "L6 skipped: cannot apply scaffold cysteine rules; missing region "
            "column(s): " + ", ".join(missing_cys_regions)
        )
        return df

    # Professor's IMGT scaffold mapping after removing alignment-gap characters:
    # FWR1 has 25 residues; IMGT position 10 is absent; Cys23 -> index 21.
    # CDR1 position 38 is its final residue.
    # FWR2 position 55 is its final residue.
    # FWR3 position 104 is its final residue.
    fwr1 = df["fwr1_aa"]
    cdr1 = df["cdr1_aa"]
    fwr2 = df["fwr2_aa"]
    fwr3 = df["fwr3_aa"]
    cdr3 = df["cdr3_aa"]

    # If region lengths do not match the professor's stated layout, do not
    # guess at positional mapping: skip L6 for this chunk and report it.
    expected_lengths = (
        fwr1.str.len().eq(25)
        & cdr1.str.len().eq(8)
        & fwr2.str.len().eq(17)
        & fwr3.str.len().eq(38)
    )
    stats["l6_rows"] += len(df)
    stats["l6_layout_match"] += int(expected_lengths.sum())
    if not expected_lengths.all():
        notes.add(
            "L6 skipped for rows whose FWR1/CDR1/FWR2/FWR3 lengths do not match "
            "the professor's stated IMGT scaffold layout (25/8/17/38 residues)."
        )

    # Exact locations under that stated layout.
    cys23 = fwr1.str.len().eq(25) & fwr1.str[21].eq("C")
    cys38 = cdr1.str.len().eq(8) & cdr1.str.endswith("C")
    cys55 = fwr2.str.len().eq(17) & fwr2.str.endswith("C")
    cys104 = fwr3.str.len().eq(38) & fwr3.str.endswith("C")
    cdr3_cys_count = cdr3.str.count("C")

    # Canonical Cys23-Cys104 pair is mandatory.
    canonical_pair = cys23 & cys104

    # Optional pairs: exactly one of Cys38 or Cys55, plus exactly one CDR3 Cys.
    has_one_optional_partner = cys38.astype(int) + cys55.astype(int) == 1
    optional_pair = has_one_optional_partner & cdr3_cys_count.eq(1)

    # No optional framework cysteine and no CDR3 cysteine is also valid.
    no_optional_pair = (~cys38) & (~cys55) & cdr3_cys_count.eq(0)

    # Any other cysteine pattern is unsupported by the professor's stated rules.
    # Also require the total Cys count to match the accepted pairing configuration,
    # preventing extra cysteines elsewhere in the sequence from passing.
    expected_cys_count = pd.Series(2, index=df.index)
    expected_cys_count.loc[optional_pair] = 4

    accepted_pattern = no_optional_pair | optional_pair
    valid_layout = expected_lengths

    # Rows with unsupported region lengths are not filtered by an assumed
    # positional mapping; they are retained and clearly reported for review.
    cys_ok = (
        (~valid_layout)
        | (
            canonical_pair
            & accepted_pattern
            & df["n_cys"].eq(expected_cys_count)
        )
    )

    apply_filter(STEP_NAMES[5], cys_ok)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--inputs", nargs="+", help="Raw OAS CSV/CSV.GZ input files.")
    parser.add_argument("--inspect", action="store_true", help="Print file columns and exit.")
    parser.add_argument("--min-len", type=int, default=MIN_LEN)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--max-rows", type=int, default=0, help="Read first N rows per file.")
    parser.add_argument("--chunk", type=int, default=200_000)
    parser.add_argument("--no-dedup", action="store_true",
                        help="Keep exact duplicates (NB02 will refuse such a file).")
    args = parser.parse_args()

    if args.min_len < 1 or args.max_len < args.min_len:
        parser.error("--min-len and --max-len must be positive and min <= max.")

    paths = get_paths()
    version = get_data_version()
    output_dir = paths["processed"] / version
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = (
        [Path(path) for path in args.inputs]
        if args.inputs
        else sorted([*paths["raw"].glob("*.csv"), *paths["raw"].glob("*.csv.gz")])
    )
    if not input_files:
        raise FileNotFoundError(f"No raw OAS files found in {paths['raw']}")

    if args.inspect:
        for path in input_files:
            metadata, skip_rows = read_meta_line(path)
            header = list(pd.read_csv(path, skiprows=skip_rows, nrows=0).columns)
            print(f"\n{path.name}")
            print(f"  metadata: {metadata}")
            print(f"  columns ({len(header)}): {header}")
            print(f"  region columns present: {[c for c in REGIONS if c in header]}")
            print(f"  OAS flag columns present: {[c for c in BAD_FLAG_VALUES if c in header]}")
        return

    removed: Counter = Counter()
    notes: set[str] = set()
    length_histogram: Counter = Counter()
    stats: Counter = Counter()
    survivors = []
    input_info = []
    total_rows = 0

    for path in input_files:
        metadata, skip_rows = read_meta_line(path)
        header = list(pd.read_csv(path, skiprows=skip_rows, nrows=0).columns)

        if REQUIRED not in header:
            raise ValueError(
                f"{path.name}: required sequence column '{REQUIRED}' not found. "
                f"First columns: {header[:12]}"
            )

        rows_read = 0
        rows_survived = 0

        reader = pd.read_csv(
            path,
            skiprows=skip_rows,
            usecols=lambda column: column in WANTED,
            dtype=str,
            keep_default_na=False,
            chunksize=args.chunk,
            nrows=args.max_rows or None,
        )

        for chunk in reader:
            rows_read += len(chunk)
            cleaned = clean_chunk(
                chunk, args, removed, notes, length_histogram, stats
            )
            if not cleaned.empty:
                cleaned["source_file"] = path.name
                cleaned["species"] = str(metadata.get("Species", ""))
                cleaned["isotype"] = str(metadata.get("Isotype", ""))
                cleaned["subject"] = str(metadata.get("Subject", ""))
                survivors.append(cleaned)
                rows_survived += len(cleaned)

        total_rows += rows_read
        print(
            f"{path.name}: {rows_read:,} rows -> {rows_survived:,} rows "
            f"after L1-L6; species={metadata.get('Species', '?')}, "
            f"isotype={metadata.get('Isotype', '?')}"
        )
        input_info.append(
            {
                "file": path.name,
                "sha256": sha256_of(path),
                "rows_read": rows_read,
                "metadata": metadata,
            }
        )

    if not survivors:
        raise ValueError(
            "No sequences survived. Check --inspect, the length limits, "
            "and the cleaning_report."
        )

    df = pd.concat(survivors, ignore_index=True)
    del survivors

    # Output stage (not a filter layer): exact deduplication, abundance kept.
    n_after_l6 = len(df)
    df["n_occurrences"] = df.groupby("sequence")["sequence"].transform("size")
    if not args.no_dedup:
        df = df.drop_duplicates(subset="sequence", keep="first").reset_index(drop=True)

    if stats["l6_rows"]:
        notes.add(
            f"L6 coverage: {stats['l6_layout_match']:,}/{stats['l6_rows']:,} rows "
            f"({stats['l6_layout_match'] / stats['l6_rows']:.1%}) had the 25/8/17/38 "
            "layout and were checked; the others were kept unchecked."
        )

    front_columns = ["sequence", "length", "n_cys", "n_occurrences"]
    ordered_columns = (
        [column for column in front_columns if column in df.columns]
        + [
            column for column in df.columns
            if column not in front_columns and column != REQUIRED
        ]
    )
    df = df.drop(columns=[REQUIRED], errors="ignore")
    df = df[[column for column in ordered_columns if column in df.columns]]

    report_rows = [
        {
            "step": "L0 raw rows",
            "rows_before": total_rows,
            "rows_after": total_rows,
            "removed": 0,
        }
    ]
    current_rows = total_rows
    for step_name in STEP_NAMES:
        removed_count = removed.get(step_name, 0)
        report_rows.append(
            {
                "step": step_name,
                "rows_before": current_rows,
                "rows_after": current_rows - removed_count,
                "removed": removed_count,
            }
        )
        current_rows -= removed_count

    report = pd.DataFrame(report_rows)
    if current_rows != n_after_l6:
        raise RuntimeError(
            f"Cleaning report mismatch: report says {current_rows:,} rows, "
            f"but output contains {n_after_l6:,} rows before dedup. "
            "This can happen if a layer was skipped inconsistently."
        )

    if not args.no_dedup:
        report = pd.concat([report, pd.DataFrame([{
            "step": "Exact dedup (output stage)", "rows_before": n_after_l6,
            "rows_after": len(df), "removed": n_after_l6 - len(df),
        }])], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = output_dir / "vhh_clean.csv"
    meta_path = output_dir / "vhh_clean_meta.csv"
    report_path = output_dir / "cleaning_report.csv"
    histogram_path = output_dir / "length_histogram_before_length_filter.csv"
    manifest_path = output_dir / "manifest.json"

    df[["sequence"]].to_csv(clean_path, index=False)
    df.to_csv(meta_path, index=False)
    report.to_csv(report_path, index=False)
    (
        pd.Series(length_histogram, dtype="int64")
        .sort_index()
        .rename_axis("length")
        .rename("n_sequences")
        .to_csv(histogram_path)
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/prepare_vhh.py",
        "git_commit": git_commit(ROOT),
        "data_version": version,
        "parameters": {
            "min_len": args.min_len,
            "max_len": args.max_len,
            "alphabet": STANDARD_AA,
            "max_rows_per_file": args.max_rows,
            "chunk_size": args.chunk,
            "deduplication": not args.no_dedup,
            "n_rows_before_dedup": n_after_l6,
        },
        "inputs": input_info,
        "stats": dict(stats),
        "steps": report.to_dict("records"),
        "notes": sorted(notes),
        "outputs": {
            clean_path.name: {"sha256": sha256_of(clean_path)},
            meta_path.name: {"sha256": sha256_of(meta_path)},
            report_path.name: {"sha256": sha256_of(report_path)},
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print("\n" + report.to_string(index=False))
    print(
        f"\nFINAL: {len(df):,} rows retained; "
        f"sequence length range {df['length'].min()}-{df['length'].max()}."
    )
    print(f"Output directory: {output_dir}")
    for note in sorted(notes):
        print(f"NOTE: {note}")


if __name__ == "__main__":
    main()
