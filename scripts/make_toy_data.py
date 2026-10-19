"""Generate a small FAKE raw OAS-like file for local development and CI.

Output (the only thing this script writes):
    data/toy/raw/toy_oas_raw.csv

The file imitates a real OAS download on purpose, so the toy run exercises the
same code path as the real run:
  * line 1 = JSON metadata line, line 2 = CSV header (like OAS)
  * column 'sequence_alignment_aa' + region / gene columns + junk columns
  * deliberately DIRTY rows: duplicates, gaps '-', ambiguous 'X', too short,
    too long, empty sequences  -> NB01 must remove them and report counts.

Everything after this (slim -> NB01 clean -> NB02 split) is produced by the
pipeline itself into data/toy/processed/ (git-ignored, regenerable).

Usage:  python scripts/make_toy_data.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

SEED = 42
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
N_VALID = 200          # clean, unique, length 110-130
N_DUP = 10             # exact duplicates of valid rows
N_GAP = 8              # contain '-'  (valid once the gaps are removed)
N_X = 6                # contain 'X'  (must be rejected)
N_SHORT = 6            # length < 100
N_LONG = 6             # length > 140
N_EMPTY = 3            # empty sequence

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "toy" / "raw"


def rand_seq(rng: random.Random, lo: int, hi: int) -> str:
    return "".join(rng.choices(AMINO_ACIDS, k=rng.randint(lo, hi)))


def regions(seq: str) -> dict:
    """Cut a sequence into fake FWR/CDR pieces (fixed offsets, toy only)."""
    cuts = [0, 25, 33, 50, 58, 95, 105, len(seq)]
    names = ["fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa", "cdr3_aa", "fwr4_aa"]
    return {n: seq[cuts[i]:cuts[i + 1]] for i, n in enumerate(names)}


def main() -> None:
    rng = random.Random(SEED)
    seqs: list[str] = []
    seen: set[str] = set()
    while len(seqs) < N_VALID:
        s = rand_seq(rng, 110, 130)
        if s not in seen:
            seen.add(s)
            seqs.append(s)

    rows = list(seqs)
    rows += [rng.choice(seqs) for _ in range(N_DUP)]
    for _ in range(N_GAP):
        s = list(rng.choice(seqs))
        for _ in range(3):
            s.insert(rng.randrange(len(s)), "-")
        rows.append("".join(s))
    for _ in range(N_X):
        s = list(rng.choice(seqs)); s[rng.randrange(len(s))] = "X"
        rows.append("".join(s))
    rows += [rand_seq(rng, 30, 90) for _ in range(N_SHORT)]
    rows += [rand_seq(rng, 150, 200) for _ in range(N_LONG)]
    rows += [""] * N_EMPTY
    rng.shuffle(rows)

    records = []
    for i, s in enumerate(rows):
        clean = s.replace("-", "")
        rec = {"sequence_alignment_aa": s}
        rec.update(regions(clean) if clean else {k: "" for k in
                   ["fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa", "cdr3_aa", "fwr4_aa"]})
        rec.update({
            "v_call": rng.choice(["IGHV3-3*01", "IGHV3S53*01", "IGHV1S1*01"]),
            "d_call": rng.choice(["IGHD1*01", "IGHD2*01"]),
            "j_call": rng.choice(["IGHJ4*01", "IGHJ6*01"]),
            "productive": "T", "locus": "IGH", "junction_aa": clean[95:105],
            # junk columns: must disappear in the slim step
            "sequence": "ACGT" * 10, "rev_comp": "F", "v_score": round(rng.random() * 300, 1),
            "junk_a": i, "junk_b": "x", "junk_c": "",
        })
        records.append(rec)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "toy_oas_raw.csv"
    meta = {"Run": "TOY", "Species": "toy", "Isotype": "Missing_c_domain", "Total": len(records)}
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(meta) + "\n")
    pd.DataFrame(records).to_csv(out, mode="a", index=False)
    print(f"wrote {len(records)} rows -> {out}")


if __name__ == "__main__":
    main()