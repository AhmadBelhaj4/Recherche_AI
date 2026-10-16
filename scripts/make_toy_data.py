"""Generate ~200 fake VHH sequences for local development and CI.

Output layout
-------------
data/toy/
    vhh_unique.txt              one sequence per line
    splits/
        train_vhh.txt           90 % of sequences
        val_vhh.txt              5 % of sequences
        test_vhh.txt             5 % of sequences
        manifest.json           sha256 of each split file

Sequences are random strings over the 20 standard amino acids
(``ACDEFGHIKLMNPQRSTVWY``), length 110–130, generated with a fixed seed.
All splits are pairwise disjoint (no sequence appears in two splits).

Usage
-----
    python scripts/make_toy_data.py          # writes data/toy/
    make toy                                 # same, via Makefile
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
SEED = 42
N_SEQUENCES = 200
MIN_LEN = 110
MAX_LEN = 130
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SPLIT_RATIOS = (0.90, 0.05, 0.05)  # train / val / test

ROOT = Path(__file__).parent.parent.resolve()
TOY_DIR = ROOT / "data" / "toy"
SPLITS_DIR = TOY_DIR / "splits"


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def make_sequences(rng: random.Random) -> list[str]:
    """Generate N_SEQUENCES unique random VHH-like sequences."""
    seen: set[str] = set()
    seqs: list[str] = []
    attempts = 0
    while len(seqs) < N_SEQUENCES:
        attempts += 1
        if attempts > N_SEQUENCES * 100:
            print(
                f"WARNING: could only generate {len(seqs)} unique sequences.",
                file=sys.stderr,
            )
            break
        length = rng.randint(MIN_LEN, MAX_LEN)
        seq = "".join(rng.choices(AMINO_ACIDS, k=length))
        if seq not in seen:
            seen.add(seq)
            seqs.append(seq)
    return seqs


def split(
    seqs: list[str], ratios: tuple[float, float, float], rng: random.Random
) -> tuple[list[str], list[str], list[str]]:
    """Split a shuffled list into (train, val, test) with disjoint membership."""
    shuffled = seqs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rng = random.Random(SEED)

    print(f"Generating {N_SEQUENCES} toy sequences (seed={SEED}) ...")
    seqs = make_sequences(rng)

    TOY_DIR.mkdir(parents=True, exist_ok=True)
    unique_path = TOY_DIR / "vhh_unique.txt"
    write_lines(unique_path, seqs)
    print(f"  wrote {len(seqs)} sequences -> {unique_path}")

    train, val, test = split(seqs, SPLIT_RATIOS, rng)
    split_files = {
        "train_vhh.txt": train,
        "val_vhh.txt": val,
        "test_vhh.txt": test,
    }
    for fname, subset in split_files.items():
        p = SPLITS_DIR / fname
        write_lines(p, subset)
        print(f"  wrote {len(subset):3d} sequences -> {p}")

    # Verify disjointness (sanity check)
    assert set(train).isdisjoint(val), "train n val is not empty!"
    assert set(train).isdisjoint(test), "train n test is not empty!"
    assert set(val).isdisjoint(test), "val n test is not empty!"

    # Write manifest
    manifest: dict = {
        "data_version": "toy",
        "sha256": {fname: sha256_file(SPLITS_DIR / fname) for fname in split_files},
    }
    manifest_path = SPLITS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote manifest -> {manifest_path}")
    print("Done OK")


if __name__ == "__main__":
    main()


