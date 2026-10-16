"""Tests for the Team A / Team B interface contract.

What is tested here (toy data only, no real data required):
    - Split files exist and are non-empty.
    - Every sequence matches the 20 amino-acid alphabet.
    - Train / val / test are pairwise disjoint.
    - sha256 of each split matches the manifest.

Tests requiring real data are marked with ``@pytest.mark.real`` and are
skipped by default (``make test`` runs ``pytest -m "not real"``).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.resolve()
SPLITS_DIR = ROOT / "data" / "toy" / "splits"
MANIFEST_PATH = SPLITS_DIR / "manifest.json"
SPLIT_FILES = ["train_vhh.txt", "val_vhh.txt", "test_vhh.txt"]
AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


# ── Helpers ───────────────────────────────────────────────────────────────────


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(name: str) -> list[str]:
    path = SPLITS_DIR / name
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def ensure_toy_data():
    """Generate toy data if it does not yet exist."""
    if not MANIFEST_PATH.exists():
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "make_toy_data.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"make_toy_data.py failed:\n{result.stdout}\n{result.stderr}"
        )


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("split_name", SPLIT_FILES)
def test_split_exists_and_nonempty(split_name):
    """Split file must exist and contain at least one sequence."""
    path = SPLITS_DIR / split_name
    assert path.exists(), f"{path} does not exist — run `make toy`"
    seqs = load_split(split_name)
    assert len(seqs) > 0, f"{split_name} is empty"


@pytest.mark.parametrize("split_name", SPLIT_FILES)
def test_sequences_valid_alphabet(split_name):
    """Every sequence must contain only the 20 standard amino acids."""
    seqs = load_split(split_name)
    invalid = [s for s in seqs if not AA_PATTERN.match(s)]
    assert not invalid, (
        f"{split_name}: {len(invalid)} sequence(s) contain invalid characters:\n"
        + "\n".join(invalid[:5])
    )


def test_splits_pairwise_disjoint():
    """Train, val and test must have no sequence in common."""
    train = set(load_split("train_vhh.txt"))
    val = set(load_split("val_vhh.txt"))
    test = set(load_split("test_vhh.txt"))

    assert train.isdisjoint(val), f"train ∩ val = {len(train & val)} sequences"
    assert train.isdisjoint(test), f"train ∩ test = {len(train & test)} sequences"
    assert val.isdisjoint(test), f"val ∩ test = {len(val & test)} sequences"


@pytest.mark.parametrize("split_name", SPLIT_FILES)
def test_sha256_matches_manifest(split_name):
    """sha256 of each split file must match the manifest."""
    assert MANIFEST_PATH.exists(), f"Manifest not found: {MANIFEST_PATH}"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest["sha256"].get(split_name)
    assert expected is not None, f"{split_name} not in manifest"
    actual = sha256_file(SPLITS_DIR / split_name)
    assert actual == expected, (
        f"sha256 mismatch for {split_name}\n"
        f"  expected: {expected}\n"
        f"  got:      {actual}"
    )


# ── Real-data tests (skipped by default) ─────────────────────────────────────


@pytest.mark.real
def test_real_data_placeholder():
    """Placeholder — add real-data contract tests here."""
    pytest.skip("Real-data tests require VHH_DATA to be set.")
