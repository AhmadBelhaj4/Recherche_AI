"""Data utilities: loading, cleaning and splitting VHH sequences.

Cleaning pipeline (applied in order):
1. Remove ``-`` gaps (OAS aligned sequences contain them).
2. Make everything uppercase.
3. Keep only sequences matching ``^[ACDEFGHIKLMNPQRSTVWY]+$``.
4. Count and report how many sequences were rejected and why.
5. Remove exact duplicates.

NOTE: Do NOT delete raw data files. Record download links + sha256 instead.
NOTE: Any clustering must use MMseqs2 (not cd-hit). Record the exact command
      and parameters in a comment or log file (see workflows documentation).

# TODO(owner: A) — implement load_sequences(), clean_sequences(), split_by_cluster().
"""

# TODO(owner: A)

from __future__ import annotations

from pathlib import Path


def load_sequences(path: Path) -> list[str]:
    """Load raw sequences from a plain-text file (one per line) or FASTA.

    Parameters
    ----------
    path:
        Path to a ``.txt`` (one sequence per line) or ``.fasta`` file.

    Returns
    -------
    list[str]
        Raw sequences (not yet cleaned).

    Raises
    ------
    NotImplementedError
        Until Team A implements this function.
    """
    raise NotImplementedError  # TODO(owner: A)


def clean_sequences(
    sequences: list[str],
) -> tuple[list[str], dict[str, int]]:
    """Clean a list of raw sequences following the pipeline in the module docstring.

    Parameters
    ----------
    sequences:
        Raw sequences as loaded by :func:`load_sequences`.

    Returns
    -------
    cleaned:
        List of cleaned sequences.
    report:
        Dictionary counting rejection reasons, e.g.
        ``{"gap_removed": 12, "invalid_chars": 5, "duplicates": 200}``.

    Raises
    ------
    NotImplementedError
        Until Team A implements this function.
    """
    raise NotImplementedError  # TODO(owner: A)


def split_by_cluster(
    sequences: list[str],
    cluster_tsv: Path,
    ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Split sequences into train/val/test using cluster membership.

    Whole clusters are assigned together to prevent data leakage.

    Parameters
    ----------
    sequences:
        Cleaned unique sequences.
    cluster_tsv:
        Output of ``mmseqs easy-cluster ... --format-output query,target``.
        Record the exact MMseqs2 command here once you run it.
    ratios:
        (train, val, test) fractions — must sum to 1.0.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    train, val, test:
        Three disjoint lists of sequences.

    Raises
    ------
    NotImplementedError
        Until Team A implements this function.
    """
    raise NotImplementedError  # TODO(owner: A)
