"""Evaluation metrics for generated VHH sequences.

Metrics to implement:
- validity        : fraction matching ``^[ACDEFGHIKLMNPQRSTVWY]+$``
- diversity       : mean pairwise Levenshtein distance, mean Jaccard (k-mers)
- novelty         : fraction not found in the training set (cluster-based)
- kl_divergence   : KL(generated AA freq || training AA freq)
  NOTE: Zero-frequency entries must be masked before taking log — do NOT add
  an epsilon silently. Mask them explicitly and document it here.

# TODO(owner: A) — implement all metric functions.
"""

# TODO(owner: A)

from __future__ import annotations


def validity(sequences: list[str]) -> float:
    """Fraction of sequences matching the 20-AA alphabet.

    Parameters
    ----------
    sequences:
        List of generated amino-acid strings.

    Returns
    -------
    float
        Fraction in [0, 1].

    Raises
    ------
    NotImplementedError
        Until Team A implements this function.
    """
    raise NotImplementedError  # TODO(owner: A)


def diversity(sequences: list[str], k: int = 3) -> dict[str, float]:
    """Mean pairwise Levenshtein distance and mean Jaccard similarity on k-mers.

    Parameters
    ----------
    sequences:
        List of generated amino-acid strings.
    k:
        k-mer size for Jaccard similarity (default 3).

    Returns
    -------
    dict with keys ``"mean_levenshtein"`` and ``"mean_jaccard"``.

    Raises
    ------
    NotImplementedError
        Until Team A implements this function.
    """
    raise NotImplementedError  # TODO(owner: A)


def kl_divergence(generated: list[str], reference: list[str]) -> float:
    """KL divergence between per-position amino-acid frequency distributions.

    Zero-frequency positions are masked before the log is taken.
    This is intentional — see the docstring at the top of this module.

    Parameters
    ----------
    generated:
        List of generated amino-acid strings.
    reference:
        List of reference (training-set) amino-acid strings.

    Returns
    -------
    float
        KL(generated || reference).

    Raises
    ------
    NotImplementedError
        Until Team A implements this function.
    """
    raise NotImplementedError  # TODO(owner: A)
