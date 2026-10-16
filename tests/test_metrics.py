"""Metrics tests — skipped until Team A implements src/metrics.py."""

import pytest


@pytest.mark.skip(reason="stub — Team A has not yet implemented metrics.py")
def test_validity_all_valid():
    """validity() must return 1.0 for a list of clean sequences."""
    from src.metrics import validity

    seqs = ["ACDEFGHIKLMNPQRSTVWY", "AAAA", "MMMM"]
    assert validity(seqs) == 1.0


@pytest.mark.skip(reason="stub — Team A has not yet implemented metrics.py")
def test_validity_with_invalid():
    """validity() must return < 1.0 when some sequences contain bad characters."""
    from src.metrics import validity

    seqs = ["ACDE", "ACDE-X"]  # second contains invalid chars
    assert validity(seqs) < 1.0
