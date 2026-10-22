"""tests/test_evaluation.py - Unit tests for evaluation metrics module."""

import pytest
from evaluation import (
    compute_aa_composition,
    compute_gravy,
    compute_kmer_jaccard,
    compute_levenshtein_diversity,
    compute_net_charge,
    compute_novelty,
    compute_physicochemical_profile,
    compute_uniqueness,
    compute_validity,
    evaluate_sequences,
    get_kmers,
    has_canonical_cysteines,
    is_valid_sequence,
    jaccard_similarity,
    kl_divergence,
    levenshtein_distance,
)


def test_is_valid_sequence():
    # Valid 100-140 standard amino acids
    seq_valid = "A" * 120
    assert is_valid_sequence(seq_valid, min_len=100, max_len=140)

    # Invalid lengths
    assert not is_valid_sequence("A" * 99, min_len=100, max_len=140)
    assert not is_valid_sequence("A" * 141, min_len=100, max_len=140)

    # Non-standard characters (e.g., 'X', 'B', lowercase, digits)
    assert not is_valid_sequence("A" * 119 + "X", min_len=100, max_len=140)
    assert not is_valid_sequence("a" * 120, min_len=100, max_len=140)
    assert not is_valid_sequence(None)


def test_compute_validity():
    seqs = ["A" * 120, "C" * 110, "A" * 50]  # 2 valid, 1 too short
    res = compute_validity(seqs, min_len=100, max_len=140)
    assert res["n_total"] == 3
    assert res["n_valid"] == 2
    assert pytest.approx(res["validity_rate"]) == 2 / 3


def test_compute_uniqueness():
    seqs = ["ACD", "ACD", "EFG"]
    assert pytest.approx(compute_uniqueness(seqs)) == 2 / 3
    assert compute_uniqueness([]) == 0.0


def test_levenshtein_distance():
    # Identical
    assert levenshtein_distance("ACDEF", "ACDEF") == 0.0
    # Completely disjoint same length
    assert levenshtein_distance("AAAA", "CCCC") == 1.0
    # 1 insertion/deletion in length 4 -> 1/4 = 0.25
    assert levenshtein_distance("ACDE", "ACD") == 0.25


def test_compute_levenshtein_diversity():
    seqs = ["AAAA", "CCCC", "GGGG", "TTTT"]
    div = compute_levenshtein_diversity(seqs, n_pairs=10, seed=42)
    assert 0.0 < div <= 1.0


def test_kmer_and_jaccard():
    seq = "ACDEF"
    kmers = get_kmers(seq, k=3)
    assert kmers == {"ACD", "CDE", "DEF"}

    # Identical
    assert jaccard_similarity("ACDEF", "ACDEF", k=3) == 1.0
    # Completely different
    assert jaccard_similarity("AAAA", "CCCC", k=3) == 0.0

    seqs = ["ACDEFG", "ACDEFH", "GHIKLM"]
    jacc_mean = compute_kmer_jaccard(seqs, k=3, n_pairs=5, seed=42)
    assert 0.0 <= jacc_mean <= 1.0


def test_aa_composition_and_kl_divergence():
    seqs = ["ACDEFGHIKLMNPQRSTVWY"]  # exactly 1 of each 20 AA
    comp = compute_aa_composition(seqs)
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        assert pytest.approx(comp[aa]) == 1 / 20

    # KL divergence between identical distributions should be ~0
    assert pytest.approx(kl_divergence(comp, comp), abs=1e-5) == 0.0

    # Skewed distribution should have positive KL
    skewed = {aa: 0.01 for aa in "ACDEFGHIKLMNPQRSTVWY"}
    skewed["A"] = 0.81
    assert kl_divergence(skewed, comp) > 0.5


def test_compute_novelty():
    train_set = {"SEQ_A", "SEQ_B", "SEQ_C"}
    generated = ["SEQ_A", "SEQ_D", "SEQ_E", "SEQ_F"]
    # 3 novel out of 4 -> 75%
    assert pytest.approx(compute_novelty(generated, train_set)) == 0.75


def test_physicochemical_properties():
    # Hydrophobic sequence (e.g., Isoleucine/Valine)
    gravy_high = compute_gravy("IVIVIVIV")
    # Hydrophilic sequence (e.g., Arginine/Aspartate)
    gravy_low = compute_gravy("RDRDRDRD")
    assert gravy_high > gravy_low

    # Net charge: K(+1) + R(+1) - D(-1) - E(-1) = 0
    assert compute_net_charge("KRDE") == 0.0
    assert compute_net_charge("KKRR") == 4.0

    # Canonical cysteines
    assert has_canonical_cysteines("ACDEFGHIKLC") is True
    assert has_canonical_cysteines("ACDEFGHIKLA") is False


def test_evaluate_sequences_full():
    seqs = [
        "HVQLVESGGGSVQAGGSLRLSCAVSGYTYSSYCMGWFRQAPGKEREGVAAIDSDGSTSYADSVKGRFTISQDNAKNTLYLQMNSLKPEDTAMYYCAAERTPHCKAAFQGLGLILAFTIGGQGTQVTVS",
        "HVQLVESGGGSVQAGGSLRLSCVASGDTIGRYCMGWFRQIPDKEREGVAGIESDGSTSYADSVKGRFTISQDSAKNILYLQMHSLKPEDTAMYYCAARSVGWCPLFEHWLGKRAYTPGGYFANWGQGTQV"
    ]
    report = evaluate_sequences(seqs, train_set=set(), min_len=100, max_len=140)
    assert report["n_sequences"] == 2
    assert report["validity_rate"] == 1.0
    assert report["uniqueness_rate"] == 1.0
    assert report["novelty_rate"] == 1.0
    assert report["canonical_cysteine_rate"] == 1.0
    assert "mean_gravy" in report
    assert "mean_net_charge" in report
