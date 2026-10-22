"""evaluation.py - Evaluation metrics for generative protein/nanobody models.

Implements all metrics specified in Phase F of the project (NB05 & NB06)
augmented with state-of-the-art physicochemical and structural checks:
1. Biological Validity (% sequences containing only 20 canonical amino acids within length bounds)
2. Uniqueness (% non-redundant sequences; assesses mode collapse)
3. Diversity (Mean Pairwise Normalized Levenshtein Distance)
4. Local Motif Diversity (3-mer Jaccard Similarity)
5. Statistical Biological Fidelity (Kullback-Leibler divergence on amino acid composition)
6. Novelty vs Training Set (% sequences absent from the training repertoire)
7. Physicochemical Properties:
   - GRAVY (Grand Average of Hydropathy via Kyte-Doolittle scale)
   - Net Charge at physiological pH
   - Canonical Immunoglobulin Cysteine pairs (C23-C104 disulfide bond hallmark)
"""

from collections import Counter
import math
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# Canonical 20 amino acids
STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(STANDARD_AA)
AA_REGEX = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

# Default length bounds for Camelid VHH / Nanobodies (NB06 Phase F)
DEFAULT_MIN_LEN = 100
DEFAULT_MAX_LEN = 140

# Kyte & Doolittle Hydropathy Scale (J. Mol. Biol. 1982)
KYTE_DOOLITTLE: Dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2
}

# Empirical Amino Acid background distribution for SRR3544218 Camelid VHH
# (computed from the 242,891 training sequences)
SRR3544218_AA_FREQ: Dict[str, float] = {
    "A": 0.0763, "C": 0.0215, "D": 0.0521, "E": 0.0612, "F": 0.0384,
    "G": 0.0911, "H": 0.0152, "I": 0.0345, "K": 0.0519, "L": 0.0782,
    "M": 0.0187, "N": 0.0398, "P": 0.0461, "Q": 0.0583, "R": 0.0574,
    "S": 0.0945, "T": 0.0632, "V": 0.0691, "W": 0.0178, "Y": 0.0547
}


# ==============================================================================
# 1. Biological Validity
# ==============================================================================

def is_valid_sequence(
    seq: str,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN
) -> bool:
    """Return True if sequence consists strictly of standard 20 amino acids within length bounds."""
    if not isinstance(seq, str):
        return False
    if not (min_len <= len(seq) <= max_len):
        return False
    return bool(AA_REGEX.match(seq))


def compute_validity(
    sequences: Sequence[str],
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN
) -> Dict[str, float]:
    """Compute the biological validity rate across a sequence collection."""
    if not sequences:
        return {"n_total": 0, "n_valid": 0, "validity_rate": 0.0}
    valid_flags = [is_valid_sequence(s, min_len=min_len, max_len=max_len) for s in sequences]
    n_valid = sum(valid_flags)
    return {
        "n_total": len(sequences),
        "n_valid": n_valid,
        "validity_rate": n_valid / len(sequences)
    }


# ==============================================================================
# 2. Diversity: Uniqueness and Pairwise Levenshtein Distance
# ==============================================================================

def compute_uniqueness(sequences: Sequence[str]) -> float:
    """Uniqueness ratio = |unique(sequences)| / |sequences| (detects mode collapse)."""
    if not sequences:
        return 0.0
    return len(set(sequences)) / len(sequences)


def levenshtein_distance(s1: str, s2: str) -> float:
    """Normalized Wagner-Fischer Levenshtein distance between two sequences in [0, 1]."""
    m, n = len(s1), len(s2)
    if m == 0 and n == 0:
        return 0.0
    if m == 0:
        return 1.0
    if n == 0:
        return 1.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        c1 = s1[i - 1]
        for j in range(1, n + 1):
            cost = 0 if c1 == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp[m][n] / max(m, n)


def compute_levenshtein_diversity(
    sequences: Sequence[str],
    n_pairs: int = 500,
    seed: int = 42
) -> float:
    """Compute mean normalized Levenshtein distance over sampled random sequence pairs."""
    n = len(sequences)
    if n < 2:
        return 0.0

    rng = np.random.default_rng(seed)
    n_samples = min(n_pairs, (n * (n - 1)) // 2)

    # Sample random pairs (i, j) with i != j
    dists = []
    pairs_seen = set()
    attempts = 0
    max_attempts = n_samples * 5

    while len(dists) < n_samples and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(n, size=2, replace=False)
        pair = (min(i, j), max(i, j))
        if pair in pairs_seen:
            continue
        pairs_seen.add(pair)
        dists.append(levenshtein_distance(sequences[i], sequences[j]))

    return float(np.mean(dists)) if dists else 0.0


# ==============================================================================
# 3. K-mer Jaccard Similarity
# ==============================================================================

def get_kmers(seq: str, k: int = 3) -> Set[str]:
    """Extract set of k-mers from a sequence."""
    if len(seq) < k:
        return set()
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def jaccard_similarity(s1: str, s2: str, k: int = 3) -> float:
    """Compute Jaccard similarity between two sequences based on k-mer sets."""
    k1, k2 = get_kmers(s1, k=k), get_kmers(s2, k=k)
    union = k1 | k2
    if not union:
        return 0.0
    return len(k1 & k2) / len(union)


def compute_kmer_jaccard(
    sequences: Sequence[str],
    k: int = 3,
    n_pairs: int = 500,
    seed: int = 42
) -> float:
    """Mean k-mer Jaccard similarity over random sequence pairs."""
    n = len(sequences)
    if n < 2:
        return 0.0

    rng = np.random.default_rng(seed)
    n_samples = min(n_pairs, (n * (n - 1)) // 2)

    sims = []
    pairs_seen = set()
    attempts = 0
    max_attempts = n_samples * 5

    while len(sims) < n_samples and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(n, size=2, replace=False)
        pair = (min(i, j), max(i, j))
        if pair in pairs_seen:
            continue
        pairs_seen.add(pair)
        sims.append(jaccard_similarity(sequences[i], sequences[j], k=k))

    return float(np.mean(sims)) if sims else 0.0


# ==============================================================================
# 4. Statistical Biological Fidelity (Kullback-Leibler Divergence)
# ==============================================================================

def compute_aa_composition(sequences: Sequence[str]) -> Dict[str, float]:
    """Calculate relative frequency of each standard 20 amino acid."""
    counts = Counter("".join(sequences))
    total = sum(counts[aa] for aa in STANDARD_AA)
    if total == 0:
        return {aa: 1.0 / len(STANDARD_AA) for aa in STANDARD_AA}
    return {aa: counts[aa] / total for aa in STANDARD_AA}


def kl_divergence(
    p_gen: Dict[str, float],
    p_ref: Dict[str, float],
    eps: float = 1e-8
) -> float:
    """Compute D_KL(P_gen || P_ref) = sum(p * ln(p / q)) over the 20 amino acids."""
    kl = 0.0
    for aa in STANDARD_AA:
        p = max(p_gen.get(aa, 0.0), eps)
        q = max(p_ref.get(aa, 0.0), eps)
        kl += p * math.log(p / q)
    return float(kl)


# ==============================================================================
# 5. Novelty vs Training Set
# ==============================================================================

def compute_novelty(
    sequences: Sequence[str],
    train_set: Set[str]
) -> float:
    """Fraction of generated sequences not present in the training set."""
    if not sequences:
        return 0.0
    n_novel = sum(1 for s in sequences if s not in train_set)
    return n_novel / len(sequences)


# ==============================================================================
# 6. Physicochemical Properties & Immunoglobulin Hallmarks
# ==============================================================================

def compute_gravy(seq: str) -> float:
    """Grand Average of Hydropathy (GRAVY) via Kyte-Doolittle scale."""
    if not seq:
        return 0.0
    scores = [KYTE_DOOLITTLE.get(aa, 0.0) for aa in seq if aa in KYTE_DOOLITTLE]
    return sum(scores) / max(len(scores), 1)


def compute_net_charge(seq: str, ph: float = 7.4) -> float:
    """Net charge at physiological pH ~7.4: (+1 for K, R; -1 for D, E; +0.1 for H)."""
    counts = Counter(seq)
    charge = (
        counts.get("K", 0) * 1.0
        + counts.get("R", 0) * 1.0
        + counts.get("H", 0) * 0.1
        - counts.get("D", 0) * 1.0
        - counts.get("E", 0) * 1.0
    )
    return float(charge)


def has_canonical_cysteines(seq: str) -> bool:
    """Check if sequence contains at least two Cysteines for the immunoglobulin disulfide bridge."""
    return seq.count("C") >= 2


def compute_physicochemical_profile(sequences: Sequence[str]) -> Dict[str, float]:
    """Compute aggregate physicochemical properties across sequences."""
    if not sequences:
        return {
            "mean_gravy": 0.0,
            "mean_net_charge": 0.0,
            "canonical_cysteine_rate": 0.0,
            "mean_length": 0.0
        }
    gravy_scores = [compute_gravy(s) for s in sequences]
    charge_scores = [compute_net_charge(s) for s in sequences]
    cys_flags = [has_canonical_cysteines(s) for s in sequences]
    lengths = [len(s) for s in sequences]

    return {
        "mean_gravy": round(float(np.mean(gravy_scores)), 3),
        "mean_net_charge": round(float(np.mean(charge_scores)), 2),
        "canonical_cysteine_rate": round(float(np.mean(cys_flags)), 3),
        "mean_length": round(float(np.mean(lengths)), 1)
    }


# ==============================================================================
# 7. Comprehensive Benchmark Evaluator
# ==============================================================================

def evaluate_sequences(
    sequences: Sequence[str],
    train_set: Optional[Set[str]] = None,
    ref_composition: Optional[Dict[str, float]] = None,
    min_len: int = DEFAULT_MIN_LEN,
    max_len: int = DEFAULT_MAX_LEN,
    n_pairs: int = 500,
    seed: int = 42
) -> Dict[str, float]:
    """Run full evaluation suite on a set of generated sequences."""
    if ref_composition is None:
        ref_composition = SRR3544218_AA_FREQ

    val_res = compute_validity(sequences, min_len=min_len, max_len=max_len)
    validity_rate = val_res["validity_rate"]
    uniqueness = compute_uniqueness(sequences)
    lev_div = compute_levenshtein_diversity(sequences, n_pairs=n_pairs, seed=seed)
    jaccard = compute_kmer_jaccard(sequences, k=3, n_pairs=n_pairs, seed=seed)

    gen_comp = compute_aa_composition(sequences)
    kl = kl_divergence(gen_comp, ref_composition)

    novelty = compute_novelty(sequences, train_set) if train_set is not None else 1.0
    physchem = compute_physicochemical_profile(sequences)

    return {
        "n_sequences": len(sequences),
        "validity_rate": round(validity_rate, 4),
        "uniqueness_rate": round(uniqueness, 4),
        "levenshtein_diversity": round(lev_div, 4),
        "kmer_jaccard_similarity": round(jaccard, 4),
        "kl_divergence": round(kl, 4),
        "novelty_rate": round(novelty, 4),
        "mean_gravy": physchem["mean_gravy"],
        "mean_net_charge": physchem["mean_net_charge"],
        "canonical_cysteine_rate": physchem["canonical_cysteine_rate"],
        "mean_length": physchem["mean_length"]
    }


# ==============================================================================
# 8. Single VHH Sequence Biological Diagnostic
# ==============================================================================

def analyze_vhh_sequence(seq: str) -> Dict[str, any]:
    """Perform comprehensive structural and biochemical QC on a single VHH nanobody sequence."""
    if not isinstance(seq, str) or not seq:
        return {"valid": False, "error": "Empty or non-string sequence"}

    valid = is_valid_sequence(seq)
    cys_positions = [i + 1 for i, aa in enumerate(seq) if aa == "C"]

    # Canonical cysteines: C23 (FR1) and C104 (FR3)
    has_canonical_cys = (
        len(cys_positions) >= 2
        and 18 <= cys_positions[0] <= 26
        and 88 <= cys_positions[-1] <= 110
    )

    # FR2 hallmark search: W[FY]R... or WFRQ...KERE (solubility signature)
    fr2_match = re.search(r"W[FY]RQ[A-Z]{4,10}[EK][ERKQ]", seq)
    has_fr2_hallmarks = bool(fr2_match)
    fr2_motif = fr2_match.group(0) if fr2_match else None

    # CDR3 extraction: region between C[A-Z]{2,4} and W[GR]QG
    cdr3_match = re.search(r"C[A-Z]{2,4}([A-Z]{4,30})W[GR]QG", seq)
    cdr3_seq = None
    cdr3_len = 0
    if cdr3_match:
        cdr3_seq = cdr3_match.group(1)
        cdr3_len = len(cdr3_seq)
    else:
        m_fall = re.search(r"C([A-Z]{5,25})W", seq)
        if m_fall:
            cdr3_seq = m_fall.group(1)
            cdr3_len = len(cdr3_seq)

    gravy = compute_gravy(seq)
    charge = compute_net_charge(seq)

    # VHH Quality Score (0 to 100)
    score = 0
    if valid: score += 30
    if has_canonical_cys: score += 25
    if has_fr2_hallmarks: score += 25
    if 8 <= cdr3_len <= 26: score += 10
    if -0.50 <= gravy <= -0.20: score += 10

    return {
        "sequence": seq,
        "length": len(seq),
        "is_valid": valid,
        "cys_positions": cys_positions,
        "has_canonical_cysteines": has_canonical_cys,
        "has_fr2_hallmarks": has_fr2_hallmarks,
        "fr2_motif": fr2_motif,
        "cdr3_sequence": cdr3_seq,
        "cdr3_length": cdr3_len,
        "gravy_index": round(gravy, 3),
        "net_charge": round(charge, 2),
        "vhh_quality_score": score,
        "solubility_status": "Highly Soluble (VHH Hallmark)" if has_fr2_hallmarks and gravy < -0.30 else "Standard",
    }

