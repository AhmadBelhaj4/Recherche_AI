import random

import pytest

from ngram import BigramLM
from utils import STANDARD_AA, AminoAcidTokenizer


def rand_seqs(n, seed):
    rng = random.Random(seed)
    return ["".join(rng.choices(STANDARD_AA, k=rng.randint(100, 140))) for _ in range(n)]


def test_rows_sum_to_one_and_no_mass_on_pad_bos():
    tok = AminoAcidTokenizer()
    lm = BigramLM().fit(rand_seqs(200, 0), tok)
    assert lm.probs.sum(axis=1) == pytest.approx(1.0)
    assert (lm.probs[:, [tok.pad_id, tok.bos_id]] == 0).all()


def test_random_data_perplexity_close_to_20():
    tok = AminoAcidTokenizer()
    lm = BigramLM().fit(rand_seqs(2000, 1), tok)
    _, ppl = lm.nats_per_token(rand_seqs(500, 2), tok)
    assert 19.5 < ppl < 22          # < 15 on pure random would mean a bug or a leak


def test_learns_a_pattern():
    tok = AminoAcidTokenizer()
    lm = BigramLM().fit(["ACACACACAC"] * 200, tok)
    _, ppl = lm.nats_per_token(["ACACACACAC"] * 20, tok)
    assert ppl < 2


def test_generate_is_decodable_and_deterministic():
    tok = AminoAcidTokenizer()
    lm = BigramLM().fit(rand_seqs(300, 3), tok)
    a = lm.generate(20, tok, max_len=140, seed=7)
    assert a == lm.generate(20, tok, max_len=140, seed=7)
    assert all(len(s) <= 140 for s in a)
    for s in a:
        if s:
            tok.encode(s)           # raises ValueError if invalid
