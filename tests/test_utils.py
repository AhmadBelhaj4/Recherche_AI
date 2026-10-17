import math
import random

import pytest
import torch
from torch.utils.data import DataLoader

from utils import (
    STANDARD_AA, PAD_ID, BOS_ID, EOS_ID,
    AminoAcidTokenizer, SequenceDataset, collate_fn, compute_nats_and_perplexity,
)


def random_seqs(n, seed=0):
    rng = random.Random(seed)
    return ["".join(rng.choices(STANDARD_AA, k=rng.randint(100, 140))) for _ in range(n)]


def test_vocab_is_exactly_23():
    tok = AminoAcidTokenizer()
    assert tok.vocab_size == 23
    assert (tok.pad_id, tok.bos_id, tok.eos_id) == (20, 21, 22)
    assert [tok.aa_to_id[a] for a in STANDARD_AA] == list(range(20))


def test_roundtrip_1000_random_sequences():
    tok = AminoAcidTokenizer()
    for s in random_seqs(1000):
        ids = tok.encode(s)
        assert ids[0] == BOS_ID and ids[-1] == EOS_ID
        assert tok.decode(ids) == s


@pytest.mark.parametrize("bad", ["ACDX", "acd", "AC-D", ""])
def test_invalid_sequences_are_rejected(bad):
    with pytest.raises(ValueError):
        AminoAcidTokenizer().encode(bad)


def test_batch_shape_dtype_and_padding():
    tok = AminoAcidTokenizer()
    loader = DataLoader(SequenceDataset(["ACD", "ACDEFG"], tok), batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert batch.shape == (2, 8)          # longest = BOS + 6 AA + EOS
    assert batch.dtype == torch.long
    assert batch[0, 5:].tolist() == [PAD_ID] * 3


def test_uniform_model_has_perplexity_20():
    nats, ppl = compute_nats_and_perplexity(math.log(20) * 100, 100)
    assert nats == pytest.approx(math.log(20))
    assert ppl == pytest.approx(20)
