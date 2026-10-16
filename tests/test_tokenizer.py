"""Tokenizer tests — skipped until Team B implements src/tokenizer.py."""

import pytest


@pytest.mark.skip(reason="stub — Team B has not yet implemented tokenizer.py")
def test_encode_decode_roundtrip():
    """encode then decode must return the original sequence."""
    from src.tokenizer import decode, encode

    seq = "ACDEFGHIKLMNPQRSTVWY"
    assert decode(encode(seq)) == seq


@pytest.mark.skip(reason="stub — Team B has not yet implemented tokenizer.py")
def test_vocab_size():
    """Vocabulary must have exactly 23 tokens (20 AA + PAD + BOS + EOS)."""
    from src.tokenizer import VOCAB_SIZE

    assert VOCAB_SIZE == 23
