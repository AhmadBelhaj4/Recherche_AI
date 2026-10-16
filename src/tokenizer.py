"""Tokenizer for VHH amino-acid sequences.

Vocabulary (size 23):
    20 standard amino acids: A C D E F G H I K L M N P Q R S T V W Y
    3 special tokens:        <PAD>=0  <BOS>=1  <EOS>=2

# TODO(owner: B) — implement encode(), decode(), and batch helpers.
"""

# TODO(owner: B)

from __future__ import annotations

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Special token indices
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2

VOCAB: dict[str, int] = {"<PAD>": PAD_IDX, "<BOS>": BOS_IDX, "<EOS>": EOS_IDX}
VOCAB.update({aa: i + 3 for i, aa in enumerate(AMINO_ACIDS)})

IDX_TO_TOKEN: dict[int, str] = {v: k for k, v in VOCAB.items()}

VOCAB_SIZE = len(VOCAB)  # 23


def encode(sequence: str) -> list[int]:
    """Encode a raw amino-acid string to token indices (without BOS/EOS).

    Parameters
    ----------
    sequence:
        String of uppercase amino-acid characters, e.g. ``"ACDE..."``.

    Returns
    -------
    list[int]
        List of token indices.

    Raises
    ------
    NotImplementedError
        Until Team B implements this function.
    """
    raise NotImplementedError  # TODO(owner: B)


def decode(indices: list[int], skip_special: bool = True) -> str:
    """Decode a list of token indices back to an amino-acid string.

    Parameters
    ----------
    indices:
        List of token indices as returned by :func:`encode`.
    skip_special:
        If ``True`` (default), strip ``<PAD>``, ``<BOS>``, ``<EOS>`` tokens.

    Returns
    -------
    str
        Amino-acid string.

    Raises
    ------
    NotImplementedError
        Until Team B implements this function.
    """
    raise NotImplementedError  # TODO(owner: B)
