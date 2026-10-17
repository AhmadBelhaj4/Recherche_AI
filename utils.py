"""
utils.py - Shared utilities for notebooks NB01 to NB06.
Contains tokenization, PyTorch Dataset, metrics, and seed helpers.
"""

import math
import random
import re
from collections import Counter
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_REGEX = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AminoAcidTokenizer:
    """Character-level tokenizer for the 20 standard amino acids with special tokens."""

    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"

    def __init__(self):
        self.special_tokens = [self.PAD, self.BOS, self.EOS, self.UNK]
        self.vocab = self.special_tokens + list(STANDARD_AA)
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.vocab)}
        self.id_to_token = {idx: tok for idx, tok in enumerate(self.vocab)}

        self.pad_id = self.token_to_id[self.PAD]
        self.bos_id = self.token_to_id[self.BOS]
        self.eos_id = self.token_to_id[self.EOS]
        self.unk_id = self.token_to_id[self.UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        """Convert amino-acid string to list of token IDs."""
        ids = [self.token_to_id.get(ch, self.unk_id) for ch in sequence.strip().upper()]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert list of token IDs back to amino-acid string."""
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, self.UNK)
            if skip_special_tokens and tok in self.special_tokens:
                continue
            tokens.append(tok)
        return "".join(tokens)


class SequenceDataset(Dataset):
    """PyTorch Dataset wrapping tokenized amino acid sequences."""

    def __init__(self, sequences: List[str], tokenizer: AminoAcidTokenizer):
        self.data = [tokenizer.encode(seq, add_special_tokens=True) for seq in sequences]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> List[int]:
        return self.data[idx]


def collate_fn(batch: List[List[int]], pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences to max length in the batch. Returns (inputs, targets)."""
    max_len = max(len(seq) for seq in batch)
    padded = [seq + [pad_id] * (max_len - len(seq)) for seq in batch]
    tensor = torch.tensor(padded, dtype=torch.long)
    inputs = tensor[:, :-1]
    targets = tensor[:, 1:]
    return inputs, targets


def is_valid_sequence(seq: str, min_len: int = 100, max_len: int = 140) -> bool:
    """Check if sequence has valid length and only standard 20 amino acids."""
    if not (min_len <= len(seq) <= max_len):
        return False
    return bool(AA_REGEX.match(seq))


def compute_nats_and_perplexity(total_nll: float, total_tokens: int) -> Tuple[float, float]:
    """Compute nats/token and perplexity from total negative log likelihood and token count."""
    if total_tokens <= 0:
        return float("nan"), float("nan")
    nats_per_token = total_nll / total_tokens
    perplexity = math.exp(min(nats_per_token, 100))
    return nats_per_token, perplexity


def compute_aa_composition(sequences: List[str]) -> Dict[str, float]:
    """Calculate frequencies of the 20 standard amino acids across sequences."""
    counts = Counter("".join(sequences))
    total = sum(counts[aa] for aa in STANDARD_AA) or 1
    return {aa: counts[aa] / total for aa in STANDARD_AA}
