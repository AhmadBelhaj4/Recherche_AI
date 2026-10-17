"""utils.py - shared helpers for NB01..NB06.
Tokenizer follows the professor's spec exactly: V = 23
  ids 0-19 = the 20 amino acids (in the order of STANDARD_AA)
  20 = <PAD>, 21 = <BOS>, 22 = <EOS>
"""

import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_REGEX = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")

PAD_ID, BOS_ID, EOS_ID = 20, 21, 22
VOCAB_SIZE = 23

# Provisional bounds (see CONTRACT.md): to confirm from the real camelid VHH histogram
MIN_LEN, MAX_LEN = 100, 140


# ---------- reproducibility / environment ----------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_paths() -> Dict[str, Path]:
    """Folders for raw / processed / models / results. Works locally and on Colab."""
    if os.path.exists("/content"):  # Google Colab
        base = Path("/content/drive/MyDrive/AI_Project_Shared")
        if not base.exists():
            raise FileNotFoundError(
                "Drive folder not found. Run drive.mount('/content/drive') and make sure "
                "the AI_Project_Shared shortcut is in My Drive."
            )
        return {k: base / k for k in ("raw", "processed", "models", "results")}
    root = Path(__file__).resolve().parent
    return {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "models": root / "models",
        "results": root / "results",
    }


# ---------- tokenizer ----------

class AminoAcidTokenizer:
    """Exact tokenizer: 20 AA + PAD/BOS/EOS. No <UNK>: invalid input raises an error."""

    def __init__(self):
        self.aa_to_id = {aa: i for i, aa in enumerate(STANDARD_AA)}
        self.id_to_aa = {i: aa for aa, i in self.aa_to_id.items()}
        self.pad_id, self.bos_id, self.eos_id = PAD_ID, BOS_ID, EOS_ID
        self.vocab_size = VOCAB_SIZE

    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        if not AA_REGEX.match(sequence):
            raise ValueError(f"Invalid sequence (only the 20 uppercase amino acids allowed): {sequence[:30]!r}")
        ids = [self.aa_to_id[ch] for ch in sequence]
        if add_special_tokens:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids) -> str:
        """Stops at the first <EOS>; skips <PAD> and <BOS>."""
        out = []
        for i in ids:
            i = int(i)
            if i == EOS_ID:
                break
            if i in (PAD_ID, BOS_ID):
                continue
            out.append(self.id_to_aa[i])  # KeyError on an invalid id: on purpose
        return "".join(out)


# ---------- dataset ----------

class SequenceDataset(Dataset):
    def __init__(self, sequences: List[str], tokenizer: AminoAcidTokenizer):
        self.data = [tokenizer.encode(s, add_special_tokens=True) for s in sequences]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> List[int]:
        return self.data[idx]


def collate_fn(batch: List[List[int]]) -> torch.Tensor:
    """Right-pad with PAD_ID. Returns a (B, L) long tensor.
    The shift (inputs = x[:, :-1], targets = x[:, 1:]) is done in the training loop."""
    max_len = max(len(x) for x in batch)
    padded = [x + [PAD_ID] * (max_len - len(x)) for x in batch]
    return torch.tensor(padded, dtype=torch.long)


# ---------- data checks ----------

def is_valid_sequence(seq: str, min_len: int = MIN_LEN, max_len: int = MAX_LEN) -> bool:
    return min_len <= len(seq) <= max_len and bool(AA_REGEX.match(seq))


# ---------- metrics ----------

def compute_nats_and_perplexity(total_nll: float, total_tokens: int) -> Tuple[float, float]:
    """nats/token = sum NLL / number of real tokens. perplexity = exp(nats/token). No cap."""
    if total_tokens <= 0:
        raise ValueError("total_tokens must be > 0")
    nats = total_nll / total_tokens
    return nats, math.exp(nats)


def compute_aa_composition(sequences: List[str]) -> Dict[str, float]:
    counts = Counter("".join(sequences))
    total = sum(counts[aa] for aa in STANDARD_AA)
    if total == 0:
        raise ValueError("No amino acids found: cannot compute composition")
    return {aa: counts[aa] / total for aa in STANDARD_AA}
