import sys
from pathlib import Path

import torch

# Ensure repo root is on sys.path so `models` and `utils` are importable
ROOT = next(
    (p for p in [Path(__file__).resolve().parent.parent, *Path(__file__).resolve().parents]
     if (p / "utils.py").exists()),
    Path(__file__).resolve().parent.parent,
)
sys.path.insert(0, str(ROOT))

from models import LSTMLM, CausalTransformerLM
from utils import AminoAcidTokenizer, make_loader, shift


def test_causality():
    """Logits at earlier positions must be identical regardless of the last token."""
    m = CausalTransformerLM(23).eval()
    a = torch.tensor([[21, 5, 8, 12]])
    b = torch.tensor([[21, 5, 8, 3]])
    with torch.no_grad():
        assert torch.allclose(m(a)[:, :3], m(b)[:, :3], atol=1e-6)


def test_lstm_weight_tying():
    """head.weight and embed.weight must share the same storage."""
    m = LSTMLM(23)
    assert m.head.weight.data_ptr() == m.embed.weight.data_ptr()


def test_loader_shift():
    """make_loader + shift produce (B, L-1) long tensors with overlapping content."""
    tok = AminoAcidTokenizer()
    # "ACD" encodes to [BOS, 0, 1, 3, EOS] = len 5
    # "ACDEFG" encodes to [BOS, 0, 1, 3, 4, 5, 6, EOS] = len 8 → padded to 8
    # After collate, batch shape is (2, 8); shift gives (2, 7)
    batch = next(iter(make_loader(["ACD", "ACDEFG"], tok, batch_size=2)))
    x, y = shift(batch)
    assert x.shape == y.shape == (2, 7), f"Expected (2,7) got x={x.shape} y={y.shape}"
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    # The overlapping middle slice must be equal
    assert torch.equal(x[:, 1:], y[:, :-1])