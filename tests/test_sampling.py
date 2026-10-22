import torch
import torch.nn as nn

from sampling import filter_logits, generate
from training import PRESETS, build_model
from utils import EOS_ID, PAD_ID, VOCAB_SIZE

CPU = torch.device("cpu")


def test_topk_keeps_exactly_k():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0, 4.0]])
    out = filter_logits(logits, top_k=2)
    assert torch.isfinite(out).sum().item() == 2
    assert torch.isfinite(out[0, 4]) and torch.isfinite(out[0, 1])


def test_nucleus_exact_definition():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    # plus petit ensemble de masse >= p
    assert torch.isfinite(filter_logits(logits, top_p=0.4)).sum().item() == 1     # 0.5 >= 0.4
    assert torch.isfinite(filter_logits(logits, top_p=0.85)).sum().item() == 3    # 0.5+0.3 < 0.85 <= 0.95
    assert torch.isfinite(filter_logits(logits, top_p=1.0)).sum().item() == 4


class _FavorsPadThenEos(nn.Module):
    """Faux modèle : PAD/BOS ont de très gros logits, EOS le meilleur token autorisé."""
    def forward(self, x):
        out = torch.zeros(x.size(0), x.size(1), VOCAB_SIZE)
        out[..., PAD_ID] = 100.0
        out[..., 21] = 100.0
        out[..., EOS_ID] = 50.0
        return out


def test_pad_and_bos_are_never_sampled():
    seqs, done = generate(_FavorsPadThenEos(), 20, CPU, max_len=10, temperature=1.0)
    assert all(done) and all(s == "" for s in seqs)           # EOS choisi dès le 1er pas


def test_generation_is_deterministic_and_bounded():
    torch.manual_seed(0)
    model = build_model(PRESETS["tf_small"])
    a, _ = generate(model, 30, CPU, max_len=40, seed=7, batch_size=16)
    b, _ = generate(model, 30, CPU, max_len=40, seed=7, batch_size=16)
    assert a == b and len(a) == 30
    assert all(len(s) <= 40 for s in a)


def test_untrained_model_rarely_terminates():
    torch.manual_seed(0)
    model = build_model(PRESETS["lstm"])
    seqs, done = generate(model, 10, CPU, max_len=15, seed=1)
    assert len(seqs) == len(done) == 10
