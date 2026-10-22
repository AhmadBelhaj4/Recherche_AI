import math
import random

import pytest
import torch
import torch.nn as nn

from sampling import generate
from training import (PRESETS, build_model, evaluate, logits_of, lr_at)
from utils import (PAD_ID, STANDARD_AA, VOCAB_SIZE, AminoAcidTokenizer, make_loader, set_seed, shift)

TINY_TF = dict(model="transformer", embed_dim=64, num_heads=4, num_layers=2, dim_feedforward=128, dropout=0.0)


def rand_seqs(n, length, seed=0):
    rng = random.Random(seed)
    return ["".join(rng.choices(STANDARD_AA, k=length)) for _ in range(n)]


def first_batch(seqs, bs=8):
    return next(iter(make_loader(seqs, AminoAcidTokenizer(), batch_size=bs)))


@pytest.mark.parametrize("name", ["lstm", "tf_small"])
def test_initial_loss_is_close_to_ln_23(name):
    """(a) Un modèle non entraîné doit valoir ~ln(23)=3.135 nats/token (sinon bug d'init ou de cible)."""
    set_seed(0)
    model = build_model(PRESETS[name])
    loader = make_loader(rand_seqs(32, 100), AminoAcidTokenizer(), batch_size=16)
    nats, _ = evaluate(model, loader, torch.device("cpu"))
    assert abs(nats - math.log(23)) < 0.3, nats


def test_overfit_one_batch():
    """(b) Un seul batch de 8 séquences doit être mémorisé : sinon bug dans le modèle ou la loss."""
    set_seed(0)
    model = build_model(TINY_TF)
    x, y = shift(first_batch(rand_seqs(8, 20)))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    for _ in range(300):
        opt.zero_grad()
        loss = crit(logits_of(model, x).reshape(-1, VOCAB_SIZE), y.reshape(-1))
        loss.backward()
        opt.step()
    assert loss.item() < 0.3, loss.item()


def test_motif_is_learned_and_regenerated():
    """(c) Données avec motif fixe : après entraînement, la génération greedy doit retrouver le motif."""
    set_seed(0)
    motif = "EVQLVESGGG"
    tails = rand_seqs(200, 20, seed=1)
    loader = make_loader([motif + t for t in tails], AminoAcidTokenizer(), batch_size=50, shuffle=True)
    model = build_model(TINY_TF)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    crit = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    for _ in range(40):
        for batch in loader:
            x, y = shift(batch)
            opt.zero_grad()
            crit(logits_of(model, x).reshape(-1, VOCAB_SIZE), y.reshape(-1)).backward()
            opt.step()
    seqs, _ = generate(model, 10, torch.device("cpu"), max_len=30, greedy=True)
    assert all(s.startswith(motif) for s in seqs), seqs[:3]


def test_lr_schedule_shape():
    assert lr_at(0, 1000, 1.0, 100) == pytest.approx(0.01)
    assert lr_at(99, 1000, 1.0, 100) == pytest.approx(1.0)
    assert lr_at(999, 1000, 1.0, 100) == pytest.approx(0.1, abs=1e-3)
