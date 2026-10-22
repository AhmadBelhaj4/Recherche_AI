# Guide Sprint 2 + Sprint 3 — Équipe B (Modèles & Infra)

Projet 1.1 · Recherche_AI · Sprint 2 : 26 oct – 1 nov · Sprint 3 : 2 – 8 nov

> Tout le code de ce guide a été exécuté et testé avant d'être écrit ici (10 tests pytest verts,
> entraînement, reprise, évaluation test, génération et tableau comparatif exécutés de bout en bout sur des données factices).
> Il n'a **pas** été exécuté sur tes vraies données ni sur GPU : c'est le but des étapes Colab.

---

## 0. Vue d'ensemble

| | Sprint 2 — « la chaîne d'entraînement » | Sprint 3 — « les vrais modèles » |
|---|---|---|
| But | Un `train.py` fiable qui converge sur les données jouets, puis branché sur les vraies données | LSTM + Transformer entraînés sur VHH, comparés à la baseline, 1 000 séquences livrées à A |
| Sortie | Code + tests + 1 époque réelle sur Colab | Checkpoints, courbes, `test_ppl`, FASTA + JSON (contrat C4) |
| Chiffre à battre | — | **val perplexité 11.83** (bigram, SRR3544218) ; uniforme = 20 |

**Ordre de travail (celui de l'équipe)** : VS Code + données jouets → `pytest` → push → Colab + vraies données.
Pas de MLflow : chaque run écrit ses fichiers (`history.csv`, `config.json`, `metrics.json`) et `compare_runs.py` fabrique le tableau.

### Convention de loss (identique à la baseline, sinon la comparaison est fausse)

```
nats/token = somme(NLL) / nombre de tokens réels      perplexité = exp(nats/token)
```

- `EOS` est compté (le modèle apprend à terminer), `BOS` n'est jamais prédit, `PAD` est ignoré.
- On divise la **somme** par le **nombre total de tokens**, jamais une moyenne de moyennes de batchs.

### Problèmes repérés dans le code actuel (corrigés dans ce guide)

| # | Problème | Conséquence | Correction |
|---|---|---|---|
| 1 | `CausalTransformerLM` : embeddings `N(0,1)` + weight tying | Loss initiale ≈ 47 nats au lieu de ≈ 3.1 | init `std=0.02` (S2.1) |
| 2 | NB03/NB04 lisent `PATHS["raw"].parent / "train.csv"` | Sur Colab ce chemin n'existe pas : on entraînerait sur du toy | `load_split(version, "train")` lit `processed/<version>/` (S2.2) |
| 3 | NB05 : `max(temperature, 1e-5)`, PAD/BOS tirables, `logits` modifiés en place | Séquences raccourcies en silence, epsilon non justifié | `sampling.py` exact (S2.3) |
| 4 | Pas de scheduler (warmup + cosinus), pas de reprise | Colab coupe les sessions : on perd tout | `training.py` avec `last.pt` + `--resume` (S2.2) |
| 5 | NB05 recrée `CausalTransformerLM(...)` avec les hyperparamètres par défaut | Les poids ne rentrent pas si la config a changé | la config est sauvegardée **dans** `best.pt` (S2.2) |

Les notebooks NB03/NB04 restent comme documentation pédagogique ; **l'entraînement officiel passe par `scripts/train.py`**.

---

# SPRINT 2 — Chaîne d'entraînement complète

**Objectif** : `train.py` qui converge sur les jouets, trois tests de bon sens, génération exacte, puis intégration des vraies données (vendredi).

**Livrables** : `models.py` corrigé, `training.py`, `sampling.py`, `scripts/train.py`, 2 fichiers de tests, 1 époque réelle sur Colab.

## S2.0 — Branche et état de départ (10 min)

```powershell
git checkout main; git pull
git checkout -b b/sprint2-training
python -m pytest -q          # attendu : 15 passed
powershell -ExecutionPolicy Bypass -File scripts\verify_workflow.ps1   # attendu : ALL CHECKS PASSED
```

Dans `requirements.txt`, remplace `torch>=2.0` par `torch>=2.3` (on utilise `torch.amp.GradScaler`).

## S2.1 — Corriger l'initialisation dans `models.py` (15 min)

**Pourquoi** : avec le weight tying, `head.weight` **est** `token_embed.weight`. Une matrice `N(0,1)` donne des logits énormes dès le départ,
donc une loss initiale d'environ 47 nats alors qu'un modèle au hasard doit valoir ln(23) ≈ 3.14. L'entraînement finit par corriger, mais
les premières époques sont gâchées et la loss initiale ne peut plus servir de test de bon sens.

Dans `CausalTransformerLM.__init__`, juste après `self.head.weight = self.token_embed.weight`, ajoute :

```python
        # GPT-2 style init (std 0.02). Avec N(0,1) + weight tying, la loss initiale vaut ~47 nats au lieu de ~3.1.
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
```

Dans `LSTMLM.__init__`, juste après `self.head.weight = self.embed.weight`, ajoute :

```python
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.1)
```

## S2.2 — `training.py` à la racine, à côté de `models.py` (1 h)

Ce fichier est le cœur du sprint. Il contient :

- `PRESETS` : trois tailles de modèle (un preset = une config reproductible).
- `evaluate` : nats/token et perplexité **exacts**, même convention que le bigram.
- `lr_at` : échauffement linéaire (100 pas) puis décroissance en cosinus jusqu'à 10 % du `lr`.
- `train` : AdamW, clipping des gradients, précision mixte (AMP) sur GPU, early stopping, checkpoints `best.pt` / `last.pt`, reprise.
- `load_checkpoint` : recharge un modèle à partir de son seul `run_id` (la config est dans le checkpoint).

**Choix expliqués**

- *Warmup* : au début, les gradients d'un Transformer sont instables ; on monte le `lr` progressivement.
- *Cosinus* : un `lr` qui décroît en douceur donne un meilleur minimum final qu'un `lr` constant.
- *Clipping à 1.0* : empêche un seul batch de faire exploser les poids. Avec AMP, on appelle `scaler.unscale_` **avant** de clipper, sinon on clippe des gradients multipliés par le facteur d'échelle.
- *`last.pt` à chaque époque* : Colab gratuit coupe les sessions ; sans lui, on perd des heures.
- *`best.pt` choisi sur la validation* : le test n'est jamais regardé pendant le réglage.

<!-- FILE: training.py -->
```python
"""training.py - code d'entraînement partagé (scripts/train.py, notebooks, tests).

Convention de loss IDENTIQUE à la baseline bigram :
  nats/token = somme(NLL) / nombre de tokens réels ; EOS compté, BOS jamais prédit, PAD ignoré.
"""
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from models import CausalTransformerLM, LSTMLM
from utils import (PAD_ID, VOCAB_SIZE, compute_nats_and_perplexity, get_paths, shift)

# Tailles de modèle (un preset = une config reproductible)
PRESETS = {
    "lstm":     dict(model="lstm", embed_dim=128, hidden_dim=256, num_layers=1),
    "tf_small": dict(model="transformer", embed_dim=128, num_heads=4, num_layers=4,
                     dim_feedforward=512, dropout=0.1),
    "tf_base":  dict(model="transformer", embed_dim=256, num_heads=8, num_layers=6,
                     dim_feedforward=1024, dropout=0.1),
}
DEFAULT_LR = {"lstm": 1e-3, "transformer": 6e-4}
POS_MAX_LEN = 200            # positions apprises : doit couvrir BOS + 140 résidus + EOS = 142


def build_model(cfg: dict) -> nn.Module:
    if cfg["model"] == "lstm":
        return LSTMLM(VOCAB_SIZE, cfg["embed_dim"], cfg["hidden_dim"], cfg["num_layers"])
    return CausalTransformerLM(VOCAB_SIZE, cfg["embed_dim"], cfg["num_heads"], cfg["num_layers"],
                               cfg["dim_feedforward"], max_len=POS_MAX_LEN, dropout=cfg["dropout"])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def logits_of(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """LSTMLM renvoie (logits, hidden), CausalTransformerLM renvoie logits."""
    out = model(x)
    return out[0] if isinstance(out, tuple) else out


def load_split(version: str, split: str) -> list:
    """Lit data/processed/<version>/<split>.csv (produit par NB02)."""
    path = get_paths()["processed"] / version / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : exécutez NB01 puis NB02 pour cette version.")
    return pd.read_csv(path, dtype=str, keep_default_na=False)["sequence"].tolist()


_CRIT = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> tuple:
    """(nats/token, perplexité) exacts : somme des NLL / nombre de tokens réels (pas une moyenne de batchs)."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x, y = shift(batch)
        x, y = x.to(device), y.to(device)
        logits = logits_of(model, x)
        total += _CRIT(logits.reshape(-1, VOCAB_SIZE).float(), y.reshape(-1)).item()
        n += (y != PAD_ID).sum().item()
    return compute_nats_and_perplexity(total, n)


def lr_at(step: int, total_steps: int, base_lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    """Échauffement linéaire puis décroissance en cosinus jusqu'à min_ratio * base_lr."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def train(model, cfg, train_loader, val_loader, device, run_id, epochs, lr,
          weight_decay=0.01, warmup=100, grad_clip=1.0, patience=3, amp=False,
          resume=False, log=print) -> dict:
    """Boucle d'entraînement avec checkpoints (best.pt, last.pt) et reprise (--resume)."""
    paths = get_paths()
    ckpt_dir = paths["models"] / run_id
    res_dir = paths["results"] / "runs" / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    total_steps = epochs * len(train_loader)

    state = {"epoch": 0, "step": 0, "best_val_nats": float("inf"), "best_epoch": 0,
             "bad_epochs": 0, "history": []}
    last_path = ckpt_dir / "last.pt"
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        log(f"Reprise depuis l'epoch {state['epoch']} (meilleur val nats {state['best_val_nats']:.4f})")

    t_start = time.time()
    for epoch in range(state["epoch"] + 1, epochs + 1):
        model.train()
        t0 = time.time()
        tot = torch.zeros((), device=device)
        ntok = torch.zeros((), device=device)
        for batch in train_loader:
            x, y = shift(batch)
            x, y = x.to(device), y.to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(state["step"], total_steps, lr, warmup)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = logits_of(model, x)
            loss_sum = _CRIT(logits.float().reshape(-1, VOCAB_SIZE), y.reshape(-1))
            n = (y != PAD_ID).sum()
            loss = loss_sum / n.clamp(min=1)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                                   # clip sur les vrais gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            state["step"] += 1
            tot += loss_sum.detach()
            ntok += n

        train_nats, train_ppl = compute_nats_and_perplexity(tot.item(), int(ntok.item()))
        val_nats, val_ppl = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_nats": train_nats, "train_ppl": train_ppl,
               "val_nats": val_nats, "val_ppl": val_ppl,
               "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
        state["history"].append(row)

        improved = val_nats < state["best_val_nats"]
        if improved:
            state.update(best_val_nats=val_nats, best_epoch=epoch, bad_epochs=0)
            torch.save({"state_dict": model.state_dict(), "config": cfg,
                        "epoch": epoch, "val_nats": val_nats}, ckpt_dir / "best.pt")
        else:
            state["bad_epochs"] += 1
        state["epoch"] = epoch
        torch.save({"state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(), "state": state, "config": cfg}, last_path)
        pd.DataFrame(state["history"]).to_csv(res_dir / "history.csv", index=False)

        log(f"Epoch {epoch:2d}/{epochs} | train {train_nats:.4f} | val {val_nats:.4f} "
            f"(ppl {val_ppl:.2f}) {'*' if improved else ''} | {row['seconds']:.0f}s")
        if state["bad_epochs"] >= patience:
            log(f"Early stopping : pas d'amélioration depuis {patience} epochs.")
            break

    metrics = {"best_epoch": state["best_epoch"], "best_val_nats": state["best_val_nats"],
               "best_val_ppl": math.exp(state["best_val_nats"]),
               "epochs_run": state["epoch"], "n_params": count_params(model),
               "train_seconds": time.time() - t_start}
    (res_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_checkpoint(run_id: str, device, which: str = "best"):
    """Recharge un modèle depuis models/<run_id>/<which>.pt (la config est dans le checkpoint)."""
    path = get_paths()["models"] / run_id / f"{which}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location=device)
    model = build_model(ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
```

Tailles des presets (mesurées) : `lstm` 431 104 paramètres · `tf_small` 821 888 · `tf_base` 4 796 160.

## S2.3 — `sampling.py` à la racine : génération exacte (45 min)

**Ce que ça corrige par rapport à NB05**

- **Température** : aucune constante `1e-5`. `temperature <= 0` lève une erreur ; le mode greedy est un `argmax` explicite (`greedy=True`).
- **PAD et BOS ne sont jamais tirés** : leurs logits sont mis à `-inf`. Sans ça, un `PAD` tiré au milieu serait supprimé en silence par `decode` et donnerait des séquences plus courtes sans le dire.
- **Top-k exact** : exactement `k` tokens gardés (même en cas d'égalité).
- **Nucléus exact** : le plus petit ensemble de masse cumulée ≥ p. Un token est gardé si et seulement si la masse **avant** lui est `< p` (`cum - sp < p`).
- **Une graine par batch** (`seed + b`) : résultat reproductible, batchs indépendants.
- **Suivi des séquences non terminées** : `terminées` dit si un `EOS` est apparu avant `max_len`.

<!-- FILE: sampling.py -->
```python
"""sampling.py - génération autorégressive exacte (greedy, température, top-k, nucleus).

Corrige les points faibles de NB05 : pas d'epsilon sur la température, PAD/BOS jamais tirés,
nucléus exact, graine distincte par batch, suivi des séquences non terminées.
"""
import torch
import torch.nn.functional as F

from utils import BOS_ID, EOS_ID, PAD_ID, AminoAcidTokenizer


def filter_logits(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Top-k exact (k tokens gardés) puis nucléus exact, sur des logits (B, V) déjà divisés par T."""
    logits = logits.clone()
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        kept = torch.full_like(logits, float("-inf"))
        kept.scatter_(-1, idx, vals)
        logits = kept
    if top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sp, si = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sp, dim=-1)
        # plus petit ensemble de masse cumulée >= p : un token est gardé ssi la masse AVANT lui est < p
        remove_sorted = (cum - sp) >= top_p
        remove = torch.zeros_like(remove_sorted).scatter(-1, si, remove_sorted)
        logits[remove] = float("-inf")
    return logits


@torch.no_grad()
def generate(model, n, device, max_len=140, temperature=1.0, top_k=0, top_p=1.0,
             greedy=False, seed=42, batch_size=250):
    """Retourne (séquences, terminées) ; terminée = un EOS a été produit avant max_len résidus."""
    if temperature <= 0:
        raise ValueError("temperature doit être > 0 (utiliser greedy=True pour l'argmax)")
    pos = getattr(model, "pos_embed", None)
    if pos is not None and max_len + 1 > pos.num_embeddings:
        raise ValueError(f"max_len+1={max_len + 1} dépasse les positions du modèle ({pos.num_embeddings})")

    model.eval()
    tok = AminoAcidTokenizer()
    seqs, done = [], []
    for b, start in enumerate(range(0, n, batch_size)):
        bsz = min(batch_size, n - start)
        gen = torch.Generator(device=device).manual_seed(seed + b)      # une graine par batch
        tokens = torch.full((bsz, 1), BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        for _ in range(max_len):
            out = model(tokens)
            logits = (out[0] if isinstance(out, tuple) else out)[:, -1, :].float()
            logits[:, [PAD_ID, BOS_ID]] = float("-inf")                 # jamais PAD ni BOS
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = filter_logits(logits / temperature, top_k, top_p)
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=gen)
            nxt = torch.where(finished.unsqueeze(1), torch.full_like(nxt, PAD_ID), nxt)
            tokens = torch.cat([tokens, nxt], dim=1)
            finished |= nxt.squeeze(1) == EOS_ID
            if finished.all():
                break
        for row, fin in zip(tokens.cpu().tolist(), finished.cpu().tolist()):
            seqs.append(tok.decode(row))
            done.append(bool(fin))
    return seqs, done
```

## S2.4 — Les trois tests de bon sens + tests unitaires (1 h)

Le plan du prof impose trois vérifications **avant** tout long entraînement. Elles sont automatisées dans `tests/test_training.py` :

| Test | Ce qu'il prouve | Si ça échoue |
|---|---|---|
| (a) loss initiale ≈ ln 23 ≈ 3.14 | pas de bug d'init ni de décalage inputs/cibles | revoir S2.1 et le `shift` |
| (b) surapprendre 1 batch de 8 séquences | le modèle peut apprendre (backward, loss, optimiseur OK) | bug dans le modèle ou la loss |
| (c) motif fixe `EVQLVESGGG` + queue aléatoire : le greedy doit retrouver le motif | le modèle apprend une vraie structure | positions ou masque causal cassés |

Piège classique : oublier le décalage `inputs = x[:, :-1]`, `targets = x[:, 1:]`. Le modèle recopie alors son entrée, la loss tombe à 0 tout de suite
et ne prouve rien. Le test (a) l'attrape (la loss initiale serait absurde) et le test causal existant (`test_models.py`) aussi.

<!-- FILE: tests/test_training.py -->
```python
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
```

<!-- FILE: tests/test_sampling.py -->
```python
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
```

```powershell
python -m pytest -q          # attendu : 25 passed (15 + 10 nouveaux)
```

## S2.5 — `scripts/train.py` (45 min)

Un seul point d'entrée pour le local (toy) et Colab (réel). Ce qu'il fait :

1. lit `processed/<version>/{train,val}.csv` (jamais le test) ;
2. vérifie qu'aucune séquence ne dépasse les positions du modèle (**pas de troncature silencieuse**) ;
3. écrit `results/runs/<run_id>/config.json`, puis `history.csv` et `metrics.json` pendant/après l'entraînement ;
4. sauvegarde `models/<run_id>/best.pt` et `last.pt`.

<!-- FILE: scripts/train.py -->
```python
"""Entraîne un LM autorégressif (LSTM ou Transformer) sur processed/<version>/{train,val}.csv.

Local (toy) :  python scripts/train.py --preset tf_small --epochs 3
Colab (réel) : python scripts/train.py --preset tf_small --data-version SRR3544218 --run-name tf_small-v1
Reprise      : même commande + --resume (même --run-name)
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from training import (DEFAULT_LR, POS_MAX_LEN, PRESETS, build_model, count_params, load_split, train)
from utils import AminoAcidTokenizer, get_data_version, get_device, get_paths, make_loader, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="tf_small")
    ap.add_argument("--data-version", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=None, help="sous-échantillon du train (tests rapides)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    version = args.data_version or get_data_version()
    cfg = dict(PRESETS[args.preset])
    if args.dropout is not None and "dropout" in cfg:
        cfg["dropout"] = args.dropout
    lr = args.lr or DEFAULT_LR[cfg["model"]]
    run_id = args.run_name or f"{args.preset}-{datetime.now():%Y%m%d-%H%M}"

    train_seqs, val_seqs = load_split(version, "train"), load_split(version, "val")
    if args.max_train:
        train_seqs = train_seqs[: args.max_train]
    longest = max(len(s) for s in train_seqs + val_seqs)
    if cfg["model"] == "transformer":
        assert longest + 2 <= POS_MAX_LEN, f"séquence de {longest} aa : augmenter POS_MAX_LEN (pas de troncature silencieuse)"

    tok = AminoAcidTokenizer()
    train_loader = make_loader(train_seqs, tok, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_seqs, tok, batch_size=args.batch_size, shuffle=False)
    model = build_model(cfg)
    print(f"run={run_id} | device={device} | data={version} | train={len(train_seqs):,} val={len(val_seqs):,} "
          f"| params={count_params(model):,} | lr={lr}")

    res_dir = get_paths()["results"] / "runs" / run_id
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "config.json").write_text(json.dumps(
        {"run_id": run_id, "data_version": version, "seed": args.seed, "lr": lr,
         "batch_size": args.batch_size, "epochs": args.epochs, "warmup": args.warmup,
         "weight_decay": args.weight_decay, "patience": args.patience,
         "n_train": len(train_seqs), "n_val": len(val_seqs), "model_config": cfg,
         "n_params": count_params(model), "device": str(device)}, indent=2), encoding="utf-8")

    metrics = train(model, cfg, train_loader, val_loader, device, run_id, args.epochs, lr,
                    weight_decay=args.weight_decay, warmup=args.warmup, patience=args.patience,
                    amp=not args.no_amp, resume=args.resume)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```

## S2.6 — Lancer en local sur les données jouets (VS Code, 20 min)

```powershell
python -m pytest -q
python scripts/train.py --preset tf_small --epochs 5 --batch-size 32 --warmup 5 --run-name toy-test
python scripts/train.py --preset lstm --epochs 3 --batch-size 32 --run-name toy-lstm
powershell -ExecutionPolicy Bypass -File scripts\verify_workflow.ps1
```

**À lire dans la sortie**

- Sur les données jouets (séquences **aléatoires**), la perplexité doit rester autour de **20** : il n'y a rien à apprendre à part la composition. Descendre nettement sous 15 serait suspect (fuite ou bug).
- Elle ne doit pas non plus rester à 47 nats au départ : sinon l'étape S2.1 n'est pas appliquée.
- Le but ici est « le code tourne, la loss n'explose pas, les fichiers sont écrits », pas un bon score. Vérifie que `models/toy-test/best.pt` et `results/runs/toy-test/history.csv` existent.

## S2.7 — Push, puis intégration sur Colab avec les vraies données (vendredi, 1 h)

```powershell
git add models.py training.py sampling.py scripts/train.py tests/test_training.py tests/test_sampling.py requirements.txt
git commit -m "Sprint2 B: training loop, exact sampling, sanity tests"
git push -u origin b/sprint2-training
```

Cellule Colab, depuis un runtime neuf (Runtime → Disconnect and delete runtime) :

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ["AI_PROJECT_DIR"] = "/content/drive/MyDrive/AI_Project_Shared"

!rm -rf /content/Recherche_AI
!git clone -b b/sprint2-training https://github.com/AhmadBelhaj4/Recherche_AI.git /content/Recherche_AI
%cd /content/Recherche_AI
!pip install -q -r requirements.txt
!python -m pytest -q
!nvidia-smi
!python scripts/train.py --preset tf_small --data-version SRR3544218 --epochs 1 --max-train 30000 --run-name smoke-s2
```

**Critères de réussite**

- `pytest` : 25 passed. `nvidia-smi` montre un GPU (menu Exécution → Modifier le type d'exécution → GPU).
- La ligne `run=... data=SRR3544218 | train=30,000` confirme que tu lis bien le vrai jeu et non le toy.
- La val perplexité après 1 époque est **clairement sous 20** et se rapproche de la baseline 11.83. Un échec ici veut dire un problème de données ou de code, à régler avant le Sprint 3.
- `AI_Project_Shared/models/smoke-s2/best.pt` existe sur le Drive.

Sers-toi de la durée affichée de l'époque (`... | NNs`) pour estimer le Sprint 3 : une époque complète a environ 243 000 séquences, soit environ 1 900 pas à batch 128.
Compare le temps de ce test (30 000 séquences) avec la taille totale et multiplie.

## S2.8 — Démo du vendredi (5 min) et Definition of Done

À montrer :

1. `pytest` : 25 passed, dont les trois tests de bon sens.
2. Courbe train/val d'un run (le `history.csv` du run Colab).
3. Le smoke test réel : val perplexité vs baseline 11.83.
4. Trois séquences générées par `scripts/generate.py` (écrit au Sprint 3, S3.7) ou, d'ici là, par `sampling.generate` dans un notebook.

Definition of Done : code sur Git et relu par l'équipe A · graine fixée (42) · résultat sauvé dans `results/runs/` · une phrase d'interprétation (« qu'est-ce que ça nous apprend ? »).

---

# SPRINT 3 — Les vrais modèles

**Objectif** : entraîner sur VHH un LSTM et des Transformers meilleurs que la baseline, mesurer `test_ppl` **une seule fois**, générer 1 000 séquences par modèle et les livrer à A.

**Livrables** : 3 runs complets, courbes, tableau baseline contre modèles, FASTA + JSON (contrat C4), démo.

## S3.0 — Branche (5 min)

```powershell
git checkout main; git pull          # après merge du PR du Sprint 2
git checkout -b b/sprint3-real-training
```

## S3.1 — Plan des runs

Une variable à la fois, même graine (42), **même train/val que la baseline** (`SRR3544218`) : c'est ce qui rend la comparaison honnête.

| Run (`--run-name`) | Preset | Époques | Rôle |
|---|---|---|---|
| `lstm-v1` | `lstm` (431 k params) | 10 | baseline neuronale simple |
| `tf_small-v1` | `tf_small` (822 k) | 15 | Transformer de référence |
| `tf_base-v1` | `tf_base` (4,8 M) | 15 | plus de capacité |

Règle du plan : **ne lance pas un long entraînement tant que les tests de bon sens (S2.4) ne passent pas** et que le smoke test (S2.7) n'a pas confirmé la lecture des vraies données.

## S3.2 — Préparer Colab (45 min)

Une cellule de préparation à lancer en début de session, puis une cellule par entraînement :

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ["AI_PROJECT_DIR"] = "/content/drive/MyDrive/AI_Project_Shared"

!rm -rf /content/Recherche_AI
!git clone -b b/sprint3-real-training https://github.com/AhmadBelhaj4/Recherche_AI.git /content/Recherche_AI
%cd /content/Recherche_AI
!pip install -q -r requirements.txt
!python -m pytest -q
```

Pourquoi ça survit aux coupures : les checkpoints et résultats vont directement sur le Drive (`AI_PROJECT_DIR`), et `last.pt` est sauvegardé à chaque époque.
**Après une coupure Colab**, refais la cellule de préparation puis relance la **même** commande avec `--resume` et le **même** `--run-name`.
Passe toujours `--run-name` explicitement : sans lui le nom contient l'heure et la reprise ne retrouve pas `last.pt`.

La précision mixte (AMP, float16) est activée automatiquement sur GPU et accélère l'époque sans changer la loss de façon notable. Si tu vois `nan`, relance avec `--no-amp`.

## S3.3 — Entraîner (1 à 3 h de calcul au total, à mesurer)

```python
!python scripts/train.py --preset lstm     --data-version SRR3544218 --epochs 10 --run-name lstm-v1
!python scripts/train.py --preset tf_small --data-version SRR3544218 --epochs 15 --run-name tf_small-v1
!python scripts/train.py --preset tf_base  --data-version SRR3544218 --epochs 15 --run-name tf_base-v1
```

Reprise après coupure (exemple) :

```python
!python scripts/train.py --preset tf_small --data-version SRR3544218 --epochs 15 --run-name tf_small-v1 --resume
```

L'early stopping (patience 3) arrête le run quand la validation ne s'améliore plus et garde `best.pt`.

## S3.4 — Lire les courbes (45 min)

Ouvre `results/runs/<run>/history.csv` et trace `train_nats` et `val_nats` par époque. Écris 2 phrases par run : ce que tu vois, ce que tu changes.

| Symptôme | Diagnostic | Action |
|---|---|---|
| train baisse, val remonte | surapprentissage | `--dropout 0.2`, modèle plus petit, arrêt plus tôt |
| les deux stagnent haut | sous-apprentissage | modèle plus grand, plus d'époques, vérifier le `lr` |
| loss en dents de scie | `lr` trop élevé | `--lr` plus petit (ex. 3e-4) |
| val ≥ 11.83 (bigram) | bug ou mauvais entraînement | refaire les tests de bon sens ; ne pas agrandir le modèle |
| val très basse (ex. perplexité 2–3) | suspicion : mémorisation des quasi-doublons (split aléatoire) | pas forcément un bug ; à noter dans les limites (S3.8) |

## S3.5 — `test_ppl`, calculé **une seule fois** (30 min)

`eval_test.py` charge `best.pt`, calcule le test, calcule aussi le bigram sur le même test pour la comparaison, et **refuse de relancer** si le fichier existe déjà.

<!-- FILE: scripts/eval_test.py -->
```python
"""Calcule test_ppl d'un run, UNE SEULE FOIS (refuse de relancer sans --force), avec le bigram en référence.

python scripts/eval_test.py --run-id tf_small-v1
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ngram import BigramLM
from training import evaluate, load_checkpoint, load_split
from utils import AminoAcidTokenizer, get_device, get_paths, make_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    res_dir = get_paths()["results"] / "runs" / args.run_id
    out = res_dir / "test_metrics.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} existe déjà : le test ne se calcule qu'une fois (--force pour écraser).")

    device = get_device()
    model, ck = load_checkpoint(args.run_id, device)
    version = json.loads((res_dir / "config.json").read_text(encoding="utf-8"))["data_version"]
    tok = AminoAcidTokenizer()
    test = load_split(version, "test")
    nats, ppl = evaluate(model, make_loader(test, tok, batch_size=128), device)

    bigram = BigramLM(alpha=1.0).fit(load_split(version, "train"), tok)
    b_nats, b_ppl = bigram.nats_per_token(test, tok)

    rec = {"run_id": args.run_id, "data_version": version, "best_epoch": ck["epoch"], "n_test": len(test),
           "test_nats": nats, "test_ppl": ppl, "bigram_test_nats": b_nats, "bigram_test_ppl": b_ppl}
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"test : {args.run_id} ppl {ppl:.3f} ({nats:.4f} nats) | bigram ppl {b_ppl:.3f} | uniforme 20")


if __name__ == "__main__":
    main()
```

Lance-le **après** avoir choisi tes modèles sur la validation, une fois par modèle :

```python
!python scripts/eval_test.py --run-id lstm-v1
!python scripts/eval_test.py --run-id tf_small-v1
!python scripts/eval_test.py --run-id tf_base-v1
```

Ne règle jamais un hyperparamètre en regardant le test : sinon le test devient une deuxième validation et ne mesure plus rien.

## S3.6 — Tableau comparatif (20 min)

<!-- FILE: scripts/compare_runs.py -->
```python
"""Tableau comparatif de tous les runs (+ bigram) -> results/runs.csv et tableau markdown.

python scripts/compare_runs.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from utils import get_paths


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main():
    paths = get_paths()
    rows = []
    for d in sorted((paths["results"] / "runs").glob("*")):
        cfg, met, test = read(d / "config.json"), read(d / "metrics.json"), read(d / "test_metrics.json")
        if not met:
            continue
        mc = cfg.get("model_config", {})
        rows.append({"run": d.name, "model": mc.get("model"), "layers": mc.get("num_layers"),
                     "d_model": mc.get("embed_dim"), "dropout": mc.get("dropout"),
                     "params": met["n_params"], "lr": cfg.get("lr"), "epochs": met["epochs_run"],
                     "best_epoch": met["best_epoch"], "val_nats": round(met["best_val_nats"], 4),
                     "val_ppl": round(met["best_val_ppl"], 3), "test_ppl": round(test["test_ppl"], 3) if test else None,
                     "minutes": round(met["train_seconds"] / 60, 1), "data": Path(str(cfg.get("data_version"))).name})
    if not rows:
        raise SystemExit("aucun run terminé trouvé dans results/runs/")
    df = pd.DataFrame(rows).sort_values("val_ppl")
    df.to_csv(paths["results"] / "runs.csv", index=False)
    print(df.to_string(index=False))

    for f in sorted((paths["results"] / "baseline").glob("bigram_*.json")):
        b = read(f)
        print(f"\nbaseline bigram ({f.name}) : val_ppl {b['val_ppl']:.3f} | uniforme 20")


if __name__ == "__main__":
    main()
```

```python
!python scripts/compare_runs.py
```

Le tableau (`results/runs.csv`) est la « seule version de la vérité » partagée avec A.

## S3.7 — Générer et livrer à A (contrat C4) (1 h)

`generate.py` écrit un FASTA et un JSON de même nom. Le JSON contient tout ce qu'il faut pour reproduire le lot :
checkpoint, température, top-k, top-p, graine, `n`, `max_len`, `data_version`, `contract_version`, date.
Sans lui, A ne peut ni reproduire ni comparer deux lots.

Il affiche aussi un contrôle rapide côté B (l'évaluation complète reste le travail de A) :

- **validité** : alphabet + longueur 100–140 (pas seulement « ça décode ») ;
- **terminées par EOS** : part des séquences qui finissent d'elles-mêmes (les autres sont coupées à `max_len`) ;
- **uniques**, **longueur moyenne** (à comparer au train) ;
- **copies exactes du train** : signal de mémorisation.

<!-- FILE: scripts/generate.py -->
```python
"""Génère des séquences depuis un run, écrit FASTA + JSON (contrat C4) et affiche un contrôle rapide.

python scripts/generate.py --run-id tf_small-v1 --n 1000 --temperature 1.0 --seed 42
python scripts/generate.py --run-id tf_small-v1 --top-p 0.95 --temperature 1.0
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from sampling import generate
from training import load_checkpoint, load_split
from utils import MAX_LEN, MIN_LEN, get_device, get_paths, is_valid_sequence

CONTRACT_VERSION = "1.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--family", default="vhh")
    args = ap.parse_args()

    paths = get_paths()
    device = get_device()
    model, ck = load_checkpoint(args.run_id, device)
    version = json.loads((paths["results"] / "runs" / args.run_id / "config.json")
                         .read_text(encoding="utf-8"))["data_version"]

    seqs, done = generate(model, args.n, device, max_len=args.max_len, temperature=args.temperature,
                          top_k=args.top_k, top_p=args.top_p, greedy=args.greedy, seed=args.seed)

    gen_id = f"{args.run_id}-" + ("greedy" if args.greedy else f"T{args.temperature:g}")
    if args.top_k:
        gen_id += f"-k{args.top_k}"
    if args.top_p < 1.0:
        gen_id += f"-p{args.top_p:g}"
    out_dir = paths["results"] / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta = out_dir / f"{args.family}_{gen_id}.fasta"
    with open(fasta, "w", encoding="utf-8") as f:
        for i, s in enumerate(seqs):
            f.write(f">{args.family}|{gen_id}|{i}\n{s}\n")
    meta = {"checkpoint": f"models/{args.run_id}/best.pt (epoch {ck['epoch']})", "temperature": args.temperature,
            "top_k": args.top_k, "top_p": args.top_p, "greedy": args.greedy, "seed": args.seed, "n": args.n,
            "max_len": args.max_len, "data_version": version, "contract_version": CONTRACT_VERSION,
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    fasta.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # contrôle rapide côté B (l'évaluation complète est faite par l'équipe A)
    train = load_split(version, "train")
    train_len, train_set = np.mean([len(s) for s in train]), set(train)
    lens = [len(s) for s in seqs]
    print(f"écrit : {fasta}")
    print(f"validité (alphabet + longueur {MIN_LEN}-{MAX_LEN}) : {np.mean([is_valid_sequence(s) for s in seqs]):.1%}")
    print(f"terminées par EOS : {np.mean(done):.1%} | uniques : {len(set(seqs)) / len(seqs):.1%}")
    print(f"longueur moyenne : {np.mean(lens):.1f} (train : {train_len:.1f})")
    print(f"copies exactes du train : {np.mean([s in train_set for s in seqs]):.1%}  (signal de mémorisation)")


if __name__ == "__main__":
    main()
```

Livre 1 000 séquences par modèle, température 1.0, graine 42 (consigne du plan), puis une étude de température en option :

```python
!python scripts/generate.py --run-id lstm-v1     --n 1000 --temperature 1.0 --seed 42
!python scripts/generate.py --run-id tf_small-v1 --n 1000 --temperature 1.0 --seed 42
!python scripts/generate.py --run-id tf_base-v1  --n 1000 --temperature 1.0 --seed 42
# option : autres réglages pour le Sprint 4
!python scripts/generate.py --run-id tf_small-v1 --n 1000 --temperature 0.8 --top-k 10
!python scripts/generate.py --run-id tf_small-v1 --n 1000 --top-p 0.95 --temperature 1.0
```

Les fichiers sont dans `AI_Project_Shared/results/generated/` : `vhh_<run>-T1.fasta` et `vhh_<run>-T1.json`. Préviens A (message : « lot prêt »).

**Comment lire le contrôle rapide** (à comparer avec la baseline bigram : longueur moyenne générée 84 aa contre 124 en train, séquences coupées à 15–69 aa)

- Un bon Transformer doit rapprocher la **longueur moyenne** de celle du train et monter la **validité**, parce que ses positions apprises lui donnent la notion de longueur.
- Beaucoup de séquences « non terminées » veut dire que le modèle n'a pas appris à finir : à noter, pas à cacher.
- Des **copies exactes du train** élevées veut dire mémorisation, pas création : mauvais signe pour un générateur.

## S3.8 — Limites à écrire honnêtement (pour le rapport)

- **Split aléatoire** : les quasi-doublons (hypermutation somatique, un seul sujet chameau) sont répartis au hasard entre train, val et test. Un Transformer **peut** mémoriser ; sa val/test perplexité est donc **optimiste**.
  Le bigram (483 paramètres) ne peut pas mémoriser : la comparaison favorise légèrement le Transformer. L'amélioration prévue est un split par clusters (CD-HIT ou MMseqs2 à 90 %), côté équipe A.
- **Une seule graine** par run : un écart plus petit que la variabilité entre graines n'est pas une différence. Si tu as le temps, relance le meilleur avec les graines 43 et 44 (`--seed 43`, `--run-name tf_small-v1-s43`) et donne la moyenne ± écart-type.
- **Perplexité ≠ qualité des séquences** : un modèle peut avoir une bonne perplexité et générer des séquences peu diverses. Choisis le meilleur modèle sur la perplexité **et** sur les métriques de génération de A.
- **Aucune validation en laboratoire** : on produit des séquences plausibles, pas des nanocorps prouvés fonctionnels.

## S3.9 — SH2 (seulement si A le livre)

Le dépôt actuel ne contient que la chaîne VHH. Si A produit `processed/<version_sh2>/{train,val,test}.csv`, les commandes sont les mêmes avec moins de données, donc plus de dropout et moins de patience :

```python
!python scripts/train.py --preset tf_small --data-version <version_sh2> --dropout 0.2 --patience 2 --run-name sh2-tf_small-v1
!python scripts/generate.py --run-id sh2-tf_small-v1 --family sh2 --n 1000
```

Attention : `MIN_LEN` et `MAX_LEN` de `utils.py` (100–140) sont réglés pour les VHH. Le domaine SH2 fait environ 100 aa : la « validité » affichée par `generate.py` n'est pas fiable pour SH2 tant que ces bornes ne sont pas adaptées.

## S3.10 — Push, démo et Definition of Done

```powershell
git add scripts/eval_test.py scripts/generate.py scripts/compare_runs.py
git commit -m "Sprint3 B: test eval, generation (contract C4), run comparison"
git push -u origin b/sprint3-real-training
```

Ne commit jamais `models/`, `results/`, ni de `.pt` / `.csv` / `.fasta` (déjà dans `.gitignore`, et `verify_workflow.ps1` le contrôle).
Ouvre le PR et fais-le relire par un membre de l'équipe A.

**Démo (vendredi)**

1. Tableau baseline contre modèles : val et test perplexité (bigram 11.83 en val ; uniforme 20).
2. Courbes train/val du meilleur modèle et diagnostic en deux phrases.
3. 5 séquences générées décodées, plus le contrôle rapide (validité, longueur, EOS, copies).
4. Plan B : des résultats pré-calculés (captures), au cas où Colab ne répond pas.

**Definition of Done (par tâche)**

- [ ] Code sur Git, relu par l'équipe A
- [ ] Graine fixée, résultat reproductible (commande exacte dans le README)
- [ ] Résultats dans `results/runs/` (ou figure sauvée), `test_ppl` calculé une seule fois
- [ ] Une phrase d'interprétation : « qu'est-ce que ça nous apprend ? »

---

## Annexe — Dépannage

| Erreur | Cause | Solution |
|---|---|---|
| `FileNotFoundError: ...processed/<version>/train.csv` | NB01/NB02 pas exécutés pour cette version, ou mauvais `--data-version` | vérifier le nom du dossier dans `AI_Project_Shared/data/processed/` |
| Loss initiale ≈ 47 | S2.1 non appliqué | ajouter l'init `std=0.02` |
| `AssertionError: séquence de N aa` | séquence plus longue que `POS_MAX_LEN - 2` | augmenter `POS_MAX_LEN` dans `training.py` |
| `nan` en entraînement GPU | instabilité float16 | `--no-amp` ou `--lr` plus petit |
| `CUDA out of memory` | batch trop grand pour `tf_base` | `--batch-size 64` |
| Reprise : repart de zéro | `--run-name` différent ou `--resume` oublié | même nom exact + `--resume` |
| `eval_test.py` refuse de tourner | le test a déjà été calculé (volontaire) | ne pas relancer ; `--force` seulement si le modèle a changé |
| `AttributeError: torch.amp.GradScaler` | torch < 2.3 | `torch>=2.3` dans `requirements.txt` |
| Le run Colab lit du toy | `AI_PROJECT_DIR` non défini avant l'import | définir la variable avant de lancer le script |
