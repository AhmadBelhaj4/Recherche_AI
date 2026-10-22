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
