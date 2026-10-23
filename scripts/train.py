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
