"""Fit the bigram baseline on processed/<version>/{train,val}.csv.

Local (toy):  python scripts/run_baseline.py
Colab (real): python scripts/run_baseline.py --data-version v1
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from ngram import BigramLM
from utils import AminoAcidTokenizer, get_data_version, get_paths, set_seed


def read_seqs(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)["sequence"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-version", default=None)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=160)    # provisional, as in the prof's NB02
    args = ap.parse_args()

    set_seed(args.seed)
    paths = get_paths()
    version = args.data_version or get_data_version()
    data_dir = paths["processed"] / version

    tok = AminoAcidTokenizer()
    train, val = read_seqs(data_dir / "train.csv"), read_seqs(data_dir / "val.csv")
    print(f"data_version={version} | train={len(train):,} | val={len(val):,}")

    lm = BigramLM(alpha=args.alpha).fit(train, tok)
    nats, ppl = lm.nats_per_token(val, tok)
    print(f"val: {nats:.4f} nats/token | perplexity {ppl:.3f}   (uniform over 20 AA = 20)")
    print("5 generated sequences:")
    for s in lm.generate(5, tok, max_len=args.max_len, seed=args.seed):
        print("  ", s[:60] or "<empty>")

    out_dir = paths["results"] / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"bigram_{version}_a{args.alpha:g}.json"
    record = {"model": "bigram", "data_version": version, "seed": args.seed,
              "alpha": args.alpha, "n_params": lm.n_params, "max_len": args.max_len,
              "n_train": len(train), "n_val": len(val),
              "val_nats": nats, "val_ppl": ppl,
              "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
