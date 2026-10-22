Understood: no MLflow. The prof's NB02 also settles the token-id question: it uses AA = ids 0–19 and PAD/BOS/EOS = 20/21/22, exactly like your `utils.py` and `CONTRACT.md`. So the plan PDF's 0/1/2 layout is superseded and you change nothing in the tokenizer.

Keep your repo's NB02 instead of pasting the prof's. His version is a minimal reference: it uses an undefined `seqs` and truncates inside `encode`. Yours already meets his validation criteria (round-trip, padding only on the right, disjoint splits after dedup).

## Your Sprint 1 scope (Team B)

| # | Task | Status |
|---|---|---|
| 1 | Tokenizer V=23 + tests | Done (`utils.py`, `tests/test_utils.py`) |
| 2 | Toy data | Done (`make_toy_data.py`, then NB01, then NB02) |
| 3 | n-gram baseline + perplexity | **To do** |
| 4 | Tests for the baseline | **To do** |
| 5 | Check A's 100 real sequences | **To do** (Thu/Fri) |
| 6 | Run on Colab with real data | **To do** |
| 7 | Demo | **To do** |

## Step 0: branch (2 min)

```powershell
git checkout main; git pull
git checkout -b b/sprint1-baseline
```

## Step 1: confirm the starting point (10 min)

```powershell
python scripts/make_toy_data.py
python -m pytest -q
powershell -ExecutionPolicy Bypass -File scripts\verify_workflow.ps1
```
You need `11 passed` and `ALL CHECKS PASSED`. The baseline reads `data/toy/processed/toy/train.csv` and `val.csv`. If they're missing, run NB01 then NB02 once in toy mode (Restart & Run All).

## Step 2: `ngram.py` at the repo root, next to `models.py` (~1h)

The smoothing covers only the 21 predictable tokens (20 AA + EOS). Spreading it over all 23 would put probability on PAD/BOS and inflate the perplexity.

```python
"""ngram.py - order-2 (bigram) baseline: P(next | previous), Laplace smoothing."""
import math
import numpy as np

from utils import (STANDARD_AA, BOS_ID, EOS_ID, VOCAB_SIZE,
                   AminoAcidTokenizer, compute_nats_and_perplexity)


class BigramLM:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        tok = AminoAcidTokenizer()
        # tokens the model may predict: 20 AA + EOS (never PAD / BOS)
        self.next_ids = [tok.aa_to_id[a] for a in STANDARD_AA] + [EOS_ID]
        self.counts = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float64)
        self.probs = None

    @property
    def n_params(self) -> int:
        return VOCAB_SIZE * len(self.next_ids)

    def fit(self, sequences, tokenizer):
        for s in sequences:
            ids = tokenizer.encode(s)                      # [BOS] x1..xL [EOS]
            for prev, nxt in zip(ids[:-1], ids[1:]):
                self.counts[prev, nxt] += 1
        mask = np.zeros(VOCAB_SIZE)
        mask[self.next_ids] = 1.0
        smoothed = (self.counts + self.alpha) * mask       # P = 0 on PAD / BOS
        self.probs = smoothed / smoothed.sum(axis=1, keepdims=True)
        return self

    def nats_per_token(self, sequences, tokenizer):
        """Same convention as NB03/NB04: all target tokens incl. EOS, no BOS."""
        total, n = 0.0, 0
        for s in sequences:
            ids = tokenizer.encode(s)
            for prev, nxt in zip(ids[:-1], ids[1:]):
                total -= math.log(self.probs[prev, nxt])
                n += 1
        return compute_nats_and_perplexity(total, n)       # (nats, ppl)

    def generate(self, n, tokenizer, max_len=140, seed=42):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            prev, ids = BOS_ID, []
            for _ in range(max_len):
                nxt = int(rng.choice(VOCAB_SIZE, p=self.probs[prev]))
                if nxt == EOS_ID:
                    break
                ids.append(nxt)
                prev = nxt
            out.append(tokenizer.decode(ids))
        return out
```

## Step 3: `scripts/run_baseline.py` (~30 min)

It reads only train and val. The test split stays untouched until the end of the project. Results go to a JSON file in `results/baseline/`, which is git-ignored.

```python
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
```

## Step 4: `tests/test_ngram.py` (~45 min)

```python
import random

import pytest

from ngram import BigramLM
from utils import STANDARD_AA, AminoAcidTokenizer


def rand_seqs(n, seed):
    rng = random.Random(seed)
    return ["".join(rng.choices(STANDARD_AA, k=rng.randint(100, 140))) for _ in range(n)]


def test_rows_sum_to_one_and_no_mass_on_pad_bos():
    tok = AminoAcidTokenizer()
    lm = BigramLM().fit(rand_seqs(200, 0), tok)
    assert lm.probs.sum(axis=1) == pytest.approx(1.0)
    assert (lm.probs[:, [tok.pad_id, tok.bos_id]] == 0).all()


def test_random_data_perplexity_close_to_20():
    tok = AminoAcidTokenizer()
    lm = BigramLM().fit(rand_seqs(2000, 1), tok)
    _, ppl = lm.nats_per_token(rand_seqs(500, 2), tok)
    assert 19.5 < ppl < 22          # < 15 on pure random would mean a bug or a leak


def test_learns_a_pattern():
    tok = AminoAcidTokenizer()
    lm = BigramLM().fit(["ACACACACAC"] * 200, tok)
    _, ppl = lm.nats_per_token(["ACACACACAC"] * 20, tok)
    assert ppl < 2


def test_generate_is_decodable_and_deterministic():
    tok = AminoAcidTokenizer()
    lm = BigramLM().fit(rand_seqs(300, 3), tok)
    a = lm.generate(20, tok, max_len=140, seed=7)
    assert a == lm.generate(20, tok, max_len=140, seed=7)
    assert all(len(s) <= 140 for s in a)
    for s in a:
        if s:
            tok.encode(s)           # raises ValueError if invalid
```

## Step 5: run locally on toy data

```powershell
python -m pytest -q                        # expect 15 passed
python scripts/run_baseline.py
powershell -ExecutionPolicy Bypass -File scripts\verify_workflow.ps1
```
**Expected on toy:** perplexity around 20–21 (the toy data is random, and val has only about 10 sequences, so it's noisy). Below 15 means a bug or a leak.

## Step 6: test A's real sample (Thursday/Friday)

Create `scripts/check_sample.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import AminoAcidTokenizer

tok = AminoAcidTokenizer()
seqs = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
bad = []
for s in seqs:
    try:
        assert tok.decode(tok.encode(s)) == s
    except Exception as e:
        bad.append((s[:30], repr(e)))
print(f"{len(seqs) - len(bad)}/{len(seqs)} round-trips OK")
for b in bad[:10]:
    print("FAIL", b)
```
```powershell
python scripts/check_sample.py C:\path\to\sample_100_vhh.txt
```
You need 100/100. If not, tell A which letters appear (usually `X` or `-`).

## Step 7: push, then Colab with the real data

```powershell
git add ngram.py scripts/run_baseline.py scripts/check_sample.py tests/test_ngram.py
git commit -m "Sprint1 B: bigram baseline + tests"
git push -u origin b/sprint1-baseline
```

Colab cell, from a fresh runtime. Use the branch name until you merge, then use `main`:

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ["AI_PROJECT_DIR"] = "/content/drive/MyDrive/AI_Project_Shared"

!rm -rf /content/Recherche_AI
!git clone -b b/sprint1-baseline https://github.com/AhmadBelhaj4/Recherche_AI.git /content/Recherche_AI
%cd /content/Recherche_AI
!pip install -q -r requirements.txt
!python -m pytest -q
!python scripts/run_baseline.py --data-version v1   # exact name that NB01/NB02 produced
```
NB01 and NB02 must have written `processed/<version>/train.csv` and `val.csv` to the Drive first. On real VHH data, the perplexity should come out clearly below 20. If it is close to 20, you're probably reading toy data, so check the `data_version=` line it prints.

When everything is green, open a PR into `main` and have someone from Team A review it.

## Step 8: Friday demo (5 min)

- **Tokenizer:** `AminoAcidTokenizer().encode("EVQ")` gives `[21, 3, 17, 13, 22]`, and the round-trip passes on A's 100 real sequences.
- **Baseline table:** toy perplexity (around 20) next to real perplexity (below 20), plus the 5 generated sequences.
- **Tests:** a screenshot of `pytest` showing 15 passed.

## Definition of Done (per task)

- Code is on Git and reviewed by the other team.
- Seed is fixed and the result is reproducible.
- Result saved in `results/baseline/*.json`.
- One sentence of interpretation: what the perplexity tells you about the data.

Next week (Sprint 2) you build on this with the DataLoader and the Transformer training loop. I can add the motif to the toy data now if you want, so the loss visibly drops when you sanity-check the Transformer.