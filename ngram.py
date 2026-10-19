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
