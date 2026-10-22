"""models.py - model architectures shared across notebooks.

LSTMLM and CausalTransformerLM are defined here so they can be imported
from any notebook or test without redefining the class each time.
"""

import torch
import torch.nn as nn


class LSTMLM(nn.Module):
    """LSTM language model with weight tying (Bengio et al. 2003)."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, embed_dim) if hidden_dim != embed_dim else nn.Identity()
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        # Weight tying : head.weight = embed.weight
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.1)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        out = self.proj(out)
        logits = self.head(out)
        return logits, hidden


class CausalTransformerLM(nn.Module):
    """Causal (decoder-only) Transformer LM with Pre-LN (Xiong et al. 2020)."""

    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=6,
                 dim_feedforward=512, max_len=200, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight
        # GPT-2 style init (std 0.02). Avec N(0,1) + weight tying, la loss initiale vaut ~47 nats au lieu de ~3.1.
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

    def forward(self, x):
        b, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(b, -1)
        emb = self.token_embed(x) + self.pos_embed(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        out = self.transformer(emb, mask=mask, is_causal=True)
        out = self.norm(out)
        return self.head(out)
