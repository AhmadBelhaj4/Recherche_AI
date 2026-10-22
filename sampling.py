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
