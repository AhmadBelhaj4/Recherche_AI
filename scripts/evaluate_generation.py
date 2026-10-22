"""evaluate_generation.py - Benchmark and evaluate generation sampling strategies.

Implements the complete Phase F evaluation pipeline (NB05 & NB06) for real Colab models:
- Runs generation across 4 canonical sampling configurations:
    1. greedy (Argmax)
    2. topk10_T08 (Top-k = 10, T = 0.8)
    3. nucleus_p95_T10 (Nucleus p = 0.95, T = 1.0)
    4. nucleus_p80_T12 (Nucleus p = 0.80, T = 1.2)
- Evaluates:
    - Biological Validity (% 20 standard AA, length [100, 140])
    - Uniqueness (% non-redundant sequences, mode collapse test)
    - Diversity (Mean Normalized Pairwise Levenshtein Distance)
    - Local Motif Overlap (3-mer Jaccard Similarity)
    - Statistical Fidelity (KL Divergence on Amino Acid Distribution)
    - Novelty vs Train Set (% unseen candidates)
    - Physicochemical properties (GRAVY hydropathy, Net Charge, Canonical Cys pairs)
- Saves FASTA files in results/generated/<model_id>/
- Saves JSON summary in results/evaluations/
- Can export directly to dashboard/data.json

Usage:
    python scripts/evaluate_generation.py --model-id transformer-notebook-srr --n-seqs 100
    python scripts/evaluate_generation.py --model-id lstm-notebook-srr --n-seqs 100
    python scripts/evaluate_generation.py --all
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Set

import numpy as np

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation import (
    SRR3544218_AA_FREQ,
    evaluate_sequences,
    is_valid_sequence
)
from sampling import generate
from utils import AminoAcidTokenizer, set_seed


CONFIGS = [
    ("greedy", {"greedy": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0}),
    ("topk10_T08", {"greedy": False, "temperature": 0.8, "top_k": 10, "top_p": 1.0}),
    ("nucleus_p95_T10", {"greedy": False, "temperature": 1.0, "top_k": 0, "top_p": 0.95}),
    ("nucleus_p80_T12", {"greedy": False, "temperature": 1.2, "top_k": 0, "top_p": 0.80}),
]


def load_model(model_id: str, device: str = "cpu"):
    """Load neural model and checkpoint weights."""
    import torch
    from models import CausalTransformerLM, LSTMLM

    ckpt_path = ROOT / "models" / model_id / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location=device)
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    if "lstm" in model_id.lower():
        model = LSTMLM(vocab_size=23, embed_dim=128, hidden_dim=256, num_layers=1)
    else:
        num_layers = 4
        if isinstance(raw, dict) and "config" in raw:
            num_layers = raw["config"].get("num_layers", 4)
        model = CausalTransformerLM(
            vocab_size=23,
            embed_dim=128,
            num_heads=4,
            num_layers=num_layers,
            dim_feedforward=512,
            max_len=200,
            dropout=0.0
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_train_set() -> Set[str]:
    """Load training sequences for novelty comparison if available."""
    train_paths = [
        ROOT / "data" / "processed" / "SRR3544218" / "train.csv",
        ROOT / "data" / "train.csv",
        ROOT / "data" / "toy" / "processed" / "toy" / "train.csv"
    ]
    for p in train_paths:
        if p.exists():
            import csv
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return {row["sequence"] for row in reader if "sequence" in row}
    return set()


def evaluate_model(
    model_id: str,
    n_seqs: int = 100,
    max_len: int = 140,
    device: str = "cpu",
    seed: int = 42
) -> Dict[str, dict]:
    """Generate sequences for all 4 configurations and compute full evaluation metrics."""
    set_seed(seed)
    print("=" * 75)
    print(f"  ÉVALUATION DE GÉNÉRATION & SAMPLING — {model_id}")
    print(f"  Nombre de séquences par stratégie : {n_seqs} | Longueur max : {max_len}")
    print("=" * 75)

    model = load_model(model_id, device=device)
    train_set = load_train_set()
    if train_set:
        print(f"📚 Jeu d'entraînement chargé : {len(train_set):,} séquences de référence.")
    else:
        print("ℹ️ Jeu d'entraînement absent localement : Nouveauté = 100% (non pénalisée).")

    gen_dir = ROOT / "results" / "generated" / model_id
    gen_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for name, params in CONFIGS:
        print(f"\n🔬 Génération [{name}] en cours...")
        seqs, dones = generate(
            model=model,
            n=n_seqs,
            device=device,
            max_len=max_len,
            greedy=params["greedy"],
            temperature=params["temperature"],
            top_k=params["top_k"],
            top_p=params["top_p"],
            seed=seed,
            batch_size=64
        )

        # Save to FASTA
        fasta_path = gen_dir / f"{name}.fasta"
        with open(fasta_path, "w", encoding="utf-8") as f:
            for idx, s in enumerate(seqs):
                f.write(f">{model_id}_{name}_{idx} len={len(s)}\n{s}\n")

        # Run evaluation metrics
        report = evaluate_sequences(
            sequences=seqs,
            train_set=train_set if train_set else None,
            ref_composition=SRR3544218_AA_FREQ,
            min_len=DEFAULT_MIN_LEN if "DEFAULT_MIN_LEN" in globals() else 100,
            max_len=max_len,
            n_pairs=min(500, max(20, n_seqs * 2)),
            seed=seed
        )
        report["sample_sequences"] = seqs[:3]
        report["fasta_file"] = str(fasta_path.relative_to(ROOT))
        results[name] = report

        print(f"   ✓ Validité       : {report['validity_rate'] * 100:.1f}%")
        print(f"   ✓ Unicité        : {report['uniqueness_rate'] * 100:.1f}%")
        print(f"   ✓ Levenshtein    : {report['levenshtein_diversity']:.3f}")
        print(f"   ✓ Jaccard 3-mers : {report['kmer_jaccard_similarity']:.3f}")
        print(f"   ✓ KL Divergence  : {report['kl_divergence']:.4f}")
        print(f"   ✓ Nouveauté      : {report['novelty_rate'] * 100:.1f}%")
        print(f"   ✓ Hydropathie    : {report['mean_gravy']} (GRAVY Kyte-Doolittle)")
        print(f"   ✓ Cys Pont Disulf: {report['canonical_cysteine_rate'] * 100:.1f}%")

    # Print summary table (NB06 style)
    print("\n" + "=" * 90)
    print(f"TABLEAU DE SYNTHÈSE FINAL — COMPARAISON DES STRATÉGIES ({model_id})")
    print("=" * 90)
    header = f"{'Configuration':<18} | {'Validité':<8} | {'Unicité':<8} | {'Levenshtein':<11} | {'Jaccard 3m':<10} | {'KL Compo':<8} | {'Nouveauté':<9} | {'GRAVY':<6}"
    print(header)
    print("-" * 90)
    for name, r in results.items():
        row = (
            f"{name:<18} | "
            f"{r['validity_rate'] * 100:>7.1f}% | "
            f"{r['uniqueness_rate'] * 100:>7.1f}% | "
            f"{r['levenshtein_diversity']:>11.3f} | "
            f"{r['kmer_jaccard_similarity']:>10.3f} | "
            f"{r['kl_divergence']:>8.4f} | "
            f"{r['novelty_rate'] * 100:>8.1f}% | "
            f"{r['mean_gravy']:>6.2f}"
        )
        print(row)
    print("=" * 90)

    # Save to results/evaluations/
    eval_dir = ROOT / "results" / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_json = eval_dir / f"{model_id}_generation_eval.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n💾 Évaluation sauvegardée : {out_json}")

    return results


def update_dashboard_with_evaluations():
    """Merge evaluation results into dashboard/data.json."""
    data_file = ROOT / "dashboard" / "data.json"
    if not data_file.exists():
        return

    data = json.loads(data_file.read_text(encoding="utf-8"))
    eval_dir = ROOT / "results" / "evaluations"

    all_evals = {}
    if eval_dir.exists():
        for f in eval_dir.glob("*_generation_eval.json"):
            model_id = f.stem.replace("_generation_eval", "")
            all_evals[model_id] = json.loads(f.read_text(encoding="utf-8"))

    data["generation_evaluations"] = all_evals
    data_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"📊 Dashboard data.json actualisé avec {len(all_evals)} évaluations de modèles.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate generation sampling strategies (Phase F).")
    parser.add_argument("--model-id", default="transformer-notebook-srr", help="Model run ID to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all available real Colab models")
    parser.add_argument("--n-seqs", type=int, default=100, help="Number of sequences per sampling config")
    parser.add_argument("--max-len", type=int, default=140, help="Maximum sequence generation length")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    models_to_eval = []
    if args.all:
        for m_dir in (ROOT / "models").iterdir():
            if m_dir.is_dir() and (m_dir / "best.pt").exists() and "toy" not in m_dir.name.lower():
                models_to_eval.append(m_dir.name)
    else:
        models_to_eval = [args.model_id]

    for mid in models_to_eval:
        evaluate_model(
            model_id=mid,
            n_seqs=args.n_seqs,
            max_len=args.max_len,
            device=args.device,
            seed=args.seed
        )

    update_dashboard_with_evaluations()


if __name__ == "__main__":
    main()
