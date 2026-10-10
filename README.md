# Recherche_AI

> **Feuille de route sur 1 an** — Génération contrôlée de séquences protéiques par modèles de langage autorégressifs, diffusion discrète et ponts de Schrödinger.

---

## 🗺️ Vue d'ensemble du projet

Ce dépôt regroupe l'ensemble des travaux de recherche en **génération de séquences protéiques** (notamment des VHH / nanobodies), organisés en **5 phases progressives** sur 12 à 18 mois.

```
Phase 1 (Mois 1-3)   →  Fondations : modèles autorégressifs (LSTM, Transformer)
Phase 2 (Mois 3-6)   →  Diffusion discrète (D3PM, Masked Diffusion)
Phase 3 (Mois 6-9)   →  Conditionnement & contrôle (guidance, évaluation structurale)
Phase 4 (Mois 9-12)  →  Ponts de Schrödinger discrets (OT, IPFP, SB)
Phase 5 (Mois 12-18) →  Projet de fin d'études (intégration complète)
```

---

## Phase 1 — Fondations : Modèles Autorégressifs (Mois 1–3)

### Projet 1.1 — Génération de séquences par modèle de langage

**Sujet :** Entraîner des modèles autorégressifs (LSTM et Transformer décodeur-only) sur des séquences protéiques — VHH (nanobodies, OAS) et famille SH2 (UniProt).

**Objectif :** Maîtriser le pipeline complet :
- Exploration et prétraitement des données
- Tokenisation (vocabulaire V=23 : 20 AA + `<PAD>` + `<BOS>` + `<EOS>`)
- Entraînement avec log-vraisemblance exacte (`nats/token`)
- Sampling : greedy, top-k, nucleus (température, top-p)
- Évaluation : validité, diversité (Levenshtein, Jaccard k-mers), cohérence statistique (KL divergence), nouveauté (cd-hit)

**Livrable :** Rapport + notebooks reproductibles + 10 000 séquences générées et analysées.

> ⚠️ **Piège à éviter :** Ne pas toucher à la diffusion avant de maîtriser la log-vraisemblance, le sampling et l'évaluation. La diffusion discrète sans ces bases devient une boîte noire indéboguable.

### Projet 1.2 — Classifieur de séquences (apprentissage supervisé)

**Sujet :** Prédire la nature d'une séquence (peptide vs protéine ; VHH vs scFv vs protéine ordinaire ; stabilité thermique binaire si labels disponibles).

**Objectif :** Construire un jeu de données étiqueté propre, entraîner un classifieur (Transformer léger type ESM-2 petit ou CNN 1D), analyser les erreurs.

**Pourquoi c'est important :** Ce classifieur servira de **récompense / critic** dans toutes les phases ultérieures (diffusion guidée, SB avec coût de structure).

---

## Phase 2 — Diffusion Discrète (Mois 3–6) : *"Générer par débruitage"*

### Projet 2.1 — D3PM / diffusion masquée sur séquences courtes

**Sujet :** Implémenter un modèle de diffusion discrète catégorielle (D3PM ou absorbing-state / masked diffusion) sur des peptides courts (10–50 AA).

**Formulation clé :**
- Processus forward `q(xₜ | xₜ₋₁)` sur le simplexe catégoriel
- Matrice de transition `Qₜ`, posterior `q(xₜ₋₁ | xₜ, x₀)`
- Perte ELBO/variational, réseau `θ(xₜ, t)` ou prédiction de `x₀`

**Évaluation :** Perplexité par transition, diversité, comparaison tête-à-tête avec le modèle AR de la Phase 1 (mêmes données, même budget de calcul).

### Projet 2.2 — Masked Diffusion / "LM à masques itérés"

**Sujet :** Masked discrete diffusion (MDLM / SEDD-like) sur des VHH (~110–130 AA).

**Objectif :** Sampling semi-autoregressif (démasquage par blocs), conditioning sur régions conservées (CDR fixes, scaffold constant), évaluation de la qualité des CDR générés.

**Livrable :** Générateur de CDR conditionné + benchmark AR vs diffusion.

> ⚠️ **Piège :** Matrices `Qₜ` mal normalisées, absorbing state mal gérée, accumulation d'erreurs. Règle : **formules exactes, pas d'epsilon numérique non justifié**.

---

## Phase 3 — Conditionnement & Contrôle (Mois 6–9) : *"Générer vers une cible"*

### Projet 3.1 — Diffusion guidée / inpainting fonctionnel

**Sujet :** Génération conditionnée — fixer un motif fonctionnel (signal peptide, epitope binding region, site catalytique) et générer le reste ; ou class-guided diffusion avec le classifieur de la Phase 1.

**Compétences :** Classifier guidance vs classifier-free guidance en espace discret, inpainting par masquage partiel du forward process, trade-off qualité/diversité (CFG distille les modes), évaluation conditionnelle.

### Projet 3.2 — Évaluation structurale légère

**Sujet :** Pipeline d'évaluation structurelle sans entraîner de modèle structurel soi-même — prédire la structure des séquences générées (ESMFold), mesurer **pLDDT**, **scTM**, **TM-score** contre PDB.

**Objectif :** Filtrer les séquences générées par pliabilité, quantifier le taux de "designs plausibles".

---

## Phase 4 — Ponts de Schrödinger Discrets (Mois 9–12) : *"Le cœur théorique"*

### Projet 4.1 — Sinkhorn / IPFP en espace discret

**Sujet :** Implémenter le schéma de Fortet-IPFP (ou Sinkhorn) entre deux distributions de séquences, d'abord sur l'espace produit de petites séquences, puis avec un noyau de similarité (k-mers, embeddings ESM).

**Formulation :**
```
min P : D_KL(P ‖ Rₛ)  sous contraintes de marges (μ, ν)
→ itérations potentials (u, v), contraction de coefficients
```

**Compétences :** OT entropique, marges discrètes, convergence d'IPFP, stabilité.

### Projet 4.2 — Schrödinger Bridge discret sur graphes / séquences *(Avancé)*

**Sujet :** Pont de Schrödinger **entre deux lois sur les séquences** via une chaîne de Markov à temps discret, où le noyau de référence est la dynamique forward de diffusion discrète (Phase 2) et les marges sont deux distributions empiriques.

**Exemples de paires de marges :**
- Séquences germlines → séquences après maturation somatique
- VHH de liaison faible → VHH de liaison forte (yeast display)
- Distribution naturelle → distribution optimisée pour une propriété

**Cadre :**
```
P(x₀:T) ∝ f(x₀) · R(x₀:T) · f̂(xT)
résolution : Sinkhorn itéré / DSB algorithmique
```

**Livrable :** Implémentation exacte + preuve/justification + benchmark contre (a) fine-tuning supervisé, (b) classifier guidance, (c) diffusion non pontée.

---

## Phase 5 — Projet de Fin d'Études (Mois 12–18) : *L'intégration*

### Architecture du PFE

**Titre de travail :**
> *"Pont de Schrödinger discret pour la génération contrôlée de VHH : du bruitage catégoriel au transport optimal de séquences"*

**Structure en 4 volets :**

| Volet | Description |
|-------|-------------|
| **1. Modèle génératif de base** | Masked discrete diffusion entraîné sur VHH (OAS + curation IMGT), formules exactes |
| **2. Module de contrôle** | Conditionnement sur CDR scaffold + guidance par classifieurs de développabilité (stabilité, agrégation, immunogénicité) |
| **3. Cœur SB discret** | Pont entre deux marges réelles et scientifiquement motivées |
| **4. Évaluation complète** | Diversité, log-lik, qualité CDR, pliabilité (ESMFold pLDDT), benchmarks rigoureux |

### Ce qui distingue un PFE "professionnel"

- ✅ **Question scientifique précise** (le pont répond-il mieux qu'un fine-tuning ? À quelles conditions de marges ?)
- ✅ **Formules justifiées et exactes** — chaque approximation numérique documentée et validée
- ✅ **Benchmark reproductible** : seeds fixes, données versionnées, métriques standard
- ✅ **Analyse théorique** : vérification empirique des propriétés de convergence/stabilité du schéma SB
- ✅ **Perspective** : ce qui manque pour passer à l'échelle (complexité de Sinkhorn, noyaux, séquences longues)

---

## 📁 Structure du dépôt

```
Recherche_AI/
├── project1/                          # Phase 1 — Modèles autorégressifs
│   ├── notebooks/
│   │   ├── NB01_exploration_donnees.ipynb
│   │   ├── NB02_tokenization_datasets_splits.ipynb
│   │   ├── NB03_lstm_baseline.ipynb
│   │   ├── NB04_transformer_causal.ipynb
│   │   ├── NB05_generation_sampling.ipynb
│   │   └── NB06_evaluation_metriques.ipynb
│   ├── data/
│   ├── models/
│   ├── results/
│   ├── requirements.txt
│   └── GUIDE.md                       # Guide étape par étape Phase 1
├── README.md
└── Feuille de route 1 an.odt          # Feuille de route complète
```

---

## 🛠️ Installation

```bash
git clone https://github.com/AhmadBelhaj4/Recherche_AI.git
cd Recherche_AI/project1
pip install -r requirements.txt
```

**Dépendances principales :**
- `torch >= 2.0`
- `numpy`, `pandas`, `matplotlib`
- `jupyter`

---

## 📚 Références clés

| Référence | Usage |
|-----------|-------|
| Hochreiter & Schmidhuber (1997) | Baseline LSTM |
| Vaswani et al. (2017) | Transformer |
| Radford et al. (2019) | GPT-2 décodeur-only |
| Austin et al. (2021) | D3PM — diffusion discrète |
| Holtzman et al. (2020) | Nucleus sampling |
| Ferruz et al. (2022) | ProtGPT2 — précédent direct |
| Olsen et al. (2022) | Observed Antibody Space (OAS) |
| Fu et al. (2012) | CD-HIT — déduplication |
| Cover & Thomas (2006) | Entropie, KL divergence |