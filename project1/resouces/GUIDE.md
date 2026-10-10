# Projet 1 — Génération de Séquences Protéiques par Modèles de Langage Autorégressifs

> Guide phase par phase, étape par étape, extrait de `Projet 1.odt`

---

## Vue d'ensemble

| Phase | Titre | Semaine(s) |
|-------|-------|------------|
| A | Exploration des données | 1 |
| B | Tokenisation, datasets, splits | 2 |
| C | Modèles (LSTM & Transformer) | 3–4 |
| D | Entraînement | 4–5 |
| E | Génération | 6 |
| F | Évaluation | 7–8 |
| G | Rapport final | 8 |

---

## Phase A — Exploration des données (Semaine 1)

### Tâche 1.1 — Source des données
- Télécharger les séquences **VHH** (nanobodies) depuis la base **Observed Antibody Space (OAS)** (Olsen et al. 2022).
- Télécharger les séquences de la famille **PF00017** (SH2) depuis **UniProt** (UniProt Consortium 2023).
- 📓 NB01, cellule 1.

### Tâche 1.2 — Statistiques de base
- Calculer : nombre total de séquences, longueur min / max / moyenne / médiane, histogramme des longueurs.
- 📓 NB01, cellule 2.

### Tâche 1.3 — Analyse de composition
- Calculer la fréquence de chacun des **20 AA**.
- Représenter avec un **logo de séquence (PWM)**.
- Référence : Schneider & Stephens 1990.
- 📓 NB01, cellules 3–4.

### Tâche 1.4 — Entropie et conservation
- Calculer l'entropie de Shannon position par position (bits).
- Identifier les positions conservées vs variables.
- 📓 NB01, cellules 5–6.

### Tâche 1.5 — Détection des doublons
- Déduplication exacte (`set`) + mesure du taux de redondance.
- Justifier (par écrit) pourquoi une déduplication partielle (cd-hit, 90% identité, Fu et al. 2012) sera nécessaire en Phase 3.
- 📓 NB01, cellule 7.

### Tâche 1.6 — Analyse rédigée (demi-page du rapport)
- Questions : longueurs, conservation, redondance.
- 📓 NB01, dernier bloc markdown.

---

## Phase B — Tokenisation, Datasets, Splits (Semaine 2)

### Tâche 1.7 — Tokenisation exacte
- Vocabulaire : **20 AA + `<PAD>` (idx 20) + `<BOS>` (idx 21) + `<EOS>` (idx 22)** → **V = 23**
- Encodage : `[BOS] x₁…xₗ [EOS]`, padding à droite.
- Sous-tâches :
  - Implémenter `encode()` / `decode()`
  - Test obligatoire : **round-trip sur 1000 séquences aléatoires**
- Pourquoi EOS ? Le modèle apprend la terminaison, indispensable pour la génération de longueur variable.
- 📓 NB02, cellules 1–2.

### Tâche 1.8 — Dataset, collate, DataLoader
- `SequenceDataset` + `collate` qui empile les séquences déjà paddées.
- **Validation :** batch de sortie de forme `(B, L)`, dtype `long`.
- 📓 NB02, cellule 3.

### Tâche 1.9 — Splits train / val / test
- Split après déduplication : **90% / 5% / 5%**, seed fixe.
- Expliquer la fuite de quasi-doublons du split aléatoire + amélioration future par split clusterisé (cd-hit / MMseqs2).
- **Validation :** les trois jeux sont disjoints.
- 📓 NB02, cellule 4.

---

## Phase C — Modèles (Semaines 3–4)

### Tâche 2.1 — Formulation mathématique *(à rédiger avant de coder)*

```
P_θ(x₁…xₗ) = ∏ₜ P_θ(xₜ | x<ₜ)

L = -(1/N_réel) Σ log P_θ(xₜ | x<ₜ)   [sur tokens non-PAD uniquement]
```

### Tâche 2.2 — Baseline LSTM
- Références : Hochreiter & Schmidhuber 1997 ; Bengio et al. 2003.
- Architecture : `embedding 128 → LSTM 256 (1 couche) → head vocab`
  - Weight tying : `head.weight = embed.weight`
- Sous-tâches : classe `LSTMLM`, loss exacte (`ignore_index=PAD`), test forward.
- 📓 NB03, cellules 1–2.

### Tâche 2.3 — Baseline triviale *(à calculer AVANT d'entraîner)*
- `nats/token = entropie de la composition marginale` — borne inférieure honnête que le modèle doit battre.

### Tâche 2.4 — Transformer décodeur-only
- Références : Vaswani et al. 2017 ; Radford et al. 2019 (GPT-2) ; Xiong et al. 2020 (pré-norm).
- Architecture : embedding + embedding positionnel appris, 6 blocs `TransformerEncoderLayer` (pré-norm, GELU, `batch_first`), masque causal `M[t,s] = 1 si s > t`.
- Sous-tâches : classe `CausalTransformerLM` + test unitaire du masque (un token futur ne doit pas changer les logits passés).
- 📓 NB04, cellules 1–2.

### Tâche 2.5 — Comptage des paramètres
- Tableau LSTM vs Transformer : paramètres, mémoire, temps par epoch.

---

## Phase D — Entraînement (Semaines 4–5)

### Tâche 3.1 — Boucle d'entraînement *(commune aux deux modèles)*
- **AdamW** : lr 1e-3 (LSTM) / 6e-4 (Transformer), weight_decay 0.01
- Scheduler cosinus + gradient clipping à **1.0**
- 📓 NB03, cellule 3.

### Tâche 3.2 — Évaluation exacte pendant l'entraînement
- Métrique principale : `nats/token = Σ NLL / nombre de tokens réels` *(pas la moyenne par batch !)*
- Early stopping : sauvegarde du meilleur état val.
- 📓 NB03, cellule 3.

### Tâche 3.3 — Entraîner le LSTM
- **10 epochs**, CPU OK.
- 📓 NB03, cellule 4.

### Tâche 3.4 — Entraîner le Transformer
- **15 epochs**, GPU conseillé.
- 📓 NB04, cellule 3.

### Tâche 3.5 — Évaluation finale sur test
- `test nats/token` + `perplexité = exp(nats/token)` pour les deux modèles + baseline composition.
- 📓 NB06, cellule 2.

### Tâche 3.6 — Checkpoints et journal d'expériences
- Un tableau par run : config, seed, val/test nats, durée.

---

## Phase E — Génération (Semaine 6)

### Tâche 4.1 — Formules exactes de sampling *(à rédiger)*

| Stratégie | Formule |
|-----------|---------|
| Température | `p = softmax(z / T)` |
| Greedy | `argmax` |
| Top-k | Renormaliser sur les k meilleurs (Fan et al. 2018) |
| Nucleus | Plus petit ensemble de masse cumulée ≥ p, renormalisation exacte (Holtzman et al. 2020) |

> Point d'exactitude : `(cum - sp) < top_p` — la définition exacte, pas d'epsilon artificiel.

### Tâche 4.2 — Fonction de génération
- Décodage itératif depuis `<BOS>`, arrêt à `<EOS>`.
- Masque des séquences terminées (pas de modification après EOS).
- Seeds fixes et distincts par batch pour la reproductibilité.
- 📓 NB05, cellules 1–2.

### Tâche 4.3 — Générer 4 configurations

| # | Configuration | N séquences |
|---|--------------|-------------|
| 1 | Greedy | 10 000 |
| 2 | Top-k=10, T=0.8 | 10 000 |
| 3 | Nucleus p=0.95, T=1.0 | 10 000 |
| 4 | Nucleus p=0.80, T=1.2 | 10 000 |

- Sauvegarder en **FASTA**.
- 📓 NB05, cellule 3.

### Tâche 4.4 — Inspection qualitative
- 20 séquences par configuration : motifs répétés ? longueurs plausibles ?

### Tâche 4.5 — Hypothèses à tester en Phase F
- Le greedy dégénère-t-il ? (réf. Holtzman et al. 2020) — à confirmer par les métriques.

---

## Phase F — Évaluation (Semaines 7–8)

### Tâche 5.1 — Validité
- 100% des séquences sur l'alphabet des 20 AA ; distribution des longueurs vs données réelles.

### Tâche 5.2 — Diversité *(par configuration)*
- Taux d'unicité
- Distance de Levenshtein normalisée moyenne (programmation dynamique exacte)
- Jaccard k-mers (k=3) sur 1000 paires
- 📓 NB06, cellules 3–4.

### Tâche 5.3 — Cohérence statistique
- `KL(p_gén ‖ p_données)` sur la composition AA (somme exacte sur les 20 lettres)
- Comparaison visuelle des fréquences
- 📓 NB06, cellule 5.

### Tâche 5.4 — Nouveauté
- Pipeline : `cd-hit -c 0.9` entre générés et train (Fu et al. 2012) ; contrôle interne par k-mer containment.
- 📓 NB06, cellule 6.

### Tâche 5.5 — Tableau de synthèse et figures

| Config | Unicité | Levenshtein | Jaccard | KL compo | Nouveauté |
|--------|---------|-------------|---------|----------|-----------|
| Greedy | … | … | … | … | … |
| Top-k=10, T=0.8 | … | … | … | … | … |
| Nucleus p=0.95, T=1.0 | … | … | … | … | … |
| Nucleus p=0.80, T=1.2 | … | … | … | … | … |

- Interpréter le trade-off qualité / diversité.

### Tâche 5.6 — Limites *(rédaction honnête)*
- Split aléatoire
- Absence de validation structurelle (viendra avec ESMFold en Phase 3)
- Pas de contrôle fonctionnel — la diffusion discrète et le SB y répondront.

---

## Phase G — Rapport final (Semaine 8)

### Tâches 6.1–6.4 — Rédaction

| Section | Contenu |
|---------|---------|
| Données | Description jeu de données, statistiques |
| Méthodologie | Formules exactes, architecture, choix techniques |
| Résultats | Tableaux, figures, comparatifs |
| Discussion | Trade-offs, limites, perspectives |
| Reproductibilité | Seeds, versions, ordre d'exécution des notebooks |

> Structure précise fournie en fin de NB06.

---

## Fichiers livrés

| Fichier | Tâches couvertes |
|---------|-----------------|
| `NB01_exploration_donnees.ipynb` | 1.1 – 1.6 |
| `NB02_tokenization_datasets_splits.ipynb` | 1.7 – 1.9 |
| `NB03_lstm_baseline.ipynb` | 2.2 – 2.3, 3.1 – 3.4, 3.6 |
| `NB04_transformer_causal.ipynb` | 2.4 – 2.5, 3.5 |
| `NB05_generation_sampling.ipynb` | 4.1 – 4.5 |
| `NB06_evaluation_metriques.ipynb` | 5.1 – 5.6, structure du rapport |
| `README.md` | Ordre d'exécution et dépendances |

---

## Bibliographie

1. Hochreiter S., Schmidhuber J. (1997). *Long Short-Term Memory.* Neural Computation 9(8). — baseline LSTM.
2. Vaswani A. et al. (2017). *Attention Is All You Need.* NeurIPS 30. — transformer.
3. Radford A. et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI. — décodeur-only.
4. Bengio Y. et al. (2003). *A Neural Probabilistic Language Model.* JMLR 3.
5. Fan A., Lewis M., Dauphin Y. (2018). *Hierarchical Neural Story Generation.* ACL. — top-k.
6. Holtzman A. et al. (2020). *The Curious Case of Neural Text Degeneration.* ICLR. — nucleus.
7. Olsen T.H. et al. (2022). *Observed Antibody Space.* J. Immunology. — données VHH.
8. UniProt Consortium (2023). *UniProt Knowledgebase 2023.* NAR 51(D1).
9. Schneider T.D., Stephens R.M. (1990). *Sequence logos.* NAR 18(20). — PWM.
10. Fu L. et al. (2012). *CD-HIT.* Bioinformatics 28(23). — nouveauté/splits.
11. Cover T.M., Thomas J.A. (2006). *Elements of Information Theory*, 2e éd. — entropie, KL.
12. Ferruz N. et al. (2022). *ProtGPT2.* Nature Communications. — précédent direct.
