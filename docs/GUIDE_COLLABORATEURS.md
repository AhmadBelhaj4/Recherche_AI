# 🧬 Guide Complet Collaborateur — Entraînement, Synchronisation & Évaluation BioLM Studio

Ce guide explique étape par étape à toute l'équipe comment :
1. **Entraîner les modèles (Transformer & LSTM) sur Google Colab avec GPU**.
2. **Sauvegarder automatiquement les checkpoints (`best.pt`) sur le Google Drive partagé `AI_Project_Shared`**.
3. **Synchroniser les modèles réels sur votre PC en une commande**.
4. **Calculer toutes les métriques de génération et de sampling (Phase F / NB05 & NB06)**.
5. **Lancer et utiliser le Dashboard interactif BioLM Studio en local**.

---

## 📋 Architecture Globale du Projet

```
+-------------------------------------------------------------+
|                     GOOGLE COLAB (GPU T4)                   |
|  - Entraînement sur le dataset réel SRR3544218 (242k seqs)  |
|  - NB03 (LSTM Baseline) ou NB04 (Causal Transformer)        |
|  - Section 7 : Sauvegarde automatique vers Google Drive     |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               GOOGLE DRIVE (AI_Project_Shared)              |
|  - models/transformer-notebook-srr/best.pt (3.15 MB)        |
|  - models/lstm-notebook-srr/best.pt (1.65 MB)               |
|  - results/runs/*/history.csv, results/baseline/*.json      |
+------------------------------+------------------------------+
                               |
                               v [scripts/sync_drive.py]
+-------------------------------------------------------------+
|                        PC LOCAL                             |
|  - models/ : Checkpoints PyTorch réels                      |
|  - evaluation.py : Métriques Phase F (Validité, Diversité)  |
|  - scripts/evaluate_generation.py : Benchmark 4 samplers    |
|  - scripts/serve_dashboard.py : Serveur BioLM Studio:8765   |
|  - dashboard/ : Interface Web Obsidian Dark Glassmorphic   |
+-------------------------------------------------------------+
```

---

## 🚀 Étape 1 : Entraînement sur Google Colab (GPU)

### 1.1. Ouvrir le Notebook sur Colab
Dans votre navigateur, ouvrez l'un des notebooks officiels du projet :
- **Transformer Causal** : `project1/notebooks/NB04_transformer_causal.ipynb`
- **LSTM Baseline** : `project1/notebooks/NB03_lstm_baseline.ipynb`

### 1.2. Activer l'accélérateur GPU
1. Dans le menu Colab, cliquez sur **Exécution** > **Modifier le type d'exécution**.
2. Sélectionnez **GPU T4** (gratuit) > Enregistrer.

### 1.3. Exécuter l'entraînement
1. Exécutez la **Cellule 0** (Bootstrap & Montage Google Drive) :
   - Autorisez l'accès à Google Drive lorsque Colab vous le demande.
2. Exécutez les cellules 1 à 6 pour charger les données réelles `SRR3544218` et lancer l'entraînement.
   - **Transformer** : 15 époques (~18-20 minutes). Perplexité cible : **~2.22**.
   - **LSTM** : 10 époques (~8 minutes). Perplexité cible : **~2.25**.

### 1.4. Sauvegarde Automatique (Section 7)
La **Section 7** déjà intégrée dans le notebook s'exécute à la fin de l'entraînement :
- Elle exporte automatiquement `best.pt`, `history.csv`, et `metrics.json` dans :
  `/content/drive/MyDrive/AI_Project_Shared/models/<RUN_ID>/best.pt`
- **Vous n'avez aucun fichier à télécharger manuellement !**

---

## 🔄 Étape 2 : Synchronisation sur votre PC Local

### Option Recommandée : Google Drive pour PC + `sync_drive.py`

1. Installez l'application gratuite **[Google Drive pour PC (Windows / Mac)](https://www.google.com/intl/fr_ALL/drive/download/)**.
2. Dans Google Drive Web, vérifiez que le dossier partagé **`AI_Project_Shared`** a bien un raccourci dans **Mon Drive** (*Clic droit > Ajouter un raccourci > Mon Drive*).
3. Ouvrez un terminal dans le répertoire du projet `reacherche_ai` et lancez :

```bash
python scripts/sync_drive.py
```

Le script détecte automatiquement votre lecteur Google Drive Windows (`G:\` ou `.shortcut-targets-by-id`) et synchronise :
- `models/transformer-notebook-srr/best.pt` (3.15 MB)
- `models/lstm-notebook-srr/best.pt` (1.65 MB)
- Les historiques d'entraînement (`history.csv`)
- Les métriques de baseline Bigram (`results/baseline/bigram_SRR3544218_a1.json`)

---

## 🧪 Étape 3 : Évaluation & Métriques de Génération (Phase F)

Pour évaluer la qualité biologique et comparer les 4 stratégies d'échantillonnage de `NB05` et `NB06`, lancez le benchmark automatisé :

```bash
python scripts/evaluate_generation.py --all --n-seqs 100
```

### Que fait ce script ?
Il génère pour chaque modèle les 4 configurations canoniques du contrat :
1. **`greedy`** : Échantillonnage déterministe argmax ($T=1.0, k=0, p=1.0$).
2. **`topk10_T08`** : Filtrage Top-$k$ ($k=10, T=0.8$).
3. **`nucleus_p95_T10`** : Échantillonnage Nucléus ($top-p=0.95, T=1.0$).
4. **`nucleus_p80_T12`** : Échantillonnage Nucléus haute température ($top-p=0.80, T=1.2$).

### Métriques Calculées & Critères d'Excellence :

| Métrique | Formule / Définition | Valeur Cible | Interprétation Biologique |
| :--- | :--- | :---: | :--- |
| **Validité Biologique** | % séquences sur 20 AA standards dans $[100, 140]$ | **100.0%** | Aucune hallucination de caractère hors alphabet. |
| **Taux d'Unicité** | $\frac{\mid\text{set}(S)\mid}{\mid S\mid}$ | **100.0%** *(Stochastique)*<br>**2.0%** *(Greedy)* | Détecte l'effondrement de mode (mode collapse). Le greedy répète la même séquence, le top-k/nucléus génère 100% de candidats distincts. |
| **Diversité Levenshtein** | Distance d'édition moyenne normalisée | **0.28 – 0.36** | Variabilité idéale des boucles hypervariables (CDR1, CDR2, CDR3) sans casser le framework structural. |
| **Similarité Jaccard 3-mers** | $\frac{\mid k_1 \cap k_2\mid}{\mid k_1 \cup k_2\mid}$ | **0.25 – 0.35** | Diversité des tripeptides locaux. |
| **Divergence KL Compo** | $D_{KL}(P_{\text{gen}} \parallel P_{\text{train}})$ | **$< 0.05$ nats** | Fidélité statistique parfaite à la fréquence des acides aminés des VHH camélidés naturels. |
| **Taux de Nouveauté** | % séquences non présentes dans le train | **100.0%** | Preuve de génération *de novo* et non de par cœur/mémorisation. |
| **Hydropathie (GRAVY)** | Échelle de Kyte-Doolittle | **-0.45 à -0.35** | Solubilité biologique assurée (évite les agrégats hydrophobes insolubles). |
| **Pont Disulfure (Cys)** | Présence des cystéines C23 et C104 | **100.0%** | Conservation absolue du repliement canonique immunoglobuline. |

Les fichiers FASTA générés sont stockés dans `results/generated/<model_id>/` et les résultats JSON dans `results/evaluations/`.

---

## 🖥️ Étape 4 : Lancer BioLM Studio en Local

Lancez le serveur local :

```bash
python scripts/serve_dashboard.py --port 8765
```

Ouvrez ensuite votre navigateur sur :
👉 **`http://localhost:8765`**

### Fonctionnalités de BioLM Studio :
1. **Leaderboard** : Classement officiel des modèles réels (Transformer SRR : PPL 2.22 vs Bigram : PPL 11.83). Seuls les modèles réels Colab sont affichés (les jeux jouets sont masqués).
2. **Training Curves** : Courbes dynamiques interactives de la perte (nats) et de la perplexité par époque.
3. **Compare** : Comparateur en radar face-à-face entre le Transformer et le LSTM.
4. **Generation Playground** : Générateur interactif en direct branché sur PyTorch CPU. Vous pouvez entrer un amorçage (ex: `HVQLVESGG...`), régler la température, top-k, top-p, et inspecter la composition résidu par résidu.
5. **🧪 Sampling & Benchmarks (Phase F)** : Tableau comparatif complet des 4 stratégies de sampling avec KPI de validité, unicité, diversité, et téléchargement direct des fichiers FASTA.
6. **What-If Studio** : Simulateur architectural pour dimensionner vos modèles avant lancement sur GPU.

---

## 🛠️ Dépannage & Questions Fréquentes (FAQ)

### Q1 : `sync_drive.py` m'indique que le dossier `AI_Project_Shared` est introuvable ?
- **Solution** : Ouvrez Google Drive Web. Rendez-vous dans "Partagés avec moi", faites un clic droit sur `AI_Project_Shared` et choisissez **Ajouter un raccourci** dans **Mon Drive**. Vérifiez ensuite que l'application Google Drive pour PC est bien connectée.

### Q2 : Erreur `PermissionError: [WinError 5] Accès refusé` sur Windows ?
- **Solution** : Cela arrive si les DLL C-extensions de PyTorch/NumPy sont bloquées par une sandbox restreinte. Exécutez toujours vos scripts dans l'environnement virtuel du projet :
  ```bash
  .\.venv\Scripts\python.exe scripts/serve_dashboard.py
  ```

### Q3 : Comment lancer les tests unitaires ?
Pour vérifier que l'ensemble du pipeline est vert à 100% :
```bash
python -m pytest tests/
```
*(35 tests unitaires passent en ~13 secondes avec 100% de succès).*

---

## 👥 Synthèse pour l'Équipe

| Rôle / Action | Commande | Résultat Attendu |
| :--- | :--- | :--- |
| **Entraînement GPU** | Colab `NB04_transformer_causal.ipynb` | Modèle entraîné et sauvé sur Drive (`best.pt`). |
| **Sync Checkpoints** | `python scripts/sync_drive.py` | Modèles réels rapatriés dans `models/`. |
| **Évaluation Phase F** | `python scripts/evaluate_generation.py --all` | FASTA générés et métriques calculées. |
| **Dashboard Studio** | `python scripts/serve_dashboard.py` | BioLM Studio actif sur `http://localhost:8765`. |
