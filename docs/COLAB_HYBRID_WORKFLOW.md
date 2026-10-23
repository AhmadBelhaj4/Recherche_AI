# Guide du Workflow Hybride : Google Colab ⟷ BioLM Studio Local

Ce guide explique comment entraîner vos modèles de protéines sur les GPU de **Google Colab**, sauvegarder automatiquement les checkpoints et les métriques selon les meilleures pratiques de Machine Learning, et les tester immédiatement sur votre machine locale via **BioLM Studio**.

---

## 🏗️ Architecture du Workflow Hybride

| Composant | Rôle | Stockage | Pourquoi ? |
|---|---|---|---|
| **Poids lourds (`best.pt`)** | Checkpoints PyTorch (~5 à 100 Mo) | **Google Drive** (`/MyDrive/recherche_ai_models/`) | Ne pollue pas Git, persiste même si Colab se déconnecte |
| **Métriques (`history.csv`, `metrics.json`)** | Courbes de perte, perplexité (~5 Ko) | **GitHub** (`results/runs/<run_id>/`) | Synchronisation instantanée en 1 commande `git pull` |
| **Échantillons Démo (`sample_generations`)** | 5 à 10 séquences générées sur GPU | Dans `metrics.json` | Permet de tester le modèle sur PC **sans télécharger le `.pt`** |
| **Dashboard Local** | Visualisation & Inférence | **PC Local** (`http://localhost:8765`) | Leaderboard, Comparateur Radar, Playground & AI Advisor |

---

## ⚡ Étape 1 : Dans votre Notebook Colab (Entraînement)

À la fin de votre boucle d'entraînement dans `NB03`, `NB04` ou tout script Colab, appelez simplement la fonction utilitaire du projet :

```python
# 1. Au début du notebook Colab (une seule fois) :
from google.colab import drive
drive.mount('/content/drive')

# 2. Après l'entraînement de votre modèle :
from colab_sync import save_colab_run

save_colab_run(
    run_id="transformer-sprint2-real",
    model=model,            # Votre modèle PyTorch entraîné
    history=history,        # Liste des métriques par époque
    config=config,          # Dictionnaire des hyperparamètres
    sync_drive=True,        # Copie automatique de best.pt dans Google Drive
    push_git=True,          # Git commit & push automatique des métriques
    n_samples=5             # Génère 5 séquences de test sur le GPU Colab
)
```

### Que fait cette fonction automatiquement ?
1. Sauvegarde `config.json` et `history.csv` dans `results/runs/<run_id>/`.
2. Génère 5 séquences protéiques réelles sur le GPU T4 (en 0.2s) et les stocke dans `metrics.json`.
3. Copie le fichier `best.pt` dans `/content/drive/MyDrive/recherche_ai_models/<run_id>/best.pt`.
4. Effectue un `git add` et `git push` pour envoyer **uniquement les métriques légères** sur votre dépôt GitHub.

---

## 🖥️ Étape 2 : Sur votre PC Local (Visualisation & Test)

Dès que Colab a fini :

### Option A : Via Git (Le plus rapide)
Ouvrez votre terminal local :
```bash
git pull
.\.venv\Scripts\python.exe scripts/export_dashboard_data.py
```
👉 Ouvrez votre dashboard : **[http://localhost:8765](http://localhost:8765)** et cliquez sur **Refresh**.  
Le nouveau modèle apparaît immédiatement dans le Leaderboard, les courbes de perte sont tracées, et ses séquences générées sont disponibles dans le Playground !

---

### Option B : Via le bouton "📥 Import Run" (Sans toucher à Git)
Si vous n'avez pas configuré de token Git dans Colab :
1. Dans Colab, téléchargez l'archive en 1 clic :
   ```python
   from google.colab import files
   import shutil
   shutil.make_archive("mon_run", "zip", "results/runs/transformer-sprint2-real")
   files.download("mon_run.zip")
   ```
2. Ouvrez **BioLM Studio** dans votre navigateur : [http://localhost:8765](http://localhost:8765)
3. Cliquez sur le bouton violet **"📥 Import Run"** dans la barre de navigation.
4. Sélectionnez le fichier `.zip` ou les fichiers `metrics.json` et `history.csv`.
5. Le modèle est immédiatement intégré et analysé à l'écran !

---

## 🧬 Étape 3 : Tester le Modèle en Local

Vous avez deux façons de tester votre nouveau modèle sur votre PC :

### 1. Test Immédiat (Zéro téléchargement)
Les échantillons générés par le GPU Colab sont déjà dans les métriques. Dans l'onglet **Generation Playground**, sélectionnez votre modèle : ses séquences synthétiques, leur masse moléculaire ($kDa$), point isoélectrique ($pI$), hydrophobicité et composition en acides aminés sont calculés en temps réel !

### 2. Inférence PyTorch Locale Complète (Avec nouveaux prompts)
Si vous voulez que votre PC génère de nouvelles séquences avec n'importe quel préfixe (ex. `MVL...`, température sur-mesure) :
1. Téléchargez simplement le fichier `best.pt` depuis votre Google Drive (`MyDrive/recherche_ai_models/<run_id>/best.pt`).
2. Déposez-le dans le dossier local du projet :
   ```
   reacherche_ai/models/<run_id>/best.pt
   ```
3. Le serveur backend (`serve_dashboard.py`) détecte automatiquement le fichier et effectue l'inférence neuronale en direct sur votre CPU !
