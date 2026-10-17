# Recherche_AI

Research project (Prof. Abdallah): controlled generation of peptides / VHH nanobodies
by discrete diffusion and Schrödinger bridges. Team of 4 (Team A: data, Team B: models).

## Where things are

| Place | Holds | Never holds |
|---|---|---|
| GitHub (this repo) | code, notebooks (outputs cleared), SOURCES.md | data, model files |
| Google Drive `AI_Project_Shared/` | `raw/`, `processed/vN/`, `models/<project>/`, `results/<project>/` | code |
| Colab | heavy jobs (download, clean, train, generate) | anything to keep |

## Repo layout

- `utils.py`, `scripts/`, `tests/`: shared by all projects
- `project1/`: autoregressive baseline (NB01 to NB06)
- `docs/`: professor's plan
- `CONTRACT.md`: Team A / Team B agreement. `SOURCES.md`: where the data came from.

## Local setup

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    nbstripout --install
    python -m pytest -v

## Colab (first cell of every heavy notebook)

    from google.colab import drive
    drive.mount('/content/drive')
    import os
    if not os.path.exists('/content/Recherche_AI'):
        !git clone https://github.com/AhmadBelhaj4/Recherche_AI.git
    %cd /content/Recherche_AI
    !git pull
    !pip install -q -r requirements.txt

Colab only pulls. Code is edited locally.

## Cell 1 of every notebook (finds utils.py from any folder)

    import sys
    from pathlib import Path
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "utils.py").exists())
    sys.path.insert(0, str(ROOT))
    from utils import *

## Rules

1. Nobody pushes to `main`: branch, Pull Request, one review.
2. Nothing over 5 MB in Git. No data, no checkpoints.
3. Never overwrite a Drive file: create `processed/v2`, never edit `v1`.
4. Every `processed/vN` has a `manifest.json` (made by `scripts/clean_data.py`).
5. Clear notebook outputs before committing.
