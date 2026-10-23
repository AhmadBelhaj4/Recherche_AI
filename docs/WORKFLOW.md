# WORKFLOW — How this repo works (simple guide)

Last updated: 2026-10-18. Update the **status table (section 10)** whenever something changes.

This guide explains:
1. what is in the repo and where the files live,
2. how the notebooks find their files (locally and on Colab),
3. the three ways to run the project,
4. the Git routine to follow,
5. what to check, and what to do when something breaks.

---

## 1. The big picture

The project trains a small language model on antibody sequences (Projet 1.1) and generates new sequences.
The pipeline is a chain of notebooks:

```
NB01 explore data -> NB02 tokenize + split -> NB03 LSTM -> NB04 Transformer -> NB05 generate -> NB06 evaluate
```

Shared code lives in two Python files, so notebooks do not repeat it:

| File | What it contains |
|---|---|
| `utils.py` | tokenizer (23 tokens), dataset, `make_loader`, `shift`, `get_paths`, metrics |
| `models.py` | the two models: `LSTMLM` and `CausalTransformerLM` |

---

## 2. Folder map

```
Recherche_AI/
├── utils.py                  shared helpers
├── models.py                 the two models
├── requirements.txt          Python packages
├── README.md                 quick start
├── scripts/
│   ├── make_toy_data.py      makes 200 FAKE sequences for testing
│   └── clean_data.py         cleans real OAS data (not yet linked to the notebooks)
├── tests/                    pytest tests (11 tests)
├── docs/                     plan and this guide
├── project1/notebooks/       NB01 ... NB06
│
├── data/        (NOT in GitHub)   real data, toy data, train/val/test files
├── models/      (NOT in GitHub)   saved checkpoints (.pt)
└── results/     (NOT in GitHub)   figures and generated FASTA files
```

`data/`, `models/` and `results/` are in `.gitignore`. They are created on your machine when you run things.

---

## 3. What is in GitHub and what is not

| Thing | In GitHub? | Why |
|---|---|---|
| Code, notebooks (without outputs), tests, docs | Yes | this is the project |
| Real OAS file `1287151_1_Paired_All.csv` (Sprint 1) | **No** | ~17.4k human paired sequences (too big to republish) |
| Real SRA file `SRR3544218` (Sprint 2/3) | **No** | 242,891 camelid VHH nanobodies (downloaded via script/drive) |
| `data/`, `models/`, `results/` | **No** | generated files |
| Toy data | **No** | recreated by `make_toy_data.py` |

**Consequence:** a fresh clone (for example on Colab) has **no data at all**.
Without the real file, the notebooks use toy data. Toy data is for testing only.

---

## 4. Dataset Clarification & Biological Context

> [!IMPORTANT]
> **Clarification on Datasets Used in the Project:**
> - **Dataset A — Sprint 1 (Historical Baseline)** : `data/raw/1287151_1_Paired_All.csv` from Observed Antibody Space (OAS). Contains ~17,400 paired **human conventional heavy chains (VH)**. In human VH, framework positions 37, 44, 45, 47 (Kabat) are hydrophobic ($V_{37}, G_{44}, L_{45}, W_{47}$) because they pack against the light chain ($V_L$).
> - **Dataset B — Sprint 2 & 3 (BioLM Studio Production Models)** : `data/processed/SRR3544218/` from NCBI SRA. Contains **242,891 single-domain camelid nanobodies (VHH)** (alpaca/llama). Camelid VHH possess hydrophilic hallmark mutations in FR2 ($F/Y_{37}, E_{44}, R_{45}, G_{47}$) allowing autonomous aqueous solubility without a light chain.
> - **Current Models & Dashboard** : All production neural models (`transformer-notebook-srr`, `lstm-notebook-srr`, `bigram_SRR3544218_a1`) are trained strictly on **Dataset B (SRR3544218, 242k camelid VHH)**. Biological claims regarding FR2 solubility hallmarks apply strictly to these camelid models.

---

## 5. Golden rules

1. **Never trust a number from toy data.** 180 / 10 / 10 sequences = toy data. The real camelid VHH corpus has 242,891 sequences (split ~194k train / 24k val / 24k test).
2. **Never commit notebook outputs.** Outputs contain your Windows paths, logs and huge images.
3. **Never use "Save to GitHub" from Colab.** It commits outputs and rewrites the file.
4. **Edit on your PC, then push, then test on Colab.** Colab clones what is on GitHub, not what is on your PC.
5. **Always test from a fresh Colab runtime** (Runtime -> Disconnect and delete runtime, then Run all).
6. **Do not work on `main`.** Use a branch, check it, then merge.
7. **Do not change the science silently** (loss, hyperparameters, sampling). Those changes get their own branch.

---

## 5. How the notebooks find their files (the "bootstrap" cell)

Every notebook starts with a **bootstrap cell**. It does three things:

1. Looks for `utils.py`, starting from the current folder and going up.
   - On your PC this finds it immediately.
2. If it is not found and the folder `/content` exists (= we are on Colab), it runs `git clone` of the repo into `/content/Recherche_AI` and searches there.
3. Calls `get_paths()` and stores the result in `PATHS`.

`PATHS` is a dictionary:

| Key | Folder |
|---|---|
| `PATHS["raw"]` | `data/raw` (put the OAS file here) |
| `PATHS["processed"]` | `data/processed` |
| `PATHS["models"]` | `models` |
| `PATHS["results"]` | `results` |

By default they are inside the repo. If you set the environment variable `AI_PROJECT_DIR`, they go there instead (useful for Google Drive).

**The branch setting:** the bootstrap has `BRANCH = "main"`. On Colab it clones that branch.
Only change it to another branch name for a temporary test, and set it back to `"main"` before committing.

---

## 6. The three ways to run the project

### A. Local on your PC (daily development)

```powershell
cd C:\Users\USER\Desktop\Recherche_AI
pip install -r requirements.txt
pytest -q                          # expect: 11 passed
python scripts/make_toy_data.py    # optional: creates data/toy/
```

Then open a notebook and use **Restart & Run All**.

Where the real file goes:

```
data/raw/1287151_1_Paired_All.csv
```

### B. Colab with toy data (smoke test: "does the code run anywhere?")

1. Open the notebook from GitHub in Colab.
2. Runtime -> Run all. Nothing else to do.

What happens: the bootstrap clones the repo, the notebook sees no data, and runs `make_toy_data.py` itself.
**This only proves the code runs. The numbers mean nothing.**

### C. Colab with the real data (real runs)

The real file is only on your PC, so Colab must receive it. Two options.

**Option 1 - upload each session (simple, but repeated).**
After the bootstrap cell has run, upload the file into `/content/Recherche_AI/data/raw/` using the Colab file browser.

**Option 2 - Google Drive (recommended).**
1. On your Drive, create `Recherche_AI_run/data/raw/` and put the OAS file there once.
2. In the very first cell of the notebook, **before** the bootstrap cell, run:

```python
from google.colab import drive
drive.mount("/content/drive")
import os
os.environ["AI_PROJECT_DIR"] = "/content/drive/MyDrive/Recherche_AI_run"
```

Now `data/`, `models/` and `results/` all live on Drive. Checkpoints survive a runtime reset, so NB05 can use the model trained in NB04 even in a new session.

**Warning (not tested yet):** the toy-data script writes into the cloned repo, but with `AI_PROJECT_DIR` set the notebooks look for toy data on Drive. So with `AI_PROJECT_DIR` set and **no real data**, the toy fallback will probably fail. Put the real data on Drive first.

---

## 7. Example: a full day of work

```
1. PC:    git checkout -b b/my-change            (new branch)
2. PC:    edit code or a notebook
3. PC:    Restart & Run All, check results
4. PC:    pytest -q
5. PC:    nbstripout <notebook>                  (remove outputs)
6. PC:    git diff -w --stat                     (small diff?)
7. PC:    git add <files>; git commit; git push
8. Colab: fresh runtime, open from GitHub (branch), Run all
9. If OK: merge into main
10. Colab: fresh runtime, open from main, Run all   (final proof)
```

Step 8 for a branch only works if the bootstrap `BRANCH` points to that branch, because Colab clones the branch named there.

---

## 8. Git routine (PowerShell)

### Start work
```powershell
git checkout main
git pull
git checkout -b b/short-name
```

### Check before every commit
```powershell
git status --short
git diff -w --stat
```
- A notebook diff should be a **few dozen lines**. Hundreds of lines = outputs are inside. Run `nbstripout` on it.
- Do not use `git add .` or `git commit -a`. Add files one by one.

### Strip outputs
```powershell
nbstripout project1/notebooks/NB02_tokenization_datasets_splits.ipynb
```
One-time setup so it happens automatically at commit:
```powershell
nbstripout --install
```

### Validate a notebook
```powershell
python -c "import nbformat;nbformat.validate(nbformat.read('project1/notebooks/NB02_tokenization_datasets_splits.ipynb',4));print('valid')"
```

### Commit and push
```powershell
git add project1/notebooks/NB02_tokenization_datasets_splits.ipynb
git commit -m "NB02: short clear message"
git push
```
The first push of a new branch: `git push -u origin b/short-name`.

### Merge (only after all checks pass)
```powershell
git status --short          # must print nothing
pytest -q                   # all pass
git checkout main
git merge --no-ff b/short-name
git push origin main
```
Keep the branch until the final Colab test from `main` passes.

### Undo uncommitted output pollution on a notebook
```powershell
git restore project1/notebooks/NB03_lstm_baseline.ipynb
```
**Look first** with `git diff -w <file>`. If the diff contains real code you wrote, do not restore it.

---

## 9. How to read what a notebook prints

| You see | It means |
|---|---|
| `Train : 180 ... Val : 10 ... Test : 10` | toy data. Not a result. |
| `Source : ...\data\toy\train.csv` (NB03) | training on toy data |
| `Source : ...\data\train.csv` | training on the splits written by NB02 |
| Val nats about 3.05 on toy data | normal: random sequences cannot be predicted better than about ln(20) = 3.0 |
| NB04 epoch 1 at about 47 nats | known initialization problem (section 10), not a Colab bug |
| NB05 "Modèle non pré-entraîné" | no checkpoint found, generating with random weights (demo mode) |

---

## 10. Status (update me)

| Notebook | Local | Colab | Bootstrap fixed |
|---|---|---|---|
| NB01 | runs with the real OAS file | needs the OAS file uploaded | **no** (old version) |
| NB02 | OK | OK (verified) | yes |
| NB03 | OK | not yet tested | yes |
| NB04 | OK | not tested | **no** |
| NB05 | OK | not tested | **no** |
| NB06 | not ported | not ported | **no** |

### Known gaps

1. **Real data does not reach NB02.** NB01 writes `data/heavy_unique.txt`. NB02 reads `data/processed/vhh_clean.csv`. Nothing connects them, so NB02 may silently use toy data. Check the sizes it prints.
2. **`clean_data.py` expects the column `sequence_alignment_aa`.** The paired OAS file has `sequence_alignment_aa_heavy`, so the script fails on it as is.
3. **Two writers of the splits.** NB02 writes `data/train.csv`; `clean_data.py` writes into its output folder. Running both can give different splits. Choose one source.
4. **The data is human VH (paired OAS, 17.5k), not VHH.** The professor should know this is a deliberate substitute.
5. **Transformer initialization.** The tied embedding starts too large, so epoch 1 shows about 47 nats instead of about 3.1.
6. **Not yet implemented from the plan:** cosine scheduler (task 3.1), marginal-composition baseline (task 2.3), exact nucleus condition `(cum - sp) < top_p`, no `max(temperature, 1e-5)` epsilon, masking PAD/BOS during sampling, per-batch seeds (task 4.2).
7. **Numbers written in NB01's report text** (S about 10.5 %, G about 8.8 %, W about 1.5 %, length stats) must be compared with the printed output before the report is sent. Only the redundancy numbers were checked.

---

## 11. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: Could not locate utils.py` | notebook is not inside the repo and the clone did not happen or failed | on Colab check the clone output and `BRANCH`; locally open the notebook inside the repo |
| `FileNotFoundError: .../data/toy/vhh_clean.csv` | fresh clone has no toy data | the fallback cell should create it; otherwise run `python scripts/make_toy_data.py` |
| `ModuleNotFoundError: torch` | wrong Python environment | `pip install -r requirements.txt`, select the right kernel |
| `ModuleNotFoundError: logomaker` | NB01 dependency | `pip install logomaker requests` |
| `NameError: PROJECT_ROOT` | old NB01 variable name | use `ROOT_DIR` |
| `SyntaxError` on an f-string with `{PATHS["x"]}` | double quotes inside double quotes on Python < 3.12 | write `{PATHS['x']}` |
| `TypeError: collate_fn() got an unexpected keyword 'pad_id'` | old loader code | use `make_loader(...)` and `shift(batch)` |
| Notebook diff has hundreds of lines | outputs are inside | `nbstripout <file>` |
| `git commit` says "no changes added" | you did not `git add` the file, or nothing changed | check `git status --short` |
| Colab cloned old code | you did not push, or `BRANCH` is wrong | `git push`, check `BRANCH` |

---

## 12. Glossary

- **Branch**: a separate line of work; `main` stays safe until you merge.
- **Bootstrap cell**: first cell of a notebook; finds the repo and sets `PATHS`.
- **Toy data**: 200 fake sequences used only to test that code runs.
- **nbstripout**: tool that removes outputs from notebooks.
- **Checkpoint**: saved model weights (`models/*.pt`).
- **nats/token**: the loss per token, in natural-log units. Lower is better. A model that guesses uniformly over 20 amino acids scores ln(20), about 3.0.
- **Fresh runtime**: a Colab session with nothing left over from before.
