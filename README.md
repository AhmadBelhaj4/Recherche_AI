# Recherche_AI: VHH Protein Sequence Generation

Autoregressive Language Models (LSTM + Causal Transformer) for generating novel VHH (nanobody) protein sequences from the Observed Antibody Space (OAS) dataset.

---

## 1. Target Workflow

```text
Laptop (write code, test on toy data)
   │
   ▼
GitHub (code, notebooks, scripts only)
   │
   ▼
Google Colab (download raw OAS, heavy GPU training & sampling)
   │
   ▼
Google Drive (small clean outputs + best model checkpoints only)
```

---

## 2. Repository Structure

```text
Recherche_AI/
├── notebooks/
│   ├── NB01_exploration_donnees.ipynb          # Phase A: Data cleaning & exploration (Cells 1-7)
│   ├── NB02_tokenization_datasets_splits.ipynb     # Phase B: Tokenizer, PyTorch Dataset & 90/5/5 split
│   ├── NB03_lstm_baseline.ipynb                # Phase C&D: LSTM Baseline, AdamW training loop, nats/token
│   ├── NB04_transformer_causal.ipynb           # Phase C&D: Causal Transformer decoder-only (Pre-LN)
│   ├── NB05_generation_sampling.ipynb          # Phase E: Sampling (Greedy, Top-k, Nucleus) & FASTA export
│   └── NB06_evaluation_metriques.ipynb         # Phase F: Validity, Levenshtein, Jaccard, KL divergence
├── scripts/
│   ├── make_toy_data.py                        # Generates 200 fake sequences for 1-second laptop tests
│   └── clean_data.py                           # Cleans raw OAS data and creates 90/5/5 splits
├── CONTRACT.md                                 # 10-line data contract
├── utils.py                                    # Single shared helper (Tokenizer, Dataset, Metrics, Seeds)
├── requirements.txt                            # Clean, minimal dependencies (torch, pandas, numpy, etc.)
├── .gitignore                                  # Ignores data/, *.csv, *.pt, checkpoints/
├── ANTIGRAVITY_TASK.md                         # Agent task brief
└── EXPLAINED.md                                # Beginner-friendly workflow explanation
```

---

## 3. Quickstart

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate toy data (test locally in 1 second)
```bash
python scripts/make_toy_data.py
```
This generates 200 toy sequences in `data/toy/` and splits them into `train.csv`, `val.csv`, and `test.csv`.

### Step 3: Run any notebook
Open any notebook inside `notebooks/` in VS Code or Jupyter. All settings (`SEED = 42`, batch size, lengths) are located in **Cell 1** of each notebook.
