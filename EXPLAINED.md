# 📖 The Simplified Beginner's Guide: Our Workflow & Architecture

> **Welcome to the simplified setup!**  
> We have stripped away all unnecessary complexity (no complex `src/` packages, no MLflow, no external YAML configs, no CI robots).  
> The project is now 100% focused on simplicity, readability, and following the professor's plan.

---

# Table of Contents
1. [The Big Picture (The 4 Kitchens)](#1-the-big-picture-the-4-kitchens)
2. [Key Architecture Decisions](#2-key-architecture-decisions)
3. [File Map of the Simplified Repository](#3-file-map-of-the-simplified-repository)
4. [Dictionary of Key Terms (Plain English)](#4-dictionary-of-key-terms-plain-english)
5. [How to Run the Project (Laptop & Colab)](#5-how-to-run-the-project-laptop--colab)

---

# 1. The Big Picture (The 4 Kitchens)

```text
+-------------------------+
|       YOUR LAPTOP       |  Write code in VS Code.
|    (The Writing Desk)   |  Test instantly on 200 toy sequences (python scripts/make_toy_data.py).
+-------------------------+
             |
             v
+-------------------------+
|         GITHUB          |  Stores clean code, notebooks, and scripts only.
|    (The Recipe Book)    |  Never stores heavy CSVs or model weights.
+-------------------------+
             |
             v
+-------------------------+
|      GOOGLE COLAB       |  Downloads raw OAS files directly using wget.
|     (The Heavy Oven)    |  Runs heavy training and sampling on free GPUs.
+-------------------------+
             |
             v
+-------------------------+
|      GOOGLE DRIVE       |  Stores ONLY final clean outputs (train.csv, val.csv, test.csv)
|       (The Fridge)      |  and the best trained model checkpoints (best.pt).
+-------------------------+
```

---

# 2. Key Architecture Decisions

1. **Real Code Inside Notebooks (`notebooks/NB01..NB06`)**:
   - No confusing 3-cell wrappers. Each notebook contains full, runnable code with clear Markdown explanations following the professor's cell numbers.
   - **Cell 1 in every notebook contains all settings:** `SEED = 42`, batch size, embedding dimension, sequence lengths, and learning rates. You never need to look for hidden configuration files.

2. **One Shared Helper: `utils.py`**:
   - Instead of a multi-folder `src/` package, all shared code is centralized in one simple file in the root directory: `utils.py`.
   - It provides:
     - `AminoAcidTokenizer`: Tokenizes the 20 biological amino acids (`A, C, D...`) plus `<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`.
     - `SequenceDataset` & `collate_fn`: PyTorch dataset and padding batcher.
     - `set_seed(42)` & `get_device()`: Deterministic reproducibility and automatic CUDA GPU detection.
     - Metrics: `compute_nats_and_perplexity`, `is_valid_sequence`, and amino-acid composition.

3. **Data Splitting: Random 90% / 5% / 5% (Professor's Plan)**:
   - Data is cleaned (gaps removed, 20 amino acids only, length 100–140, exact duplicates removed).
   - Then randomly split into **90% train, 5% validation, and 5% test** using fixed `seed = 42`.
   - All three splits are verified to be strictly disjoint (no sequence appears in two sets).
   - Cluster-aware splitting (via CD-HIT/MMseqs2) is kept as an optional later extension.

4. **Minimal Team Contract (`CONTRACT.md`)**:
   - Shrunk to 10 lines defining the exact interface between Team A (Data) and Team B (Models).

---

# 3. File Map of the Simplified Repository

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
├── README.md                                   # Project overview and run instructions
├── ANTIGRAVITY_TASK.md                         # Agent task brief
├── GIT_AND_WORKFLOW_GUIDE.md                   # In-depth guide to Git, team workflow & codebase
└── EXPLAINED.md                                # This beginner guide
```

---

# 4. Dictionary of Key Terms (Plain English)

* **VHH / Nanobodies:** Mini-antibodies found in camels and llamas. Compact, stable, and powerful for medicine.
* **OAS (Observed Antibody Space):** The large public repository containing millions of real antibody sequencing reads.
* **Amino Acids (20 Letters):** The biological alphabet: `A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y`.
* **Gaps (`-`):** Alignment placeholders inserted by databases that must be erased before training an AI.
* **Tokenizer (`utils.py`):** Translates biological characters into numbers so the neural network can process them.
* **Autoregressive Language Model (LSTM / Transformer):** A model that predicts the next amino acid one token at a time from left to right.
* **Weight Tying (`head.weight = embed.weight`):** Sharing weights between the input embedding matrix and output linear head to save memory and improve training stability.
* **nats/token:** The natural log loss per real (non-PAD) amino acid token, measuring prediction accuracy.
* **Perplexity (`exp(nats/token)`):** The effective number of choices the model is hesitating between at each position. Lower is better.
* **Sampling Strategies:**
  * **Greedy:** Picks the highest-probability token every step (deterministic).
  * **Top-k:** Samples only among the top $k$ most likely amino acids.
  * **Nucleus (Top-p):** Dynamically cuts off the tail of low-probability amino acids once cumulative mass reaches $p$.
* **Toy Data (`scripts/make_toy_data.py`):** 200 synthetic sequences generated in 1 second so you can test code on your laptop without downloading 8 GB files.

---

# 5. How to Run the Project (Laptop & Colab)

### On Your Laptop (Fast Local Verification):
1. Generate the toy data (takes 1 second):
   ```bash
   python scripts/make_toy_data.py
   ```
2. Open any notebook in VS Code (`notebooks/NB01` to `NB06`).
3. Run the cells. They automatically detect the local toy data and run on your CPU without freezing your machine.

### On Google Colab (Full Training & Heavy Runs):
1. Upload the repository or clone it on Colab.
2. Open `NB01` to download and clean the OAS data, or run `scripts/clean_data.py`.
3. Open `NB03` (LSTM) or `NB04` (Transformer), select GPU runtime, and run all cells.
4. Open `NB05` to generate 10,000 new nanobody sequences.
5. Open `NB06` to evaluate the generated sequences and produce the final comparison report table.
