# Task brief: VHH Generation Repo (Simplified Workflow)

This repository provides a clean, reproducible workflow for generating VHH (nanobody) protein sequences with autoregressive language models (LSTM + Causal Transformer).

## 1. Target Workflow

```
Laptop (write code, test on toy data)
   │
   ▼
GitHub (code and notebooks only, no large data/weights)
   │
   ▼
Colab (download raw OAS via wget, heavy training/sampling on GPU)
   │
   ▼
Google Drive (small clean outputs + best model checkpoints only)
```

## 2. Architecture & Rules

1. **Self-Contained Notebooks (`notebooks/NB01..NB06`)**:
   - Real, executable Python code inside the cells following the professor's syllabus and cell numbering.
   - **First cell of each notebook** defines all settings (`SEED = 42`, batch size, sequence lengths, learning rates).
2. **Shared Utilities (`utils.py`)**:
   - Single clean file in project root for functions shared across notebooks (character-level tokenizer for the 20 amino acids, PyTorch `SequenceDataset`, `collate_fn`, device helper, metrics).
3. **Data Splitting**:
   - Random **90% train / 5% val / 5% test** split after exact sequence deduplication (`seed = 42`), strictly disjoint.
   - Cluster-aware split (via MMseqs2/CD-HIT) is an optional later extension.
4. **Clean Contract (`CONTRACT.md`)**:
   - Shrunk to 10 lines defining primary column `"sequence"`, alphabet `^[ACDEFGHIKLMNPQRSTVWY]+$`, lengths `100-140`, and filenames.
5. **No Heavy Bloat**:
   - No complex `src/` package nesting.
   - No external YAML configs (settings stay in notebook Cell 1).
   - No MLflow tracking overhead (`src/tracking.py` deleted).
   - No `data_manifest/` or CI overhead.
   - `.gitignore` prevents any `.csv`, `.fasta`, `.pt`, or `data/` from being committed to Git.
