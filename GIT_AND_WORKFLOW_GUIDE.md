# 💡 Simple Guide: How Our Project Works

If the other documents were too long or confusing, **read this page**.  
Everything is explained in plain, normal words.

---

## 1. What are we doing? (In 30 seconds)

1. We have a list of **camel antibody proteins** (called VHH or Nanobodies).
2. We clean this list to keep only good sequences.
3. We teach an AI model (like a mini-ChatGPT for biology) how these sequences are written.
4. The AI generates **10,000 brand new sequences** that could be used as future medicines.

---

## 2. Where does the work happen?

Do not do heavy work on your personal laptop. Here is the split:

| Where? | What do you do there? | Why? |
|---|---|---|
| **Your Laptop** | Write code in VS Code. | Fast and comfortable to type. |
| **GitHub** | Save and share your code. | So your teammates have the exact same files. |
| **Google Colab** | Run the heavy code (training the AI). | Colab gives you free cloud GPUs so your laptop doesn't overheat. |
| **Google Drive** | Store only the final clean data and saved AI models. | Because Google Drive is small (15 GB), we only put finished files there. |

---

## 3. What is each file in our folder?

Here is every file in your folder and its job in **one simple sentence**:

* **`notebooks/`** (This is where the real work happens):
  * **`NB01_exploration_donnees.ipynb`**: Loads raw antibody files, checks their lengths, and plots graphs.
  * **`NB02_tokenization_datasets_splits.ipynb`**: Turns letters (`A, C, D...`) into numbers and splits data into Train (90%), Val (5%), and Test (5%).
  * **`NB03_lstm_baseline.ipynb`**: Builds and trains our first AI model (a simple LSTM).
  * **`NB04_transformer_causal.ipynb`**: Builds and trains our second AI model (a smarter Transformer).
  * **`NB05_generation_sampling.ipynb`**: Uses the trained model to generate 10,000 new protein sequences.
  * **`NB06_evaluation_metriques.ipynb`**: Checks if the generated sequences are realistic and creates the final comparison table.

* **`scripts/`**:
  * **`make_toy_data.py`**: Creates 200 fake sequences in 1 second so you can test code on your laptop without downloading gigabytes.
  * **`clean_data.py`**: A helper script to clean real data files and create the 90/5/5 splits.

* **`utils.py`**:
  * A single helper toolbox file. It holds common functions (like turning letters into numbers and setting random seeds) so we don't copy-paste the same code into every notebook.

* **`CONTRACT.md`**:
  * A 10-line text file listing the rules: sequence lengths must be 100 to 140, only the 20 normal amino acids, and the column must be named `"sequence"`.

* **`requirements.txt`**:
  * The list of Python libraries needed (`torch`, `pandas`, `numpy`, `matplotlib`).

---

## 4. Git explained in 3 words

* **Commit:** A save point in your project (like saving a game).
* **Branch (`b/s0-scaffold`):** A duplicate copy of the project where you can work freely without touching or breaking the `main` code.
* **Pull Request (PR):** When your work on your branch is done, you show it to your teammates on GitHub and ask: *"Can we merge this into main?"*

---

## 5. Why are settings in Cell 1 of each notebook?

Instead of hiding settings in separate configuration files, **every notebook has its settings right at the very top (Cell 1)**:

```python
SEED = 42          # Makes sure random numbers are identical for everyone
BATCH_SIZE = 64    # How many sequences the AI reads at once
MIN_LEN = 100      # Shortest allowed sequence
MAX_LEN = 140      # Longest allowed sequence
```

If you want to test a different batch size or length, you just edit that cell. Nothing is hidden.

---

## 6. What should you do right now?

### Step 1: Test locally on your laptop (Takes 10 seconds)
Open your terminal in VS Code and run:
```bash
python scripts/make_toy_data.py
```
This creates 200 fake sequences in `data/toy/`. Now you can open any notebook in VS Code and run cells to see how they work.

### Step 2: Running real data on Google Colab
When you want to train on real data:
1. Open Google Colab in your browser.
2. Open the notebook (for example, `NB03` or `NB04`).
3. Turn on GPU (`Runtime` -> `Change runtime type` -> `T4 GPU`).
4. Click `Run all cells`. Colab does all the heavy math for you.
