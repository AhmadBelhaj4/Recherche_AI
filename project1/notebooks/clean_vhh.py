import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no display needed; saves the plot as a file
import matplotlib.pyplot as plt
from pathlib import Path

# ============ 1. EDIT ONLY THIS PART ============
BASE_DIR = Path(r"C:\Users\USER\Desktop\Datasets\CSV")

FILES = {
    "camel1_a": BASE_DIR / "SRR3544217_1_Heavy_Missing_c_domain.csv",
    "camel1_b": BASE_DIR / "SRR3544218_1_Heavy_Missing_c_domain.csv",
    "camel2_a": BASE_DIR / "SRR3544219_1_Heavy_Missing_c_domain.csv",
    "camel2_b": BASE_DIR / "SRR3544220_1_Heavy_Missing_c_domain.csv",
    "camel3_a": BASE_DIR / "SRR3544221_1_Heavy_Missing_c_domain.csv",
    "camel3_b": BASE_DIR / "SRR3544222_1_Heavy_Missing_c_domain.csv",
}

MIN_LEN, MAX_LEN = 100, 140     # change after looking at the histogram
OUT_DIR = BASE_DIR.parent / "Output"  # saves next to your CSV folder
# =================================================

OUT_DIR.mkdir(parents=True, exist_ok=True)

COL = "sequence_alignment_aa"
OK = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
report = []

def log(step, n):
    before = report[-1][2] if report else n
    report.append((step, before, n))
    print(f"{step:<32} rows: {n:,}")

# --- Step 1: load the 6 files and stack them ---
frames = []
for name, path in FILES.items():
    d = pd.read_csv(path, low_memory=False)
    if COL not in d.columns:            # OAS files sometimes have a metadata first line
        d = pd.read_csv(path, skiprows=1, low_memory=False)
    d = d[[COL]].copy()
    d["source_file"] = name
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
log("1. raw (6 files)", len(df))

# --- Step 2: remove gaps "-" and clean the text ---
df = df.dropna(subset=[COL])
df["sequence"] = (df[COL].astype(str).str.replace("-", "", regex=False)
                  .str.upper().str.strip())
df = df[df["sequence"] != ""]
log("2. gaps removed, no empty", len(df))

# --- Step 3: keep only the 20 amino acids ---
bad = df[~df["sequence"].apply(lambda s: bool(OK.match(s)))]
print("   examples rejected:", bad["sequence"].head(3).tolist())
df = df[df["sequence"].apply(lambda s: bool(OK.match(s)))]
log("3. only 20 letters", len(df))

# --- Step 4: look at lengths, then keep the window ---
lengths = df["sequence"].str.len()
print("   length min / median / max:", lengths.min(), int(lengths.median()), lengths.max())

plt.figure(figsize=(10, 4))
lengths.hist(bins=60, color="steelblue", edgecolor="white")
plt.axvline(MIN_LEN, color="red",    linestyle="--", label=f"MIN={MIN_LEN}")
plt.axvline(MAX_LEN, color="orange", linestyle="--", label=f"MAX={MAX_LEN}")
plt.xlabel("Sequence length"); plt.ylabel("Count")
plt.title("VHH sequence length distribution")
plt.legend(); plt.tight_layout()
hist_path = OUT_DIR / "length_histogram.png"
plt.savefig(hist_path, dpi=150)
print(f"   Histogram saved to: {hist_path}")

df = df[lengths.between(MIN_LEN, MAX_LEN)]
log(f"4. length {MIN_LEN}-{MAX_LEN}", len(df))

# --- Step 5: remove exact duplicates ---
df = df.drop_duplicates(subset="sequence")
log("5. no duplicates", len(df))

# --- Step 6: save ---
df[["sequence"]].to_csv(OUT_DIR / "vhh_clean.csv", index=False)
df[["sequence", "source_file"]].to_csv(OUT_DIR / "vhh_clean_meta.csv", index=False)
print(f"\nOutput saved to: {OUT_DIR}")

# --- Final report table ---
rep = pd.DataFrame(report, columns=["step", "rows_before", "rows_after"])
rep["removed"] = rep["rows_before"] - rep["rows_after"]
print(rep.to_string(index=False))
print("Per file kept:\n", df["source_file"].value_counts())
