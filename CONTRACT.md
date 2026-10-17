# Team Contract
- Input column: "sequence" (string, uppercase, no gaps/dashes)
- Alphabet: ^[ACDEFGHIKLMNPQRSTVWY]+$ (20 amino acids only)
- Length bounds: 100 <= length <= 140 (PROVISIONAL: confirm from the camelid VHH length histogram)
- Extra VHH filters: TO DECIDE (the professor said the alphabet filter alone is not enough)
- Tokenizer: V = 23; ids 0-19 = AA in order ACDEFGHIKLMNPQRSTVWY; PAD=20, BOS=21, EOS=22
- Split: exact deduplication first, then random 90/5/5, seed = 42, strictly disjoint
- Files (CSV, column 'sequence'): vhh_clean.csv, train.csv, val.csv, test.csv
- Each processed/vN folder on Drive contains these 4 files + manifest.json (never edited after announcement)
