# Team Contract
- Input column: "sequence" (string, uppercase, no gaps/dashes)
- Allowed alphabet: ^[ACDEFGHIKLMNPQRSTVWY]+$ (20 amino acids only)
- Length bounds: 100 <= length <= 140
- Split strategy: Random 90% train, 5% val, 5% test (seed = 42)
- Deliverable files (CSV, column 'sequence'):
  1. vhh_clean.csv (all unique sequences)
  2. train.csv (90% training split)
  3. val.csv (5% validation split)
  4. test.csv (5% test split)
