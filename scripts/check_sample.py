import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import AminoAcidTokenizer

tok = AminoAcidTokenizer()
seqs = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
bad = []
for s in seqs:
    try:
        assert tok.decode(tok.encode(s)) == s
    except Exception as e:
        bad.append((s[:30], repr(e)))
print(f"{len(seqs) - len(bad)}/{len(seqs)} round-trips OK")
for b in bad[:10]:
    print("FAIL", b)
