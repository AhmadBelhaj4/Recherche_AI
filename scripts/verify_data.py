"""Verify data integrity by comparing file sha256 hashes against a manifest.

Usage
-----
    # Check toy data (default manifest: data/toy/splits/manifest.json)
    python scripts/verify_data.py --data-dir toy

    # Check real data (default manifest: data_manifest/manifest.json)
    python scripts/verify_data.py --data-dir real

    # Override manifest path
    python scripts/verify_data.py --data-dir toy --manifest path/to/manifest.json

Exit codes
----------
0   All hashes match — prints ``data OK: <version>``
1   At least one hash mismatch — prints ``DATA MISMATCH`` and details
2   A file listed in the manifest does not exist
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()

# Default manifest paths per data-dir value
DEFAULT_MANIFESTS = {
    "toy": ROOT / "data" / "toy" / "splits" / "manifest.json",
    "real": ROOT / "data_manifest" / "manifest.json",
}

# Default data dirs where the split files reside
DEFAULT_DATA_DIRS = {
    "toy": ROOT / "data" / "toy" / "splits",
    "real": ROOT / "data" / "real",  # overridden by VHH_DATA if set
}


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(manifest_path: Path, data_dir: Path) -> int:
    """Verify all files listed in the manifest.

    Returns 0 on success, 1 on mismatch, 2 on missing file.
    """
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("data_version", "unknown")
    checksums: dict[str, str] = manifest.get("sha256", {})

    all_ok = True
    for fname, expected_hash in checksums.items():
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"MISSING: {fpath}", file=sys.stderr)
            return 2
        actual_hash = sha256_file(fpath)
        if actual_hash != expected_hash:
            print(
                f"DATA MISMATCH: {fname}\n"
                f"  expected: {expected_hash}\n"
                f"  got:      {actual_hash}",
                file=sys.stderr,
            )
            all_ok = False

    if all_ok:
        print(f"data OK: {version}")
        return 0
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify data integrity against a sha256 manifest."
    )
    parser.add_argument(
        "--data-dir",
        choices=["toy", "real"],
        default="real",
        help=(
            '"toy" checks data/toy/splits/ against data/toy/splits/manifest.json; '
            '"real" checks against data_manifest/manifest.json.'
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override the default manifest path.",
    )
    args = parser.parse_args()

    manifest_path: Path = args.manifest or DEFAULT_MANIFESTS[args.data_dir]
    data_dir: Path = DEFAULT_DATA_DIRS[args.data_dir]

    sys.exit(verify(manifest_path, data_dir))


if __name__ == "__main__":
    main()
