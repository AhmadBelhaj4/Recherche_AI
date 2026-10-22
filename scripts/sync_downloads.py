"""sync_downloads.py - Automatically import checkpoints from Downloads folder.

When Colab auto-downloads best.pt to your browser, running this script
instantly detects it and moves it into models/<run_id>/best.pt without
any manual folder navigation.

Usage:
    python scripts/sync_downloads.py
    python scripts/sync_downloads.py --run-id transformer-notebook-toy
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = Path.home() / "Downloads"


def sync_from_downloads(run_id: str | None = None) -> bool:
    if not DOWNLOADS.exists():
        print(f"Downloads folder {DOWNLOADS} not found.")
        return False

    # Search for candidate .pt files in Downloads (most recent first)
    pt_files = sorted(DOWNLOADS.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pt_files:
        print(f"No .pt files found in {DOWNLOADS}.")
        return False

    # Pick target run_id
    if not run_id:
        # Check available runs in results/runs
        runs = sorted((ROOT / "results" / "runs").iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            run_id = runs[0].name
            print(f"Auto-selected latest run: '{run_id}'")
        else:
            run_id = "colab-model"

    target_dir = ROOT / "models" / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / "best.pt"

    src = pt_files[0]
    shutil.copy(src, dest)
    print("=" * 60)
    print(f"  SUCCESS! Checkpoint imported automatically:")
    print(f"  Source : {src} ({src.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  Dest   : {dest}")
    print(f"  Model '{run_id}' is now ready for local PyTorch generation!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync downloaded Colab checkpoints")
    parser.add_argument("--run-id", type=str, default=None, help="Target run ID")
    args = parser.parse_args()
    sync_from_downloads(args.run_id)
