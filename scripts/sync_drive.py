r"""sync_drive.py - Continuous Google Drive Desktop Sync Utility for AI_Project_Shared.

Automatically detects the Google Drive 'AI_Project_Shared' folder on Windows
(including virtual shortcut targets on G:\) and synchronizes trained checkpoints (.pt),
evaluation metrics (history.csv), and baselines into the local project.

Usage:
    python scripts/sync_drive.py
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent


def find_ai_project_shared() -> Path | None:
    """Detect AI_Project_Shared folder on Windows Google Drive."""
    # 1. Check G: drive and other drive letters
    for letter in "GHIJKLMNOPQRSTUVWXYZFED":
        try:
            drive_root = Path(f"{letter}:/")
            if not drive_root.exists():
                continue

            # Check inside .shortcut-targets-by-id (Google Drive virtual shortcut targets)
            shortcut_dir = drive_root / ".shortcut-targets-by-id"
            if shortcut_dir.exists():
                for target_id in shortcut_dir.iterdir():
                    cand = target_id / "AI_Project_Shared"
                    if cand.exists():
                        return cand

            # Check direct subfolders in Drive root
            for sub in ["Mon Drive", "My Drive", ""]:
                cand = drive_root / sub / "AI_Project_Shared"
                if cand.exists() and cand.is_dir():
                    return cand

        except (PermissionError, OSError):
            continue

    # 2. Check User Home folders
    for sub in ["Google Drive", "GoogleDrive", "Drive", "OneDrive"]:
        try:
            cand = Path.home() / sub / "AI_Project_Shared"
            if cand.exists() and cand.is_dir():
                return cand
        except (PermissionError, OSError):
            continue

    return None


def sync_drive():
    print("=" * 65)
    print("  BioLM Studio — Google Drive Sync (AI_Project_Shared)")
    print("=" * 65)

    shared_dir = find_ai_project_shared()

    if not shared_dir or not shared_dir.exists():
        print("\n❌ Dossier 'AI_Project_Shared' introuvable sur Google Drive.")
        print("Vérifiez que Google Drive pour PC est lancé et que le dossier")
        print("'AI_Project_Shared' est bien synchronisé ou présent dans Mon Drive.")
        print("=" * 65)
        return False

    print(f"✅ Dossier Google Drive détecté :\n   👉 {shared_dir}\n")

    # Local target directories
    local_models = ROOT / "models"
    local_runs = ROOT / "results" / "runs"
    local_baseline = ROOT / "results" / "baseline"
    local_models.mkdir(parents=True, exist_ok=True)
    local_runs.mkdir(parents=True, exist_ok=True)
    local_baseline.mkdir(parents=True, exist_ok=True)

    synced_items = 0

    # 1. Sync Baselines
    drive_baseline = shared_dir / "results" / "baseline"
    if drive_baseline.exists():
        for b_file in drive_baseline.glob("*.json"):
            dest = local_baseline / b_file.name
            if not dest.exists() or b_file.stat().st_mtime > dest.stat().st_mtime:
                shutil.copy2(b_file, dest)
                print(f"  📊 [Baseline] {b_file.name} ➔ results/baseline/")
                synced_items += 1

    # 2. Sync Models and Runs
    # Search for all best.pt inside shared_dir
    for root, dirs, files in os.walk(shared_dir):
        root_path = Path(root)
        for file in files:
            if file == "best.pt":
                # Determine run_id
                # Check if path contains 'transformer' or 'lstm' and 'SRR3544218'
                lower_path = str(root_path).lower()
                if "srr3544218" in lower_path:
                    if "transformer" in lower_path:
                        run_id = "transformer-notebook-srr"
                    elif "lstm" in lower_path:
                        run_id = "lstm-notebook-srr"
                    else:
                        run_id = "model-srr3544218"
                else:
                    # Use directory name or parent
                    run_id = root_path.name if root_path.name != "models" else "colab-model"

                # Copy checkpoint
                target_model_dir = local_models / run_id
                target_model_dir.mkdir(parents=True, exist_ok=True)
                dest_pt = target_model_dir / "best.pt"

                src_pt = root_path / file
                size_mb = src_pt.stat().st_size / 1024 / 1024

                if not dest_pt.exists() or src_pt.stat().st_mtime > dest_pt.stat().st_mtime:
                    shutil.copy2(src_pt, dest_pt)
                    print(f"  🧠 [Modèle]   {run_id:<26} ➔ models/{run_id}/best.pt ({size_mb:.2f} MB)")
                    synced_items += 1
                else:
                    print(f"  ✅ [À jour]   {run_id:<26} ({size_mb:.2f} MB)")

            elif file == "history.csv":
                lower_path = str(root_path).lower()
                if "srr3544218" in lower_path:
                    if "transformer" in lower_path:
                        run_id = "transformer-notebook-srr"
                    elif "lstm" in lower_path:
                        run_id = "lstm-notebook-srr"
                    else:
                        run_id = "model-srr3544218"
                else:
                    run_id = root_path.name

                target_run_dir = local_runs / run_id
                target_run_dir.mkdir(parents=True, exist_ok=True)
                dest_csv = target_run_dir / "history.csv"

                src_csv = root_path / file
                if not dest_csv.exists() or src_csv.stat().st_mtime > dest_csv.stat().st_mtime:
                    shutil.copy2(src_csv, dest_csv)
                    print(f"  📈 [History]  {run_id:<26} ➔ results/runs/{run_id}/history.csv")
                    synced_items += 1

    print("\n" + "=" * 65)
    print(f"✨ Synchronisation terminée avec succès !")
    print("   Actualisation de dashboard/data.json...")

    # Run export script automatically
    try:
        from scripts.export_dashboard_data import main as export_main
        export_main()
    except Exception as e:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "export_dashboard_data.py")], check=False)

    print("🚀 BioLM Studio est à jour avec vos vrais modèles SRR3544218 !")
    print("   Ouvrez : http://localhost:8765")
    print("=" * 65)
    return True


if __name__ == "__main__":
    sync_drive()
