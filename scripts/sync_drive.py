r"""sync_drive.py - Continuous Google Drive Desktop Sync Utility.

Automatically scans for Google Drive for Desktop mounted drives (e.g. G:\Mon Drive\recherche_ai_models)
and links/syncs all trained model checkpoints into models/<run_id>/best.pt.

Usage:
    python scripts/sync_drive.py
    python scripts/sync_drive.py --link    (creates Windows Directory Junctions - instant, 0 disk space)
    python scripts/sync_drive.py --copy    (copies .pt files locally)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent


def find_google_drive_folder() -> Path | None:
    """Detect the Google Drive for Desktop folder on Windows."""
    # Check drive letters (G:, H:, etc.)
    for letter in "GHIJKLMNOPQRSTUVWXYZFED":
        try:
            root = Path(f"{letter}:/")
            if root.exists():
                for sub in ["Mon Drive", "My Drive", ""]:
                    candidate = root / sub / "recherche_ai_models"
                    if candidate.exists():
                        return candidate
                    drive_parent = root / sub
                    if drive_parent.exists() and sub:
                        return candidate
        except (PermissionError, OSError):
            continue

    # Check User Profile directory
    for folder_name in ["Google Drive", "GoogleDrive", "Drive", "OneDrive"]:
        try:
            candidate = Path.home() / folder_name / "recherche_ai_models"
            if candidate.exists():
                return candidate
        except (PermissionError, OSError):
            continue

    return None


def sync_models(use_junction: bool = False, verbose: bool = True):
    print("=" * 65)
    print("  BioLM Studio — Google Drive Desktop Continuous Sync")
    print("=" * 65)

    drive_path = find_google_drive_folder()

    if not drive_path or not drive_path.parent.exists():
        print("\n❌ Google Drive pour PC n'a pas été détecté sur votre machine.")
        print("\nPour activer la synchronisation continue 100% automatique :")
        print("1. Téléchargez gratuitement Google Drive pour PC (Windows) :")
        print("   👉 https://www.google.com/intl/fr_ALL/drive/download/")
        print("2. Connectez-vous avec le même compte Google que celui de Colab.")
        print("3. Dès que l'application est lancée, votre Google Drive apparaîtra")
        print("   comme un lecteur Windows 'G:\\Mon Drive\\'.")
        print("\nRelancez ensuite ce script : `python scripts/sync_drive.py`.")
        print("=" * 65)
        return False

    drive_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Dossier Google Drive détecté : {drive_path}")

    # Scan for runs in Google Drive
    runs = [d for d in drive_path.iterdir() if d.is_dir()]
    if not runs:
        print(f"ℹ️ Aucun modèle trouvé dans {drive_path} pour l'instant.")
        print("   Dès que votre entraînement Colab (Section 7) se termine,")
        print("   le fichier 'best.pt' sera déposé ici automatiquement !")
        return True

    print(f"📦 {len(runs)} modèle(s) trouvé(s) sur Google Drive :\n")
    models_local = ROOT / "models"
    models_local.mkdir(parents=True, exist_ok=True)

    synced_count = 0
    for run_dir in runs:
        run_id = run_dir.name
        pt_file = run_dir / "best.pt"
        if not pt_file.exists():
            continue

        target_run_dir = models_local / run_id
        target_pt = target_run_dir / "best.pt"

        size_mb = pt_file.stat().st_size / 1024 / 1024

        if use_junction and os.name == "nt" and not target_run_dir.exists():
            # Create Windows directory junction (mklink /J)
            try:
                cmd = f'cmd /c mklink /J "{target_run_dir}" "{run_dir}"'
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
                print(f"  🔗 [Junction] {run_id:<30} ➔ {run_dir} ({size_mb:.2f} MB)")
                synced_count += 1
                continue
            except Exception:
                pass

        # Otherwise copy if missing or newer
        target_run_dir.mkdir(parents=True, exist_ok=True)
        if not target_pt.exists() or pt_file.stat().st_mtime > target_pt.stat().st_mtime:
            shutil.copy2(pt_file, target_pt)
            print(f"  ⬇️ [Copié]    {run_id:<30} ➔ models/{run_id}/best.pt ({size_mb:.2f} MB)")
            synced_count += 1
        else:
            print(f"  ✅ [À jour]   {run_id:<30} (déjà synchronisé, {size_mb:.2f} MB)")

    print(f"\n✨ Synchronisation terminée : {synced_count} nouveau(x) modèle(s) synchronisé(s) !")
    print("   Ces modèles sont immédiatement utilisables dans BioLM Studio : http://localhost:8765")
    print("=" * 65)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync models from Google Drive Desktop")
    parser.add_argument("--link", action="store_true", help="Create directory junctions instead of copying")
    args = parser.parse_args()
    sync_models(use_junction=args.link)
