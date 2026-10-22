"""colab_sync.py - Hybrid Colab <-> Local PC Sync Utility.

Handles the industry-standard hybrid ML workflow:
1. Model weights (.pt) -> Google Drive (persistent, safe from Colab VM timeout, no Git bloat).
2. Lightweight metrics (.json, .csv) -> Git / Dashboard (instant sync to local PC).
3. Sample generations -> pre-computed on Colab GPU in 0.2s and embedded in metrics.json.
4. One-click Zip export for Colab users who don't want to configure Git credentials.

Usage in Google Colab:
    from colab_sync import save_colab_run
    save_colab_run(run_id="transformer-colab-real", model=model, history=history, config=config, sync_drive=True, push_git=True)
"""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def is_running_in_colab() -> bool:
    """Detect if the environment is Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def save_colab_run(
    run_id: str,
    model=None,
    history: list[dict] | None = None,
    config: dict | None = None,
    test_metrics: dict | None = None,
    device=None,
    sync_drive: bool = True,
    drive_folder: str = "/content/drive/MyDrive/recherche_ai_models",
    push_git: bool = False,
    git_branch: str = "main",
    n_samples: int = 5,
    auto_download_zip: bool = False,
) -> dict:
    """Save an experiment run using the hybrid workflow.

    Returns the generated metrics dictionary.
    """
    in_colab = is_running_in_colab()
    res_dir = ROOT / "results" / "runs" / run_id
    res_dir.mkdir(parents=True, exist_ok=True)
    models_dir = ROOT / "models" / run_id
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BioLM Hybrid Sync: Processing run '{run_id}'")
    print(f"{'='*60}")

    # 1. Save config.json
    if config:
        config_path = res_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"  [1/5] Config saved: {config_path}")

    # 2. Save history.csv
    if history:
        hist_path = res_dir / "history.csv"
        with open(hist_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"  [2/5] Training history saved: {hist_path} ({len(history)} epochs)")

    # 3. Generate sample sequences on Colab GPU (Pre-computed samples)
    sample_generations = []
    if model is not None:
        try:
            import torch
            from sampling import generate
            dev = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            print(f"  [3/5] Sampling {n_samples} protein sequences on {dev}...")
            seqs, _ = generate(model, n=n_samples, device=dev, max_len=80, temperature=0.85, seed=42)
            sample_generations = seqs
            print(f"        Generated {len(seqs)} samples (e.g. {seqs[0][:25]}...)")
        except Exception as e:
            print(f"  [3/5] Note: Sample generation skipped ({e})")

    # 4. Save metrics.json
    metrics = {}
    if history:
        best_row = min(history, key=lambda x: x.get("val_nats") or float("inf"))
        total_sec = sum(r.get("seconds", 0) or 0 for r in history)
        metrics = {
            "best_epoch": int(best_row.get("epoch", 0)),
            "best_val_nats": float(best_row.get("val_nats", 0)),
            "best_val_ppl": float(best_row.get("val_ppl", 0)),
            "final_train_nats": float(history[-1].get("train_nats", 0)),
            "final_train_ppl": float(history[-1].get("train_ppl", 0)),
            "epochs_run": len(history),
            "train_seconds": round(total_sec, 2),
            "sample_generations": sample_generations,
        }
        if config and "n_params" in config:
            metrics["n_params"] = config["n_params"]
        elif model is not None:
            try:
                metrics["n_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            except Exception:
                pass

        metrics_path = res_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"  [4/5] Metrics summary saved: {metrics_path} (Best PPL: {metrics['best_val_ppl']:.2f})")

    # 5. Save test_metrics.json if present
    if test_metrics:
        (res_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    # 6. Save heavy checkpoint (.pt) to Google Drive
    best_pt = models_dir / "best.pt"
    if sync_drive and in_colab:
        try:
            drive_dest = Path(drive_folder) / run_id
            drive_dest.mkdir(parents=True, exist_ok=True)
            if best_pt.exists():
                shutil.copy(best_pt, drive_dest / "best.pt")
                print(f"  [5/5] Checkpoint synced to Google Drive: {drive_dest / 'best.pt'}")
            last_pt = models_dir / "last.pt"
            if last_pt.exists():
                shutil.copy(last_pt, drive_dest / "last.pt")
        except Exception as e:
            print(f"  [5/5] Drive sync skipped: {e}")
    else:
        print(f"  [5/5] Local checkpoint saved at: {best_pt}")

    # 7. Optional Git Push from Colab (only lightweight text files)
    if push_git:
        try:
            print("  [GIT] Staging results/runs/ for git push...")
            subprocess.run(["git", "add", f"results/runs/{run_id}"], check=True)
            commit_msg = f"colab run: {run_id} (val_ppl: {metrics.get('best_val_ppl', 0):.2f})"
            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
            subprocess.run(["git", "push", "origin", git_branch], check=True)
            print("  [GIT] Successfully pushed metrics to GitHub!")
        except Exception as e:
            print(f"  [GIT] Git push skipped or failed: {e}")

    # 8. Optional Auto-download zip in Colab
    if auto_download_zip and in_colab:
        try:
            from google.colab import files
            zip_name = f"{run_id}_results"
            zip_file = shutil.make_archive(zip_name, "zip", res_dir)
            files.download(zip_file)
            print(f"  [ZIP] Auto-downloading {zip_name}.zip to your browser!")
        except Exception as e:
            print(f"  [ZIP] Download error: {e}")

    print(f"\nDone! On your local PC, run:\n   git pull\n   python scripts/export_dashboard_data.py\n")
    return metrics


def package_run_as_zip(run_id: str, include_model: bool = False) -> str:
    """Create a clean zip archive of a run for easy transport."""
    res_dir = ROOT / "results" / "runs" / run_id
    if not res_dir.exists():
        raise FileNotFoundError(f"Run directory {res_dir} does not exist.")

    zip_base = ROOT / f"{run_id}_dashboard_pack"
    archive = shutil.make_archive(str(zip_base), "zip", res_dir)
    return archive
