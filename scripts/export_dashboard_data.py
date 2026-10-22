"""Export all training runs and baselines into a single JSON file for the dashboard.

Usage:
    python scripts/export_dashboard_data.py
    python scripts/export_dashboard_data.py --output dashboard/data.json

Reads:
  results/runs/*/config.json, metrics.json, history.csv, test_metrics.json
  results/baseline/*.json
Writes:
  dashboard/data.json (or --output path)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def read_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def read_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({k: float(v) if v else None for k, v in row.items()})
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "dashboard" / "data.json"))
    args = ap.parse_args()

    runs_dir = ROOT / "results" / "runs"
    baseline_dir = ROOT / "results" / "baseline"

    # Collect runs
    models = []
    if runs_dir.is_dir():
        for run_path in sorted(runs_dir.iterdir()):
            if not run_path.is_dir():
                continue
            run_id = run_path.name
            config = read_json(run_path / "config.json")
            metrics = read_json(run_path / "metrics.json")
            history = read_csv(run_path / "history.csv")
            test_metrics = read_json(run_path / "test_metrics.json")

            entry = {
                "run_id": run_id,
                "type": "neural",
                "config": config,
                "metrics": metrics,
                "history": history or [],
                "test_metrics": test_metrics,
            }

            # Derive model type, params, and metrics if missing
            history_rows = history or []
            if not metrics and history_rows:
                # Compute best metrics from history
                best_row = min(history_rows, key=lambda r: r.get("val_nats") or float("inf"))
                total_time = sum(r.get("seconds", 0) or 0 for r in history_rows)
                metrics = {
                    "best_epoch": int(best_row.get("epoch", 0)),
                    "best_val_nats": best_row.get("val_nats"),
                    "best_val_ppl": best_row.get("val_ppl"),
                    "final_train_nats": history_rows[-1].get("train_nats"),
                    "final_train_ppl": history_rows[-1].get("train_ppl"),
                    "epochs_run": len(history_rows),
                    "train_seconds": round(total_time, 2),
                }
                entry["metrics"] = metrics

            if config and "model_config" in config:
                mc = config["model_config"]
                entry["model_type"] = mc.get("model", "unknown")
                entry["n_params"] = config.get("n_params", 0)
                entry["data_version"] = config.get("data_version", "unknown")
            else:
                # Infer from run_id
                m_type = "lstm" if "lstm" in run_id.lower() else ("transformer" if "transformer" in run_id.lower() else "unknown")
                entry["model_type"] = m_type
                entry["data_version"] = "toy" if "toy" in run_id.lower() else "real"
                # Standard parameter estimates
                if m_type == "lstm":
                    entry["n_params"] = 431104
                elif m_type == "transformer":
                    entry["n_params"] = 821888
                else:
                    entry["n_params"] = 0

            # Expose sample generations if available
            if metrics and "sample_generations" in metrics:
                entry["sample_generations"] = metrics["sample_generations"]
            else:
                entry["sample_generations"] = []

            models.append(entry)

    # Collect baselines
    baselines = []
    if baseline_dir.is_dir():
        for bp in sorted(baseline_dir.glob("*.json")):
            data = read_json(bp)
            if data:
                data["run_id"] = bp.stem
                data["type"] = "baseline"
                baselines.append(data)

    # Reference values
    references = {
        "uniform_ppl": 20.0,
        "ln_23": 3.1355,
        "bigram_val_ppl_real": 11.83,  # SRR3544218 baseline
    }

    output = {
        "exported_at": __import__("datetime").datetime.now().isoformat(),
        "models": models,
        "baselines": baselines,
        "references": references,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(models)} runs + {len(baselines)} baselines -> {out_path}")


if __name__ == "__main__":
    main()
