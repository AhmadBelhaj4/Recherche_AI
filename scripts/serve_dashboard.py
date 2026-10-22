"""serve_dashboard.py - Local server for BioLM Studio.

Serves the dashboard frontend with API support for live model generation
and metrics inspection.

Usage:
    python scripts/serve_dashboard.py
    python scripts/serve_dashboard.py --port 8080
"""

import argparse
import http.server
import json
import socketserver
import sys
from pathlib import Path
import urllib.parse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DASHBOARD_DIR = ROOT / "dashboard"


class BioLMRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler with /api endpoints and static file serving."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({"status": "ok", "service": "BioLM Studio Server"})
            return
        elif parsed.path == "/api/data":
            data_file = DASHBOARD_DIR / "data.json"
            if data_file.exists():
                self._send_json(json.loads(data_file.read_text(encoding="utf-8")))
            else:
                self._send_json({"error": "data.json not found"}, status=404)
            return

        # Serve static files from dashboard/
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(body)
            except Exception:
                payload = {}

            result = self._handle_generation(payload)
            self._send_json(result)
        elif parsed.path == "/api/upload_checkpoint":
            query = urllib.parse.parse_qs(parsed.query)
            model_id = query.get("model_id", ["default"])[0]
            content_length = int(self.headers.get("Content-Length", 0))
            raw_data = self.rfile.read(content_length)

            target_dir = ROOT / "models" / model_id
            target_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = target_dir / "best.pt"
            checkpoint_file.write_bytes(raw_data)

            print(f"📥 Checkpoint received via Web Dashboard: {checkpoint_file} ({len(raw_data)/1024/1024:.2f} MB)")
            self._send_json({"status": "ok", "message": f"Checkpoint best.pt saved for model {model_id}", "size_mb": round(len(raw_data)/1024/1024, 2)})
            return

        self._send_json({"error": "Endpoint not found"}, status=404)

    def _handle_generation(self, payload: dict) -> dict:
        """Attempt neural generation using PyTorch checkpoints; fall back to Markov."""
        model_id = payload.get("model_id", "toy-test")
        prompt = payload.get("prompt", "")
        temperature = float(payload.get("temperature", 0.85))
        top_p = float(payload.get("top_p", 0.90))
        top_k = int(payload.get("top_k", 0))
        max_len = int(payload.get("max_len", 60))
        greedy = bool(payload.get("greedy", False))

        # Check if PyTorch can be imported
        try:
            import torch
            from utils import AminoAcidTokenizer, BOS_ID, EOS_ID, PAD_ID
            from models import LSTMLM, CausalTransformerLM
            from sampling import generate

            checkpoint_path = self._find_checkpoint(model_id)
            config_path = ROOT / "results" / "runs" / model_id / "config.json"

            if checkpoint_path and checkpoint_path.exists():
                print(f"⚡ Inférence active avec checkpoint : {checkpoint_path}")
                device = torch.device("cpu")
                state_dict = torch.load(checkpoint_path, map_location=device)

                # Initialize appropriate architecture
                if "lstm" in model_id.lower():
                    model = LSTMLM(vocab_size=23, embed_dim=128, hidden_dim=256, num_layers=1)
                else:
                    # Transformer
                    num_layers = 4
                    if config_path.exists():
                        cfg = json.loads(config_path.read_text(encoding="utf-8"))
                        mc = cfg.get("model_config", {})
                        num_layers = mc.get("num_layers", 4)
                    model = CausalTransformerLM(
                        vocab_size=23,
                        embed_dim=128,
                        num_heads=4,
                        num_layers=num_layers,
                        dim_feedforward=512,
                        max_len=200,
                        dropout=0.0
                    )

                model.load_state_dict(state_dict)
                model.eval()

                # Generate
                seqs, dones = generate(
                    model=model,
                    n=1,
                    device=device,
                    max_len=max_len,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    greedy=greedy,
                    seed=42
                )
                seq = seqs[0] if seqs else "MAALVQ"
                # If user specified prompt, prepend or replace prefix
                if prompt and not seq.startswith(prompt):
                    seq = prompt + seq[len(prompt):] if len(seq) > len(prompt) else prompt
                return {"sequence": seq, "finished": dones[0] if dones else True, "backend": "pytorch"}

        except Exception as err:
            pass

        # Check if pre-computed samples from Colab GPU exist in metrics.json
        metrics_file = ROOT / "results" / "runs" / model_id / "metrics.json"
        if metrics_file.exists():
            try:
                m_data = json.loads(metrics_file.read_text(encoding="utf-8"))
                samples = m_data.get("sample_generations", [])
                if samples:
                    import random
                    base = random.choice(samples)
                    seq = prompt + base[len(prompt):max_len] if len(base) > len(prompt) else prompt + base
                    return {"sequence": seq[:max_len], "finished": True, "backend": "colab_gpu_precomputed"}
            except Exception:
                pass

        # Markov Fallback
        seq = self._markov_generate(prompt=prompt, max_len=max_len)
        return {"sequence": seq, "finished": True, "backend": "markov_fallback"}

    def _find_checkpoint(self, model_id: str) -> Path | None:
        """Find best.pt in local models/ or on mounted Google Drive on Windows."""
        # 1. Local path
        local_path = ROOT / "models" / model_id / "best.pt"
        if local_path.exists():
            return local_path

        # 2. Check virtual Windows drive letters for Google Drive for Desktop (G:, H:, etc.)
        for letter in "GHIJKLMNOPQRSTUVWXYZFED":
            try:
                root_drive = Path(f"{letter}:/")
                if root_drive.exists():
                    for sub in ["Mon Drive", "My Drive", ""]:
                        cand = root_drive / sub / "recherche_ai_models" / model_id / "best.pt"
                        if cand.exists():
                            print(f"🔍 Checkpoint trouvé directement sur Google Drive Windows : {cand}")
                            return cand
            except (PermissionError, OSError):
                continue

        # 3. Check User Profile folder (Google Drive, OneDrive)
        for folder_name in ["Google Drive", "GoogleDrive", "Drive", "OneDrive"]:
            try:
                cand = Path.home() / folder_name / "recherche_ai_models" / model_id / "best.pt"
                if cand.exists():
                    print(f"🔍 Checkpoint trouvé dans {folder_name} : {cand}")
                    return cand
            except (PermissionError, OSError):
                continue

        return None

    def _markov_generate(self, prompt: str, max_len: int) -> str:
        """Lightweight Markov chain protein generator from baseline data."""
        import random
        baseline_file = ROOT / "results" / "baseline" / "bigram_toy_a1.json"
        sample_pool = []
        if baseline_file.exists():
            try:
                b_data = json.loads(baseline_file.read_text(encoding="utf-8"))
                sample_pool = b_data.get("sample_generations", [])
            except Exception:
                pass

        if sample_pool:
            base = random.choice(sample_pool)
            seq = prompt + base[len(prompt):max_len] if len(base) > len(prompt) else prompt + base
            return seq[:max_len]

        # Generic amino acid random sequence
        aas = "ACDEFGHIKLMNPQRSTVWY"
        seq = prompt if prompt else "M"
        for _ in range(max_len - len(seq)):
            seq += random.choice(aas)
        return seq

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_server(port: int = 8765):
    handler = BioLMRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    
    current_port = port
    for _ in range(10):
        try:
            with socketserver.TCPServer(("", current_port), handler) as httpd:
                print("=" * 60)
                print(f"  BioLM Studio is running at: http://localhost:{current_port}")
                print(f"  Serving files from: {DASHBOARD_DIR}")
                print(f"  Press Ctrl+C to stop the server.")
                print("=" * 60)
                httpd.serve_forever()
                break
        except OSError:
            current_port += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start BioLM Studio dashboard server")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    args = parser.parse_args()
    run_server(args.port)
