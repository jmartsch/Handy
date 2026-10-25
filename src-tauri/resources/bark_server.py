"""Persistent Bark worker for Handy. Protocol: one JSON per line on stdin ({"wav": path}),
one JSON per line on stdout ({"text": ...} | {"error": ...}). Sends {"ready": true} once loaded."""
import importlib.util
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    import numpy as np
    import soundfile as sf
    import torch

    spec = importlib.util.spec_from_file_location("official_bark", root / "bark_loader.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    model = loader.load_model(str(root / "model.bark"), device=device, dtype=torch.float32)
    out = sys.stdout
    out.write(json.dumps({"ready": True}) + "\n")
    out.flush()
    for line in sys.stdin:
        try:
            wav = json.loads(line)["wav"]
            wave, sr = sf.read(wav, dtype="float32", always_2d=True)
            text = loader.transcribe(model, wave.mean(1), sr) if len(wave) else ""
            out.write(json.dumps({"text": text}) + "\n")
        except Exception as e:  # keep worker alive
            out.write(json.dumps({"error": str(e)}) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
