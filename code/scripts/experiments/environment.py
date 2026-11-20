"""Record current interpreter and actual dependency/GPU availability."""
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.atomic import write_json

report = {"python": sys.version, "executable": sys.executable,
          "platform": platform.platform(), "packages": {}}
for name in ("torch", "numpy", "Pillow", "pytest"):
    try:
        report["packages"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        report["packages"][name] = None
try:
    import torch
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except ImportError:
    report["cuda_available"] = "unverified: torch not installed"
write_json("artifacts/phase1/environment.json", report)
print(json.dumps(report, indent=2))
