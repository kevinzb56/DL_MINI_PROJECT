"""Build a clean, verifiable CrackLab source and demo submission."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.frontend_build import fingerprint
from scripts.release_integrity import MANIFEST, MODEL_ARCHIVES, digest, safe_member, verify_archive
from scripts.project_layout import archive_path

SOURCE_DIRS = (
    "src", "inspection", "followup", "scripts", "tests", "configs",
    "data/manifests", "data/references", "docs", "web/src", "web/public", "web/dist",
    "release_assets",
)
ROOT_FILES = (
    "README.md", "SUBMISSION_README.md", "START_HERE.md", "MODEL_CARD.md", "LICENSE", "NOTICE.md",
    "Setup_CrackLab.ps1", "Start_CrackLab.cmd", "Start_CrackLab_Groq.ps1",
    "PRD_Lightweight_Few_Shot_Road_Crack_Segmentation.md",
    "requirements.txt", "requirements-app.txt", "requirements-app.lock.txt",
    "requirements-release.lock.txt", "requirements-phase7.txt",
    "requirements-phase7.lock.txt", "requirements-report.txt",
    "requirements-authoring.lock.txt", "web/README.md", "web/index.html",
    "web/package.json", "web/package-lock.json", "web/vite.config.js",
    "phase7_results.zip", "phase7_results.zip.sha256",
)
EVIDENCE_FILES = (
    "output/pdf/CrackLab_Final_Report.pdf",
    "output/phase8/evidence.json",
    "output/phase8/release_validation.json",
    "output/phase8/demo/demo.json",
    "output/multisource_test_v1/FINAL_TEST_REPORT.md",
    "output/multisource_test_v1/FROZEN_SELECTION.json",
    "output/multisource_test_v1/summary.json",
    "output/multisource_512_continuation/selected_model/best.pt",
    "output/multisource_512_continuation/selected_model/selection.json",
    "output/milano_v1/FINAL_REPORT.md",
    "output/followup_v1/downloaded_audit/RESULTS.md",
)
SKIP_PARTS = {"__pycache__", ".pytest_cache", "node_modules"}
SKIP_SUFFIXES = {".pyc", ".tmp", ".partial", ".log"}


def select(root):
    selected = {}

    def add(path, name=None):
        if not path.is_file() or path.is_symlink():
            return
        name = safe_member(name or path.relative_to(root).as_posix())
        if any(part in SKIP_PARTS for part in path.parts) or path.suffix in SKIP_SUFFIXES:
            return
        if path.name.startswith(".env"):
            return
        selected[name] = path

    for directory in SOURCE_DIRS:
        for path in (root / directory).rglob("*"):
            add(path)
    for name in (*ROOT_FILES, *EVIDENCE_FILES, *MODEL_ARCHIVES):
        path = archive_path(root, name) if name.endswith(".zip") or name.endswith(".zip.sha256") else root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required submission file missing: {name}")
        add(path, name)
    return selected


def build(root, target):
    root = Path(root).resolve()
    target = Path(target).resolve()
    if target.is_relative_to(root / "scripts"):
        raise ValueError("Write the submission outside scripts/")
    if json.loads((root / "web/dist/build_inputs.json").read_text()) != fingerprint(root / "web"):
        raise ValueError("Frontend is stale: run npm ci and npm run build in web/")
    selected = select(root)
    for name, expected in MODEL_ARCHIVES.items():
        if digest(archive_path(root, name)) != expected:
            raise ValueError(f"Model archive checksum mismatch: {name}")
    frozen = json.loads((root / "output/multisource_test_v1/FROZEN_SELECTION.json").read_text())
    if digest(root / "output/multisource_512_continuation/selected_model/best.pt") != frozen["checkpoint"]["sha256"]:
        raise ValueError("Multi-source checkpoint differs from frozen selection")
    entries = {name: {"sha256": digest(path), "bytes": path.stat().st_size}
               for name, path in sorted(selected.items())}
    manifest = {
        "version": "2026-11-20-submission",
        "scope": "Current CrackLab source, runnable local demo, selected multi-source model, and key evidence",
        "files": entries,
        "excluded": "Local environments, credentials, downloaded datasets, workspaces, recovery backups, and intermediate outputs",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(selected.items()):
            archive.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix in {".zip", ".pt", ".png", ".jpg", ".pdf"} else zipfile.ZIP_DEFLATED)
        archive.writestr(MANIFEST, json.dumps(manifest, indent=2))
    result = verify_archive(temporary)
    temporary.replace(target)
    target.with_suffix(target.suffix + ".sha256").write_text(f"{digest(target)}  {target.name}\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT.parent / "results/submission/CrackLab_Submission_2026-11-20.zip")
    args = parser.parse_args()
    print(json.dumps(build(ROOT, args.output), indent=2))
