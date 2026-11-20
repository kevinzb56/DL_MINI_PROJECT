"""Build a self-contained Kaggle Phase 1 notebook; never include access URLs."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "kaggle"
OUTPUT.mkdir(parents=True, exist_ok=True)
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
    for directory in ("src", "scripts", "tests", "configs", "docs", "data/manifests"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.name != Path(__file__).name:
                archive.write(path, path.relative_to(ROOT).as_posix())
    for filename in ("README.md", "requirements.txt", ".gitignore"):
        archive.write(ROOT / filename, filename)
payload = buffer.getvalue()
(OUTPUT / "phase1_source.zip").write_bytes(payload)
encoded = base64.b64encode(payload).decode("ascii")
digest = hashlib.sha256(payload).hexdigest()
cells = []


def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(True)})


markdown("""# Road Crack Segmentation — Phase 1 on Kaggle

Self-contained source bundle. No private session link or credentials are included.
Select a GPU accelerator. Internet is needed only for missing packages/data downloads.
Run cells in order. This notebook checks the environment and recovery foundation;
it does not train a crack model or claim Phase 1 data completion.
""")
code(f'''import base64, hashlib, io, os, zipfile
from pathlib import Path
payload = base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
project = Path('/kaggle/working/road_crack_phase1')
project.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for item in archive.infolist():
        target = (project / item.filename).resolve()
        assert target.is_relative_to(project.resolve()), 'Unsafe archive entry'
        if target.exists():
            assert target.read_bytes() == archive.read(item), 'Existing source differs; preserve and use a new project directory'
    archive.extractall(project)
os.chdir(project)
print('Project:', project)
print('Source SHA256:', hashlib.sha256(payload).hexdigest())
''')
code('''import importlib.util, subprocess, sys
# Keep Kaggle's preinstalled torch/CUDA stack. Install only missing utilities.
if importlib.util.find_spec('torch') is None:
    raise RuntimeError('PyTorch missing: select the standard Kaggle GPU image before proceeding')
missing = [pkg for module, pkg in [('numpy','numpy'), ('PIL','Pillow'), ('pytest','pytest')]
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
subprocess.run([sys.executable, 'scripts/experiments/environment.py'], check=True)
import torch
print('CUDA runtime:', torch.version.cuda)
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable. Enable a GPU accelerator and rerun')
print('GPU:', torch.cuda.get_device_name(0))
x = torch.randn(32, 32, device='cuda')
print('GPU smoke result:', float((x @ x.T).mean()))
''')
code('''import json, subprocess, sys
evidence = Path('artifacts/phase1')
evidence.mkdir(parents=True, exist_ok=True)
result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-rA'],
                        text=True, capture_output=True)
(evidence / 'tests.txt').write_text(result.stdout + result.stderr)
(evidence / 'requirements.kaggle.lock.txt').write_text(
    subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
(evidence / 'verification.json').write_text(json.dumps({
    'test_returncode': result.returncode,
    'scope': 'Foundation fixtures only; no real-data manifest or training yet',
    'gpu_available': torch.cuda.is_available(),
    'torch_version': torch.__version__}, indent=2))
print(result.stdout, result.stderr)
assert result.returncode == 0, 'Tests failed: inspect artifacts/phase1/tests.txt'
''')
markdown('''## Optional: acquire real data

Run the next cell when Internet is enabled. Existing repositories are preserved.
DeepCrack is restricted to noncommercial research/education per its authors.
CrackMap terms, annotations and grouping metadata require review. Downloading alone
does not complete the audit. No group IDs or final splits are invented here.
''')
code('''import subprocess
for name, url in [
    ('DeepCrack', 'https://github.com/yhlleo/DeepCrack.git'),
    ('CrackMap', 'https://github.com/ikatsamenis/CrackMap.git')]:
    destination = Path('data/raw') / name
    if not destination.exists():
        subprocess.run(['git', 'clone', '--depth', '1', url, str(destination)], check=True)
    commit = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    print(name, commit)
    (evidence / f'{name}_commit.txt').write_text(commit + '\\n')
print('Next: inspect pairs/polarity and parent groups, create index_v1.csv per README, then run prepare_data.py.')
''')
markdown('''## Export evidence before ending the session

Run this cell even if a previous cell failed. It bundles code and results, excluding
raw datasets. Download the bundle from the notebook Output/Files panel and keep it
outside the Kaggle session. Use Kaggle Save Version/output retention as an additional
copy. Files in a live working directory alone are not a verified external backup.
The notebook never automatically ends the session or deletes checkpoints.
''')
code('''import hashlib, zipfile
backup = Path('/kaggle/working/phase1_results.zip')
with zipfile.ZipFile(backup, 'w', zipfile.ZIP_DEFLATED) as archive:
    for directory in ['src', 'scripts', 'tests', 'configs', 'docs', 'data/manifests', 'artifacts']:
        for path in sorted(Path(directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.write(path, path.as_posix())
    for name in ['README.md', 'requirements.txt']:
        archive.write(name, name)
digest = hashlib.sha256(backup.read_bytes()).hexdigest()
backup.with_suffix('.zip.sha256').write_text(digest + '  ' + backup.name + '\\n')
print('Download:', backup)
print('SHA256:', digest)
from IPython.display import FileLink, display
display(FileLink(str(backup)))
''')
notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"}}, "cells": cells}
for i, cell in enumerate(cells):
    cell['id'] = f'phase1-{i:02d}'
target = OUTPUT / "Road_Crack_Phase1_Kaggle.ipynb"
target.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(target)
print('Bundle SHA256:', digest)
