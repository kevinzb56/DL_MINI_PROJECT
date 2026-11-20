"""Build a self-contained GPU handoff and a private prepared-data ZIP."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[2]


def build():
    prepared=ROOT/'data/external/milano_prepared_v1'
    audit=json.loads((prepared/'preparation_audit.json').read_text())
    if audit['state']!='prepared':
        raise ValueError('Preparation audit blocked')
    out=ROOT/'output/kaggle'; out.mkdir(exist_ok=True,parents=True)
    package=out/'Milano_Prepared_v1.zip'
    existed=package.exists()
    if not existed:
        with zipfile.ZipFile(package,'w',zipfile.ZIP_STORED) as z:
            for p in sorted(prepared.rglob('*')):
                if p.is_file(): z.write(p,p.relative_to(prepared).as_posix())
    h=hashlib.sha256()
    with package.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''): h.update(chunk)
    digest=h.hexdigest()
    if existed:
        if package.with_suffix('.zip.sha256').read_text().split()[0]!=digest:
            raise ValueError('Existing package changed')
        with zipfile.ZipFile(package) as z:
            if z.read('manifest.json')!=(prepared/'manifest.json').read_bytes():
                raise ValueError('Existing package uses a different manifest')
    package.with_suffix('.zip.sha256').write_text(digest+'  '+package.name+'\n')
    bundle=io.BytesIO()
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT/'src').glob('*.py')): z.write(p,p.relative_to(ROOT).as_posix())
        for name in ('scripts/experiments/train_baseline.py','scripts/data/prepare_milano.py'):
            z.write(ROOT/name,name)
        for arch in ('litecrack','litecrack_ba'):
            config=json.loads((ROOT/f'configs/{arch}.json').read_text())
            config.update(protocol='milano_v1_publisher_splits',evaluation_splits=['source_holdout'])
            z.writestr(f'configs/milano_{arch}.json',json.dumps(config,indent=2))
    payload=base64.b64encode(bundle.getvalue()).decode()
    cells=[]
    def cell(kind,source): cells.append({'cell_type':kind,'metadata':{},'source':source.splitlines(True),**({'outputs':[],'execution_count':None} if kind=='code' else {})})
    cell('markdown',"""# Milano: LiteCrack versus LiteCrack-BA
Attach `Milano_Prepared_v1.zip` as a **private** Kaggle input. Enable GPU.
Uses publisher train/validation/test splits, random initialization, seed 42, 256px,
100 epochs maximum, patience 15, threshold 0.5 and per-image macro metrics.
This is a new dataset experiment, not three-seed confirmation of the original study.
Labels include pavement distress. Site independence and vegetation performance are unverified.
Review preparation_audit.json and training_review.jpg before running.
Run each model cell separately; download its recovery ZIP and SHA256 immediately.
Interrupting the subprocess with SIGINT allows the trainer to checkpoint; a hard
runtime loss can replay up to 29 steps. Unexported Kaggle storage is ephemeral.
Restore a recovery ZIP in a fresh runtime with the optional restore cell.
Runtime/code/config identities must match for exact recovery; no dependencies are upgraded here.
""")
    cell('code',f'''import os, sys, json, hashlib, zipfile, base64, io, subprocess, shutil
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch, torchvision, numpy, PIL
assert torch.cuda.is_available(), 'Enable Kaggle GPU before continuing'
ROOT=Path('/kaggle/working/cracklab_milano_v1')
ROOT.mkdir(exist_ok=True)
payload=base64.b64decode({payload!r})
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    for item in z.infolist():
        target=ROOT/item.filename
        if target.exists() and target.read_bytes()!=z.read(item):
            raise ValueError('Code changed; use a fresh project/runtime')
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(z.read(item))
os.chdir(ROOT); sys.path.insert(0,str(ROOT))
archives=list(Path('/kaggle/input').rglob('Milano_Prepared_v1.zip'))
assert len(archives)==1, 'Attach exactly one prepared dataset ZIP'
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''): h.update(chunk)
    return h.hexdigest()
assert file_sha(archives[0])=={digest!r}, 'Dataset archive checksum mismatch'
DATA=ROOT/'data'
if not (DATA/'manifest.json').exists():
    with zipfile.ZipFile(archives[0]) as z:
        assert all((DATA/n).resolve().is_relative_to(DATA.resolve()) for n in z.namelist())
        assert shutil.disk_usage(ROOT).free>sum(x.file_size for x in z.infolist())+3*1024**3
        z.extractall(DATA)
manifest_path=DATA/'manifest.json'
assert file_sha(manifest_path)==(DATA/'manifest.json.sha256').read_text().split()[0]
audit=json.loads((DATA/'preparation_audit.json').read_text())
assert audit['state']=='prepared' and not audit['decoded_cross_split_duplicates']
from src.dataset import verify_files
manifest=json.loads(manifest_path.read_text()); verify_files(manifest['records'],DATA)
print('Verified',len(manifest['records']),'pairs; GPU:',torch.cuda.get_device_name())
print('Similarity candidates:',len(audit['similarity_candidates']))
print('Versions:',torch.__version__,torchvision.__version__,numpy.__version__,PIL.__version__)
''')
    cell('markdown','Optional recovery: set RECOVERY to an attached, trusted recovery ZIP. Leave None for a new run. Restore only into a fresh artifacts directory.')
    cell('code', '''RECOVERY=None  # e.g. Path('/kaggle/input/my-backup/milano_recovery_litecrack_....zip')
if RECOVERY is not None:
    RECOVERY=Path(RECOVERY)
    assert file_sha(RECOVERY)==RECOVERY.with_suffix('.zip.sha256').read_text().split()[0]
    assert not (ROOT/'artifacts').exists(), 'Restore only into a fresh project'
    with zipfile.ZipFile(RECOVERY) as z:
        assert all((ROOT/n).resolve().is_relative_to(ROOT.resolve()) for n in z.namelist())
        z.extractall(ROOT)
runtime={'torch':torch.__version__,'torchvision':torchvision.__version__,
         'numpy':numpy.__version__,'pillow':PIL.__version__,'cuda':torch.version.cuda,
         'gpu':torch.cuda.get_device_name(),'python':sys.version}
lock=ROOT/'artifacts/runtime.json'
if lock.exists(): assert json.loads(lock.read_text())==runtime, 'Recovery runtime changed'
else:
    lock.parent.mkdir(exist_ok=True); lock.write_text(json.dumps(runtime,indent=2))
''')
    cell('code', '''from datetime import datetime, timezone
def export_recovery(arch):
    run=ROOT/'artifacts'/arch
    if not run.exists(): return
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=ROOT.parent/f'milano_recovery_{arch}_{stamp}.zip'
    with zipfile.ZipFile(backup,'w',zipfile.ZIP_STORED) as z:
        # Preserve logs/config and all pointers, but only latest/best checkpoint payloads.
        for model_run in sorted((ROOT/'artifacts').iterdir()):
            if model_run.is_file():
                z.write(model_run,model_run.relative_to(ROOT).as_posix()); continue
            keep=set()
            pointers={}
            for pointer in model_run.glob('checkpoints/*.json'):
                content=pointer.read_bytes(); info=json.loads(content); target=pointer.parent/info['file']
                assert target.resolve().parent==pointer.parent.resolve()
                assert file_sha(target)==info['sha256'], 'Corrupt recovery pointer'
                keep.add(target.resolve())
                pointers[pointer.resolve()]=content
            for p in model_run.rglob('*'):
                if p.resolve() in pointers:
                    z.writestr(p.relative_to(ROOT).as_posix(),pointers[p.resolve()])
                elif p.is_file() and (p.suffix!='.pt' or p.resolve() in keep):
                    z.write(p,p.relative_to(ROOT).as_posix())
    backup.with_suffix('.zip.sha256').write_text(file_sha(backup)+'  '+backup.name+chr(10))
    from IPython.display import display, FileLink
    display(FileLink(str(backup)),FileLink(str(backup.with_suffix('.zip.sha256'))))

def run_model(arch):
    run=ROOT/'artifacts'/arch
    if (run/'status.json').exists() and json.loads((run/'status.json').read_text())['state']=='training_complete' and (run/'evaluation.json').exists():
        print('Already completed:',arch); export_recovery(arch); return
    assert shutil.disk_usage(ROOT).free>3*1024**3, 'Export and restore in fresh storage'
    cmd=[sys.executable,'scripts/experiments/train_baseline.py','--config',f'configs/milano_{arch}.json',
         '--manifest',str(manifest_path),'--data-root',str(DATA),'--run-dir',str(run)]
    if (run/'checkpoints/latest.json').exists(): cmd.append('--resume')
    process=subprocess.Popen(cmd)
    try:
        code=process.wait()
    except KeyboardInterrupt:
        import signal
        process.send_signal(signal.SIGINT)
        code=process.wait()
    finally:
        if process.poll() is not None: export_recovery(arch)
    assert code==0, 'Training failed; inspect failure.json and preserve recovery'
''')
    for arch in ('litecrack','litecrack_ba'): cell('code',f"run_model({arch!r})\n")
    cell('code', '''comparison={}
for arch in ('litecrack','litecrack_ba'):
    run=ROOT/'artifacts'/arch
    assert json.loads((run/'status.json').read_text())['state']=='training_complete'
    comparison[arch]=json.loads((run/'evaluation.json').read_text())
(ROOT/'artifacts/comparison.json').write_text(json.dumps(comparison,indent=2))
print(json.dumps(comparison,indent=2))
export_recovery('litecrack_ba')
''')
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
    path=out/'Road_Crack_Milano_v1.ipynb'
    path.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
    print(path); print(package); print(digest)


if __name__=='__main__': build()
