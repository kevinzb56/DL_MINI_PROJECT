"""Make a new recovery ZIP including latest/best states, without pruning originals."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from followup.adaptation import sha
from scripts.release_integrity import safe_member


def package(root,target):
    root=Path(root).resolve();target=Path(target).resolve()
    if target.is_relative_to(root):raise ValueError('Backup must be outside the run tree')
    selected={}
    for path in root.rglob('*'):
        parts=path.relative_to(root).parts
        # pytest basetemp trees intentionally contain damaged checkpoint fixtures.
        # They are test scratch data, never recoverable training state.
        if ('test_evidence' in parts and any(p.startswith('tmp-') for p in parts)):
            continue
        if not path.is_file() or path.suffix=='.pt' or path.name=='FOLLOWUP_BACKUP_MANIFEST.json' or any(p in ('__pycache__','.pytest_cache','.venv','node_modules') for p in parts):continue
        name=path.relative_to(root).as_posix();safe_member(name);selected[name]=path
    for pointer in root.rglob('checkpoints/*.json'):
        if pointer.name not in ('latest.json','best.json'):continue
        if 'test_evidence' in pointer.relative_to(root).parts:continue
        entry=json.loads(pointer.read_text());snapshot=(pointer.parent/entry['file']).resolve()
        if snapshot.parent!=pointer.parent.resolve() or sha(snapshot)!=entry['sha256']:raise ValueError('Recovery pointer corrupt')
        selected[snapshot.relative_to(root).as_posix()]=snapshot
    # Local run directories may contain results only. Preserve the exact available
    # workflow source separately as well as any code already inside a Kaggle project.
    code_root=Path(__file__).resolve().parents[2]
    for folder in ('src','followup','inspection','configs'):
        for path in (code_root/folder).glob('*'):
            if path.is_file() and path.suffix in ('.py','.json'):
                selected['_followup_source/'+path.relative_to(code_root).as_posix()]=path
    for name in ('scripts/experiments/run_followup.py','scripts/experiments/train_baseline.py','scripts/data/restore_datasets.py',
                 'scripts/data/audit_kaggle_data.py','scripts/release_integrity.py','scripts/operations/package_followup.py',
                 'scripts/evaluation/summarize_followup.py','scripts/operations/followup_annotation.py','tests/test_followup.py',
                 'data/manifests/pilot_v1.json','data/manifests/pilot_v1.json.sha256'):
        path=code_root/name
        if path.is_file():selected['_followup_source/'+name]=path
    if not selected:raise ValueError('Nothing to back up')
    inventory={n:{'sha256':sha(p),'bytes':p.stat().st_size} for n,p in sorted(selected.items())}
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for name,path in sorted(selected.items()):z.write(path,name)
        z.writestr('FOLLOWUP_BACKUP_MANIFEST.json',json.dumps({'files':inventory,
                   'created':datetime.now(timezone.utc).isoformat(),'scope':'Latest/best recoverable states; older snapshots remain at origin.'},indent=2))
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise ValueError('ZIP CRC failed')
        import hashlib
        for name,entry in inventory.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=entry['sha256']:raise ValueError('Backup checksum failed')
    target.with_suffix(target.suffix+'.sha256').write_text(sha(target)+'  '+target.name+'\n')
    return {'archive':str(target),'sha256':sha(target),'files':len(inventory),'bytes':target.stat().st_size}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(package(a.root,a.output),indent=2))
