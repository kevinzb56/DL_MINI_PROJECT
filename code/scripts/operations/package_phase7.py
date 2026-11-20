"""Create an immutable local Phase 7 evidence and recovery backup."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from inspection.store import Store


def package(target):
    root=Path(__file__).resolve().parents[2]
    report=json.loads((root/'output/phase7/real_validation.json').read_text())
    workspace=Path(report['workspace']).resolve()
    if not workspace.is_relative_to(root):raise ValueError('Validation workspace must remain inside the project')
    store=Store(workspace)
    selected={}
    def add(path,relative=None):
        path=Path(path)
        if path.is_file():selected[path.resolve()]=relative or path.relative_to(root).as_posix()
    for folder in ('inspection','src','scripts','docs','web/src','web/public','data/manifests','configs'):
        for path in (root/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.json','.sha256','.md','.jsx','.css','.svg','.jpg','.png'):
                add(path)
    for name in ('README.md','PRD_Lightweight_Few_Shot_Road_Crack_Segmentation.md','requirements.txt','requirements-app.txt',
                 'requirements-phase7.txt','requirements-phase7.lock.txt','Start_CrackLab.cmd','web/index.html','web/package.json',
                 'web/package-lock.json','web/vite.config.js','output/phase7-tests.xml'):
        add(root/name)
    for path in (root/'output/phase7').rglob('*'):add(path)
    for path in (root/'tests').glob('test_*.py'):add(path)
    add(workspace/'catalog.sqlite3');add(workspace/'frozen_manifest.json')
    for path in (workspace/'source_replay').rglob('*'):add(path)
    for job in report['jobs']:
        folder=workspace/'jobs'/job['id']
        pointer=json.loads((folder/'checkpoints/latest.json').read_text())
        checkpoint=(folder/'checkpoints'/pointer['file']).resolve()
        if checkpoint.parent!=(folder/'checkpoints').resolve() or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=pointer['sha256']:
            raise ValueError('Invalid recovery checkpoint')
        for path in folder.rglob('*'):
            if path.suffix!='.pt':add(path)
        add(checkpoint)
        for vid in (job['parent_id'],job['candidate_id']):
            version=store.version(vid,job['project'])
            model=workspace/'models'/(version['sha256']+'.pt')
            if hashlib.sha256(model.read_bytes()).hexdigest()!=version['sha256']:raise ValueError('Invalid retained model')
            add(model)
    for export in report['onnx_exports']:
        for path in (workspace/'onnx_exports'/export['id']).rglob('*'):add(path)
    inventory={name:hashlib.sha256(path.read_bytes()).hexdigest() for path,name in selected.items()}
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for path,name in sorted(selected.items(),key=lambda item:item[1]):z.write(path,name)
        z.writestr('phase7_inventory.json',json.dumps(inventory,indent=2))
        z.writestr('RESTORE.txt','This is a local Phase 7 evidence/recovery snapshot. Retain original Phase 3-5 ZIPs.\nRestore into a separate project directory; do not overwrite newer work.\nThe validation catalog references Phase 5 models not duplicated here: restore them from the original verified archives using prepare_inspection_models.py.\nNo-replay/replay job snapshots retain only their latest .pt plus pointers; historical files remain in the live validation workspace.\n')
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise ValueError('Backup ZIP integrity failed')
        for name,sha in inventory.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=sha:raise ValueError('Backup inventory mismatch')
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix+'.sha256').write_text(digest+'  '+target.name+'\n')
    print('Verified local backup:',target.name,'files:',len(inventory),'bytes:',target.stat().st_size,'SHA256:',digest)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('phase7_results.zip'))
    args=parser.parse_args();package(args.output)
