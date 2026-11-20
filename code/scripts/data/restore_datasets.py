"""Recover pinned datasets without changing frozen splits or overwriting data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.data.audit_kaggle_data import source_layout
from src.atomic import write_json


def verify_inventory(root,records):
    root=Path(root).resolve()
    for row in records:
        for field in ('image','mask'):
            path=(root/row[field]).resolve()
            if not path.is_relative_to(root): raise ValueError('Manifest path escapes data root')
            if not path.is_file(): raise FileNotFoundError(str(path))
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row[field+'_sha256']:
                raise ValueError(f"Dataset checksum mismatch: {row['image_id']} {field}; original preserved")


def complete(root,records):
    return all((root/r[field]).is_file() for r in records for field in ('image','mask'))


def restore(manifest_path,destination,sources_path,search_roots=()):
    manifest_path=Path(manifest_path)
    expected=manifest_path.with_suffix('.json.sha256').read_text().split()[0]
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Frozen manifest checksum mismatch')
    manifest=json.loads(manifest_path.read_text())
    records=manifest['records']
    destination=Path(destination).resolve()
    candidates=[destination]+[Path(p).resolve() for p in search_roots]
    input_root=Path('/kaggle/input')
    if input_root.exists():
        candidates.extend(p.parent for p in input_root.rglob('DeepCrack') if p.is_dir())
    for candidate in dict.fromkeys(candidates):
        if complete(candidate,records):
            verify_inventory(candidate,records)
            print('Reused existing data; all image/mask checksums passed:',candidate)
            return candidate
    print('Previous data unavailable/incomplete. Restoring pinned repositories to:',destination,flush=True)
    destination.mkdir(parents=True,exist_ok=True)
    sources=json.loads(Path(sources_path).read_text())
    for name,info in sources.items():
        folder=destination/name
        if not folder.exists():
            result=subprocess.run(['git','clone','--no-checkout',info['url'],str(folder)],
                                  capture_output=True,text=True)
            if result.returncode:
                raise RuntimeError(f'{name} download failed. Enable Kaggle Internet or attach the original datasets. '+result.stderr[-2000:])
            subprocess.run(['git','-C',str(folder),'checkout','--detach',info['commit']],check=True)
        elif not (folder/'.git').exists():
            print('Existing non-Git folder preserved:',folder)
        else:
            actual=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
            if actual!=info['commit']:
                raise ValueError(f'{name} is at a different commit; preserve it and choose a new destination')
    source_layout(destination)  # Safely unpack DeepCrack; original archive retained.
    verify_inventory(destination,records)
    print('Restoration complete: verified',len(records),'pairs against the unchanged frozen manifest')
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--search-root',type=Path,action='append',default=[])
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    root=restore(a.manifest,a.destination,a.sources,a.search_root)
    write_json(a.report,{'data_root':str(root),'verified':True,'pairs':len(json.loads(a.manifest.read_text())['records'])})
