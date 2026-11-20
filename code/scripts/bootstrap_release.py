"""Import trusted models and bundled source replay data into a fresh workspace."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspection.store import Store
from inspection.replay import verified_buffer
from scripts.prepare_inspection_models import prepare
from src.atomic import atomic_write


def bootstrap(root,workspace):
    prepare(root,workspace)
    store=Store(workspace)
    destination=store.folder('source_replay')
    if (destination/'manifest.json').exists():
        verified_buffer(store);return
    source=root/'release_assets/source_replay'
    if not source.exists():
        raise ValueError('No bundled replay data. Run scripts/prepare_source_replay.py for a source checkout.')
    manifest=json.loads((source/'manifest.json').read_text())
    for row in manifest['records']:
        for field in ('image','mask'):
            name=row[field];original=(source/name).resolve();target=(destination/name).resolve()
            if not original.is_relative_to(source.resolve()) or not target.is_relative_to(destination):raise ValueError('Unsafe asset path')
            content=original.read_bytes()
            if hashlib.sha256(content).hexdigest()!=row[field+'_sha256']:raise ValueError('Bundled source checksum mismatch')
            if target.exists() and target.read_bytes()!=content:raise ValueError('Existing replay file differs')
            if not target.exists():atomic_write(target,lambda stream,data=content:stream.write(data))
    atomic_write(destination/'manifest.json',lambda stream:stream.write((source/'manifest.json').read_bytes()))
    verified_buffer(store)
    print('Verified bundled source replay data imported; no dataset download required.')


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,default=root/'artifacts/phase6/workspace')
    args=parser.parse_args();bootstrap(root,args.workspace)
