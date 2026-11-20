"""Fetch only the pinned DeepCrack archive, then copy hash-verified source rows."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspection.store import Store
from inspection.replay import FROZEN_MANIFEST_SHA256,selected_source,verified_buffer
from src.atomic import atomic_write,write_json


def prepare(workspace,data_root=None):
    root=Path(__file__).resolve().parents[1]
    raw=(root/'data/manifests/pilot_v1.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=FROZEN_MANIFEST_SHA256:
        raise ValueError('Frozen manifest checksum mismatch')
    store=Store(workspace)
    rows=selected_source(json.loads(raw))
    folder=store.folder('source_replay')
    if (folder/'manifest.json').exists():
        verified_buffer(store);print('Existing verified replay buffer retained.');return
    archive=None
    if data_root is None:
        archive=root/'artifacts/phase7/source/DeepCrack_pinned.zip'
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():
            temporary=archive.with_suffix('.download')
            url='https://raw.githubusercontent.com/yhlleo/DeepCrack/8202a60701068645c883ed68b95fd3f30914d90c/dataset/DeepCrack.zip'
            print('Downloading pinned source archive; existing research files remain unchanged.',flush=True)
            with urllib.request.urlopen(url,timeout=60) as response,temporary.open('wb') as stream:
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    stream.write(chunk)
            with zipfile.ZipFile(temporary) as z:
                if z.testzip() is not None:raise ValueError('Source archive integrity failed')
            temporary.replace(archive)
    z=zipfile.ZipFile(archive) if archive else None
    result=[]
    try:
        for row in rows:
            entry=dict(row)
            for field in ('image','mask'):
                if data_root:
                    source=(Path(data_root)/row[field]).resolve()
                    if not source.is_relative_to(Path(data_root).resolve()):raise ValueError('Unsafe source path')
                    content=source.read_bytes()
                else:
                    tail='/'.join(Path(row[field]).parts[-2:])
                    members=[n for n in z.namelist() if n==tail or n.endswith('/'+tail)]
                    if len(members)!=1:raise ValueError('Ambiguous/missing source file '+tail)
                    content=z.read(members[0])
                sha=hashlib.sha256(content).hexdigest()
                if sha!=row[field+'_sha256']:raise ValueError('Source hash differs: '+row['image_id'])
                relative='files/'+sha+Path(row[field]).suffix
                target=folder/relative;target.parent.mkdir(exist_ok=True)
                if target.exists() and target.read_bytes()!=content:raise ValueError('Existing replay file differs')
                if not target.exists():atomic_write(target,lambda stream,data=content:stream.write(data))
                entry[field]=relative
            result.append(entry)
    finally:
        if z:z.close()
    write_json(folder/'manifest.json',{'protocol':'phase7_source_replay_v1','frozen_manifest_sha256':FROZEN_MANIFEST_SHA256,'records':result})
    verified_buffer(store)
    print('Verified 16 source-training replay pairs and 8 separate source-validation pairs.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,default=Path('artifacts/phase6/workspace'))
    parser.add_argument('--data-root',type=Path)
    args=parser.parse_args();prepare(args.workspace,args.data_root)
