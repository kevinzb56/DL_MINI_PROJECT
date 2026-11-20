"""Prepare only approved source/acquisition demo assets; never copy user projects."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from inspection.store import Store
from inspection.replay import verified_buffer
from src.atomic import atomic_write,write_json


def prepare(root):
    root=Path(root)
    assets=root/'release_assets';assets.mkdir(exist_ok=True)
    replay,manifest=verified_buffer(Store(root/'artifacts/phase6/workspace'))
    def copy(source,target):
        data=source.read_bytes()
        if target.exists() and target.read_bytes()!=data:raise ValueError('Existing release asset differs: '+str(target))
        if not target.exists():atomic_write(target,lambda stream:stream.write(data))
    copy(replay/'manifest.json',assets/'source_replay/manifest.json')
    for row in manifest['records']:
        for field in ('image','mask'):copy(replay/row[field],assets/'source_replay'/row[field])
    evidence=json.loads((root/'output/phase7/real_validation.json').read_text())
    frozen=json.loads((root/'data/manifests/pilot_v1.json').read_text())
    selected=[('support',evidence['target_support_ids'][0]),('validation',evidence['target_local_validation_ids'][0])]
    demo=[]
    for role,iid in selected:
        row=next(r for r in frozen['records'] if r['image_id']==iid)
        if row['split']!='target_acquisition':raise ValueError('Demo may use acquisition images only')
        entry={'role':role,'image_id':iid,'split':row['split'],'dataset':'CrackMap'}
        for field in ('image','mask'):
            sha=row[field+'_sha256'];source=root/'artifacts/phase7/validation-inputs'/(sha+Path(row[field]).suffix)
            if hashlib.sha256(source.read_bytes()).hexdigest()!=sha:raise ValueError('Demo source changed')
            name=role+('_mask' if field=='mask' else '')+'.png'
            copy(source,assets/'demo'/name);entry[field]=name;entry[field+'_sha256']=sha
        demo.append(entry)
    write_json(assets/'demo/manifest.json',{'scope':'Real acquisition images, existing public masks; no human correction-time claim. Not query evaluation.',
                                         'records':demo})
    print('Release assets prepared: 24 source pairs and 2 fixed acquisition demo pairs; no user workspace copied.')


if __name__=='__main__':prepare(Path(__file__).resolve().parents[2])
