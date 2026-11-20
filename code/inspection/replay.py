"""Source-only replay snapshots; source validation never enters the optimizer."""
import hashlib
import json
from pathlib import Path
import random
from inspection.advanced import CONFIG
from src.dataset import verify_files

FROZEN_MANIFEST_SHA256='9c97554166af5494c746d46885425f748ea46c4035863c22c90215b3506358ae'


def selected_source(manifest):
    rng=random.Random(CONFIG['replay']['seed'])
    rows=[]
    for split,count in [('source_train',CONFIG['replay']['source_train_images']),('source_validation',CONFIG['replay']['source_validation_images'])]:
        candidates=sorted((r for r in manifest['records'] if r['split']==split),key=lambda r:r['image_id'])
        rows.extend(rng.sample(candidates,count))
    return rows


def verified_buffer(store):
    folder=store.root/'source_replay'
    path=folder/'manifest.json'
    if not path.exists():
        raise ValueError('Prepare verified source replay data with scripts/prepare_source_replay.py first')
    frozen=(store.root/'frozen_manifest.json').read_bytes()
    # The imported file is pretty-printed; the canonical frozen bytes live in the repository.
    canonical=Path(__file__).resolve().parents[1]/'data/manifests/pilot_v1.json'
    if hashlib.sha256(canonical.read_bytes()).hexdigest()!=FROZEN_MANIFEST_SHA256:
        raise ValueError('Frozen research manifest changed')
    manifest=json.loads(canonical.read_text())
    if json.loads(frozen)!=manifest:
        raise ValueError('Workspace manifest differs from frozen research manifest')
    buffer=json.loads(path.read_text())
    expected=selected_source(manifest)
    rows=buffer['records']
    if buffer['protocol']!='phase7_source_replay_v1' or [r['image_id'] for r in rows]!=[r['image_id'] for r in expected]:
        raise ValueError('Replay selection differs from the frozen Phase 7 selection')
    for row,original in zip(rows,expected):
        for field in ('split','image_sha256','mask_sha256'):
            if row[field]!=original[field]:
                raise ValueError('Replay split or checksum identity changed')
    verify_files(rows,folder)
    return folder,buffer


def status(store):
    try:
        _,buffer=verified_buffer(store)
        return {'ready':True,'source_train_images':sum(r['split']=='source_train' for r in buffer['records']),
                'source_validation_images':sum(r['split']=='source_validation' for r in buffer['records']),
                'seed':CONFIG['replay']['seed'],'scope':'Fixed source-training replay buffer; separate source-validation subset. Not full source holdout.'}
    except (ValueError,OSError,KeyError) as error:
        return {'ready':False,'detail':str(error),'source_train_images':0,'source_validation_images':0}
