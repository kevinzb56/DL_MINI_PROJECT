"""Freeze reviewed audit evidence into a portable, checksummed pilot protocol."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.atomic import write_json


def freeze(audit_directory, output):
    audit_directory, output = Path(audit_directory), Path(output)
    if output.exists():
        raise ValueError('Frozen protocol exists; choose a new version')
    source = audit_directory/'candidate_manifest.json'
    candidate = json.loads(source.read_text())
    if candidate['errors']:
        raise ValueError('Audit has unresolved errors')
    records=[]
    ids, hashes = set(), set()
    groups=defaultdict(set)
    captures=defaultdict(set)
    for original in candidate['records']:
        row=dict(original)
        key=original['image_id'].split(':',1)[1]
        identity=f"{row['dataset']}:{row['partition']}:{key}"
        if identity in ids or row['pixel_sha256'] in hashes:
            raise ValueError('Duplicate canonical ID or decoded image')
        ids.add(identity); hashes.add(row['pixel_sha256'])
        groups[row['group_id']].add(row['split'])
        if row['dataset']=='CrackMap':
            match=re.fullmatch(r'(GOPR\d+)(?: \(\d+\))?',key)
            if not match: raise ValueError('Unexpected CrackMap capture naming')
            captures[match.group(1)].add(row['split'])
        derived=audit_directory/'derived_masks'/row['dataset']/row['partition']/(key+'.png')
        if hashlib.sha256(derived.read_bytes()).hexdigest()!=row['mask_sha256']:
            raise ValueError('Derived mask checksum mismatch')
        def relative_raw(path):
            marker='/data/raw/'
            if marker not in path: raise ValueError('Unknown raw path layout')
            return path.split(marker,1)[1]
        records.append({
            'image_id':identity,'original_id':original['image_id'],
            'dataset':row['dataset'],'partition':row['partition'],'split':row['split'],
            'group_id':row['group_id'],'group_provenance':row['group_provenance'],
            'image':relative_raw(row['image']), 'mask':relative_raw(row['raw_mask']),
            'image_sha256':row['image_sha256'],'mask_sha256':row['raw_mask_sha256'],
            'pixel_sha256':row['pixel_sha256'],'width':row['width'],'height':row['height'],
            'foreground_fraction':row['foreground_fraction'],'mask_values':row['mask_values']})
    if any(len(v)!=1 for v in groups.values()) or any(len(v)!=1 for v in captures.values()):
        raise ValueError('Group/capture leakage')
    counts=dict(sorted(Counter(r['split'] for r in records).items()))
    expected={'source_train','source_validation','source_holdout','target_acquisition','target_query'}
    if set(counts)!=expected:
        raise ValueError('Missing required splits')
    # A shot is one image, with a distinct GOPR capture per shot in each draw.
    support=defaultdict(list)
    for row in records:
        if row['split']=='target_acquisition': support[row['group_id']].append(row['image_id'])
    if len(support)<10: raise ValueError('Fewer than ten support capture groups')
    draws=[]
    for draw in range(5):
        rng=random.Random(42+draw)
        selected=rng.sample(sorted(support),10)
        images=[rng.choice(sorted(support[group])) for group in selected]
        draws.append({'draw':draw,'seed':42+draw,'supports':{str(k):images[:k] for k in (1,5,10)}})
    result={'schema_version':1,'protocol':'deepcrack_crackmap_pilot_v1',
        'status':'FROZEN_PILOT','seed':42,'path_base':'data/raw',
        'counts':counts,'records':records,'support_draws':draws,
        'audit_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'mask_policy':'White foreground; binary 0/255; RGB labels require identical channels.',
        'review':'24 contact-sheet examples inspected locally; not every pair visually reviewed.',
        'limitations':[
            'Pilot uses DeepCrack only as source, not the reference paper five-source protocol.',
            'CrackMap GOPR filename captures stay in one split; road/site/sequence independence is unverified.',
            'DeepCrack official train/test retained; no verified capture groups beyond similarity screening.',
            'dHash screening found no matches but cannot rule out related captures.',
            'CrackMap license terms remain unconfirmed; protocol freezing is not license clearance.',
            'Ten reused DeepCrack filename IDs qualified by native partition; images have distinct hashes.',
            'Raw images remain on Kaggle; local artifact checks verify exported evidence, not a full image re-audit.',
            'No model training or GPU trainer recovery has run.']}
    write_json(output,result)
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.json.sha256').write_text(digest+'  '+output.name+'\n')
    print(json.dumps({'manifest':str(output),'sha256':digest,'counts':counts},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit_directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    freeze(args.audit_directory,args.output)
