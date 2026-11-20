"""Audit ZIP-backed publisher pairs; archive-1 pilot is an explicit partial protocol."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
from src.archive_io import check_member, file_sha256
from scripts.data.audit_multisource_splits import audit, TABLE
from scripts.data.download_multisource_cracks import CHUNK_SHA256


def prepare(root, output, full=False):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Manifest already exists; use a new output')
    split_audit = audit(root/'splits.zip')
    with zipfile.ZipFile(root/'splits.zip') as archive:
        table = list(csv.DictReader(io.StringIO(archive.read(TABLE).decode('utf-8-sig'))))
    groups_by_archive = {
        '01_ground_gopro.zip': {'gopro_3ft','gopro_5ft'},
        '02_ground_iphone.zip': {'iphone_3ft','iphone_5ft'},
        '03_uav_low_3_10m.zip': {'uav_3_Meter','uav_6_Meter','uav_10_Meter'},
        '04_uav_mid_15_30m.zip': {f'uav_{h}_Meter' for h in (15,20,25,30)},
        '05_uav_high_35_100m.zip': {f'uav_{h}_Meter' for h in (35,40,45,50,75,100)},
        '06_uav_other.zip': {'uav_All_Heights','uav_Osmo_one','uav_Osmo_two','uav_Osmo_three'},
    }
    names = list(groups_by_archive) if full else ['01_ground_gopro.zip']
    if full and {r['group'] for r in table} != set().union(*groups_by_archive.values()):
        raise ValueError('Official collection groups differ from archive mapping')
    identities = {}
    for name in names:
        identity = file_sha256(root/name)
        if identity != CHUNK_SHA256[name]:
            raise ValueError('Archive SHA256 mismatch: '+name)
        identities[name] = identity
    records = []
    decoded = defaultdict(list)
    for name in names:
        groups = groups_by_archive[name]
        with zipfile.ZipFile(root/name) as archive:
            pairs = defaultdict(dict)
            for member in archive.namelist():
                check_member(member)
                if member.endswith('/'):
                    continue
                path = PurePosixPath(member)
                field = {'Images':'image', 'Masks':'mask'}.get(path.parent.name)
                if field is None:
                    raise ValueError('Unexpected archive member: '+member)
                if field in pairs[path.stem]:
                    raise ValueError('Duplicate pair member: '+path.stem)
                pairs[path.stem][field] = member
            selected = [r for r in table if r['group'] in groups]
            if set(pairs) != {r['image_id'] for r in selected}:
                raise ValueError('Archive IDs do not match official archive groups')
            for official in selected:
                row = dict(image_id=official['image_id'], group=official['group'],
                           publisher_split=official['split'], archive=name,
                           split={'train':'source_train','val':'source_validation','test':'source_holdout'}[official['split']])
                pair = pairs[row['image_id']]
                if set(pair) != {'image','mask'}:
                    raise ValueError('Incomplete pair')
                dimensions = []
                for field in ('image','mask'):
                    member = pair[field]
                    if PurePosixPath(member).parts[0] != official['group']:
                        raise ValueError('Member group mismatch')
                    data = archive.read(member)
                    row[field] = member
                    row[field+'_sha256'] = hashlib.sha256(data).hexdigest()
                    with Image.open(io.BytesIO(data)) as image:
                        image.load()
                        dimensions.append(image.size)
                        if field == 'image':
                            rgb = image.convert('RGB')
                            digest = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                            row['decoded_image_sha256'] = digest
                            decoded[digest].append((row['image_id'],row['publisher_split']))
                        else:
                            raw = np.asarray(image)
                            if raw.ndim == 3:
                                if raw.shape[2] != 3 or not np.all(raw == raw[...,:1]):
                                    raise ValueError('Colored mask requires explicit semantics')
                                raw = raw[...,0]
                            if raw.ndim != 2 or not set(np.unique(raw).tolist()) <= {0,255}:
                                raise ValueError('Nonbinary mask: '+member)
                            row['foreground_pixels'] = int(np.count_nonzero(raw))
                            if row['foreground_pixels'] != int(official['crack_pixels']):
                                raise ValueError('Mask foreground differs from publisher metadata')
                expected = (int(official['image_width']),int(official['image_height']))
                if dimensions != [expected,expected]:
                    raise ValueError('Image/mask dimension mismatch')
                records.append(row)
                if len(records)%50 == 0:
                    print('Audited pairs:',len(records),flush=True)
    overlap = [v for v in decoded.values() if len({split for _,split in v}) > 1]
    if overlap:
        raise ValueError('Decoded cross-split duplicate images detected')
    if full and {r['image_id'] for r in records} != {r['image_id'] for r in table}:
        raise ValueError('Incomplete full-dataset coverage')
    manifest = dict(status='AUDITED_MULTISOURCE_V1' if full else 'FROZEN_MULTISOURCE_ARCHIVE1_PILOT', schema_version=1,
                    protocol='multisource_v1_pending_training_freeze' if full else 'multisource_archive1_pilot', archives=identities,
                    splits_sha256=file_sha256(root/'splits.zip'), records=records,
                    scope='Full dataset audited; training protocol requires separate freeze' if full else 'Partial dataset pipeline pilot; no test evaluation; not final benchmark',
                    split_audit=split_audit, decoded_cross_split_duplicates=0)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    output.with_suffix('.json.sha256').write_text(file_sha256(output)+'\n')
    print(json.dumps(dict(state='prepared',pairs=len(records),splits=dict(Counter(r['publisher_split'] for r in records)))),flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full',action='store_true',help='Require and audit all six archives')
    args = parser.parse_args()
    prepare(args.root,args.output,full=args.full)
