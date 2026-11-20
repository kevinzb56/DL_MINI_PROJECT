"""Prepare an isolated, checksummed Milano copy; never alter publisher files."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(mask):
    # Explicit luminance policy; alpha must not hide any non-black annotation.
    if mask.mode == 'RGBA':
        a = np.asarray(mask)
        if np.any((a[..., 3] < 255) & np.any(a[..., :3] != 0, axis=2)):
            raise ValueError('Non-black transparent mask pixels require review')
    raw = np.asarray(mask.convert('L'))
    return raw, ((raw >= 128).astype(np.uint8) * 255)


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Output exists; use a new version, preserving prior preparation')
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Source and prepared copy must be separate trees')
    pairs = []
    for split in ('train', 'val', 'test'):
        groups = []
        for kind in ('images', 'masks'):
            files = sorted(p for p in (source/split/kind).iterdir() if p.is_file())
            mapping = {p.stem: p for p in files}
            if len(files) != len(mapping):
                raise ValueError('Duplicate stems')
            groups.append(mapping)
        if groups[0].keys() != groups[1].keys() or not groups[0]:
            raise ValueError('Missing pairs: ' + split)
        pairs.extend((split, name, groups[0][name], groups[1][name]) for name in sorted(groups[0]))
    required = sum(p.stat().st_size for _, _, ip, mp in pairs for p in (ip, mp))
    if shutil.disk_usage(output.parent).free < required + 1024**3:
        raise ValueError('Insufficient disk headroom')
    output.mkdir()
    records, stats, fingerprints = [], {}, []
    decoded = defaultdict(list)
    for split, name, ip, mp in pairs:
        tally = stats.setdefault(split, Counter())
        (output/split/'images').mkdir(parents=True, exist_ok=True)
        (output/split/'masks').mkdir(parents=True, exist_ok=True)
        image_path = output/split/'images'/ip.name
        mask_path = output/split/'masks'/(name+'.png')
        with Image.open(ip) as im, Image.open(mp) as mask:
            im.load(); mask.load()
            if im.size != mask.size:
                raise ValueError('Misaligned dimensions: ' + name)
            rgb = im.convert('RGB')
            raw, binary = normalize(mask)
            digest = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            identity = split+'/'+name
            decoded[digest].append(identity)
            # Average hash is a screening heuristic, never proof of site independence.
            thumb = np.asarray(rgb.convert('L').resize((8, 8), Image.Resampling.LANCZOS))
            bits = int.from_bytes(np.packbits(thumb >= thumb.mean()).tobytes(), 'big')
            fingerprints.append((identity, split, bits))
            Image.fromarray(binary).save(mask_path)
            tally.update(pairs=1, empty_masks=int(not binary.any()), foreground_pixels=int((binary>0).sum()),
                         pixels=int(binary.size), intermediate_pixels=int(((raw>0)&(raw<255)).sum()),
                         changed_pixels=int((raw!=binary).sum()))
        shutil.copyfile(ip, image_path)
        records.append({'image_id': 'milano/'+identity,
                        'split': {'train':'source_train','val':'source_validation','test':'source_holdout'}[split],
                        'publisher_split':split, 'image':image_path.relative_to(output).as_posix(),
                        'mask':mask_path.relative_to(output).as_posix(),
                        'image_sha256':sha(image_path),'mask_sha256':sha(mask_path),
                        'raw_mask_sha256':sha(mp),'decoded_image_sha256':digest})
        if len(records)%250==0:
            print(f'Prepared {len(records)}/{len(pairs)}', flush=True)
    exact = [v for v in decoded.values() if len({x.split('/')[0] for x in v})>1]
    similar=[]
    for i, (name, split, bits) in enumerate(fingerprints):
        for other, other_split, other_bits in fingerprints[i+1:]:
            if split != other_split:
                distance=(bits^other_bits).bit_count()
                if distance<=3:
                    similar.append({'a':name,'b':other,'average_hash_distance':distance})
    manifest={'status':'FROZEN_MILANO_V1','schema_version':1,
              'protocol':'milano_v1_publisher_splits','source':'https://zenodo.org/records/20040434',
              'normalization':'Pillow L luminance >=128 ->255; otherwise 0; no resizing',
              'label_scope':'cracks and pavement distress; publisher splits; site independence unverified',
              'records':records}
    report={'state':'blocked_exact_overlap' if exact else 'prepared', 'splits':stats,
            'decoded_cross_split_duplicates':exact,'similarity_candidates':similar,
            'similarity_policy':'64-bit average hash distance <=3; candidates only, no automatic removal',
            'limitations':['No site/capture metadata verified','Similarity candidates need review',
                           'No road-region masks; non-road false-positive area cannot be measured',
                           'Training-only visuals inspected; test performance remains unopened']}
    for filename, value in [('manifest.json',manifest),('preparation_audit.json',report)]:
        path=output/filename
        path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
        path.with_suffix(path.suffix+'.sha256').write_text(sha(path)+'  '+filename+'\n')
    train=[r for r in records if r['publisher_split']=='train']
    sheet=Image.new('RGB',(960,12*180),'white'); draw=ImageDraw.Draw(sheet)
    for j,index in enumerate(np.linspace(0,len(train)-1,12,dtype=int)):
        row=train[index]
        with Image.open(output/row['image']) as im, Image.open(output/row['mask']) as m:
            photo=im.convert('RGB'); overlay=Image.composite(Image.blend(photo,Image.new('RGB',photo.size,'red'),.6),photo,m)
            for c,view in enumerate((photo,m.convert('RGB'),overlay)):
                sheet.paste(ImageOps.contain(view,(315,155)),(c*320,j*180+22))
        draw.text((5,j*180+3),row['image_id']+' | image / normalized mask / overlay',fill='black')
    sheet.save(output/'training_review.jpg',quality=92)
    print(json.dumps({**report,'similarity_candidates':len(similar)},indent=2),flush=True)
    if exact:
        raise ValueError('Exact cross-split overlap: training blocked')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/external/milano_pavement_2026/extracted'))
    p.add_argument('--output',type=Path,default=Path('data/external/milano_prepared_v1'))
    a=p.parse_args(); prepare(a.source,a.output)
