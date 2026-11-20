"""Inventory real Kaggle data and produce reviewable candidate splits.

Never invent sequence metadata: similarity clusters are only proxy groups.
No manifest is frozen until annotation polarity and independence are reviewed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.atomic import write_json

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def filename_group(dataset, key):
    # Group numbered patches sharing a GOPR identifier conservatively.
    if dataset == 'CrackMap':
        match = re.fullmatch(r'(GOPR\d+)(?: \(\d+\))?', key)
        if match:
            return 'filename_capture_' + match.group(1)
    return None


def source_layout(data_root):
    directory = Path(data_root)/'DeepCrack/dataset'
    archive = directory/'DeepCrack.zip'
    def locate():
        return [p for p in directory.rglob('train_img') if p.is_dir()
                and all((p.parent/name).is_dir() for name in ('train_lab','test_img','test_lab'))]
    candidates = locate() if directory.exists() else []
    if not candidates and archive.exists():
        destination = directory/'extracted_v1'
        destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                target = (destination/entry.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError('Unsafe DeepCrack ZIP path')
                if ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError('Symlink in DeepCrack ZIP')
                if not entry.is_dir() and target.exists() and target.read_bytes()!=z.read(entry):
                    raise ValueError('Existing extracted dataset differs; preserve and inspect')
            z.extractall(destination)
        candidates = locate()
    if len(candidates)!=1:
        raise ValueError(f'Expected one complete DeepCrack layout, found {len(candidates)}')
    return candidates[0].parent


def file_map(directory):
    if not directory.is_dir():
        raise ValueError(f'Missing dataset directory: {directory}')
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            # Only remove documented annotation suffixes; report all ambiguity.
            key = path.stem
            for suffix in ('_mask', '_label', '_lab'):
                if key.endswith(suffix):
                    key = key[:-len(suffix)]
                    break
            if key in result:
                raise ValueError(f'Ambiguous pair key {key}: {result[key]} / {path}')
            result[key] = path
    return result


def paired_files(images, masks):
    left, right = file_map(images), file_map(masks)
    if not left or set(left) != set(right):
        raise ValueError(f'Unpaired/empty folders: {images}; '
                         f'images_without_masks={sorted(set(left)-set(right))[:20]}, '
                         f'masks_without_images={sorted(set(right)-set(left))[:20]}')
    return [(key, left[key], right[key]) for key in sorted(left)]


def audit(data_root, output, seed=42, threshold=None):
    import numpy as np
    from PIL import Image, ImageDraw
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Refusing to overwrite audit evidence; choose a new output directory')
    output.mkdir(parents=True)
    layouts = [
        ('DeepCrack', 'source_train', 'DeepCrack/dataset/train_img', 'DeepCrack/dataset/train_lab'),
        ('DeepCrack', 'source_test', 'DeepCrack/dataset/test_img', 'DeepCrack/dataset/test_lab'),
        ('CrackMap', 'target', 'CrackMap/images', 'CrackMap/masks'),
    ]
    records, errors, conversions = [], [], []
    try:
        source = source_layout(data_root)
        layouts[:2] = [
            ('DeepCrack','source_train',source/'train_img',source/'train_lab'),
            ('DeepCrack','source_test',source/'test_img',source/'test_lab')]
    except Exception as error:
        errors.append(str(error))
    derived = output / 'derived_masks'
    for dataset, partition, image_dir, mask_dir in layouts:
        try:
            pairs = paired_files(Path(data_root)/image_dir, Path(data_root)/mask_dir)
        except Exception as error:
            errors.append(str(error))
            continue
        for key, image_path, mask_path in pairs:
            try:
                with Image.open(image_path) as source:
                    image = source.convert('RGB')
                    image.load()
                with Image.open(mask_path) as source:
                    raw = np.asarray(source)
                if raw.ndim == 3:
                    if raw.shape[2] not in (3, 4) or not np.all(raw[..., 0] == raw[..., 1]) or not np.all(raw[..., 0] == raw[..., 2]):
                        raise ValueError('Colored annotations need an explicit label mapping')
                    if raw.shape[2] == 4 and not np.all(raw[..., 3] == 255):
                        raise ValueError('Transparent annotations need an explicit label mapping')
                    raw = raw[..., 0]
                if raw.ndim != 2 or image.size != (raw.shape[1], raw.shape[0]):
                    raise ValueError('Mask dimensions do not match image')
                values = np.unique(raw)
                if set(values.tolist()) <= {0, 1, 255}:
                    binary = (raw > 0).astype(np.uint8)*255
                    rule = 'positive label values become 255; zero becomes 0'
                elif threshold is not None:
                    binary = (raw >= threshold).astype(np.uint8)*255
                    rule = f'explicit mask threshold >= {threshold}'
                else:
                    raise ValueError(f'Nonbinary mask ({len(values)} values); review before explicit thresholding')
                destination = derived/dataset/partition/f'{key}.png'
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(binary).save(destination)
                small = np.asarray(image.convert('L').resize((9,8), Image.Resampling.LANCZOS))
                dhash = int(''.join('1' if b else '0' for b in (small[:,1:] > small[:,:-1]).flat),2)
                records.append({'image_id': dataset+':'+key, 'dataset':dataset, 'partition':partition,
                    'image':str(image_path.resolve()), 'mask':str(destination),
                    'raw_mask':str(mask_path.resolve()), 'mask_values':values.tolist(),
                    'image_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    'raw_mask_sha256':hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                    'mask_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
                    'pixel_sha256':hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest(),
                    'dhash':f'{dhash:016x}', 'width':image.width, 'height':image.height,
                    'foreground_fraction':float((binary>0).mean())})
                conversions.append({'image_id':dataset+':'+key,'rule':rule})
            except Exception as error:
                errors.append(f'{dataset}/{partition}/{key}: {error}')

    # Similarity components keep detected related images together. They do not
    # establish road/sequence independence and require human review.
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    matches = []
    known_groups = {}
    for i, row in enumerate(records):
        group = filename_group(row['dataset'], row['image_id'].split(':',1)[1])
        if group:
            if group in known_groups:
                parent[find(i)] = find(known_groups[group])
            known_groups[group] = i
        row['filename_group'] = group
    for i, row in enumerate(records):
        for j in range(i):
            other = records[j]
            exact = row['pixel_sha256'] == other['pixel_sha256']
            distance = (int(row['dhash'],16)^int(other['dhash'],16)).bit_count()
            if exact or distance <= 3:
                parent[find(i)] = find(j)
                matches.append({'a':row['image_id'],'b':other['image_id'],
                                'exact':exact,'dhash_distance':distance})
                if row['partition'] != other['partition']:
                    errors.append(f"Similarity crosses native partitions: {row['image_id']} / {other['image_id']}")
    for i,row in enumerate(records):
        row['group_id'] = 'similarity_proxy_' + str(find(i))
        row['group_provenance'] = 'GOPR filename capture + decoded hash + dHash<=3; site/sequence independence NOT verified'
    rng = random.Random(seed)
    allocations = {}
    for partition, selected, remaining, fraction in [
        ('source_train','source_validation','source_train',.2),
        ('target','target_acquisition','target_query',.25)]:
        groups = sorted({r['group_id'] for r in records if r['partition']==partition})
        if len(groups)<2:
            errors.append(f'Not enough independent candidate groups for {partition}')
            continue
        rng.shuffle(groups)
        n = max(1,min(len(groups)-1,round(len(groups)*fraction)))
        allocations.update({(partition,g):selected if i<n else remaining for i,g in enumerate(groups)})
    for row in records:
        row['split'] = ('source_holdout' if row['partition']=='source_test' else
                        allocations.get((row['partition'],row['group_id']),'unassigned'))
    counts = {s:sum(r['split']==s for r in records) for s in sorted({r['split'] for r in records})}
    manifest = {'schema_version':1,'status':'CANDIDATE_NOT_FROZEN','seed':seed,
                'counts':counts,'records':records,'errors':errors,
                'limitations':['White=crack polarity awaits contact-sheet review.',
                               'GOPR filename captures stay together; other similarity groups are proxies, not verified parent frames.',
                               'Near-duplicate detection is heuristic and may miss related captures.']}
    write_json(output/'candidate_manifest.json',manifest)
    write_json(output/'similarity_matches.json',matches)
    write_json(output/'mask_conversions.json',conversions)
    if records:
        with (output/'candidate_index.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['image_id','image','mask','dataset','group_id','partition'])
            writer.writeheader()
            writer.writerows({key:r[key] for key in writer.fieldnames} for r in records)
        # Up to eight examples per native partition, sampled deterministically.
        chosen=[]
        for partition in ('source_train','source_test','target'):
            pool=[r for r in records if r['partition']==partition]
            chosen.extend(random.Random(seed).sample(pool,min(8,len(pool))))
        sheet=Image.new('RGB',(768, len(chosen)*155),'white')
        draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(chosen):
            image=Image.open(row['image']).convert('RGB').resize((256,128))
            mask=Image.open(row['mask']).convert('L').resize((256,128),Image.Resampling.NEAREST)
            overlay=Image.blend(image,Image.new('RGB',image.size,'red'),.55)
            overlay=Image.composite(overlay,image,mask)
            for x,view in enumerate((image,mask.convert('RGB'),overlay)):
                sheet.paste(view,(x*256,i*155+22))
            draw.text((2,i*155+3),row['image_id']+' | original / mask / overlay',fill='black')
        sheet.save(output/'contact_sheet.jpg',quality=90)
    # This inventory documents the real folders even if pairing failed.
    write_json(output/'inventory.json',[
        str(p.relative_to(data_root)) for p in sorted(Path(data_root).rglob('*'))
        if p.is_file() and '.git' not in p.parts])
    report={'status':'review_required','valid_pairs':len(records),'counts':counts,
            'error_count':len(errors),'errors':errors,'similarity_matches':len(matches),
            'frozen':False,'training_started':False}
    write_json(output/'audit_report.json',report)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--mask-threshold',type=int)
    args=parser.parse_args()
    if args.mask_threshold is not None and not 1<=args.mask_threshold<=255:
        parser.error('mask threshold must be between 1 and 255')
    audit(args.data_root,args.output,args.seed,args.mask_threshold)
