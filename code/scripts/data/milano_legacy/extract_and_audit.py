"""Extract publisher splits and audit pairing and basic segmentation compatibility."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import shutil
import zipfile
import numpy as np
from PIL import Image, ImageOps, ImageDraw

base = Path(__file__).resolve().parents[4] / 'data/external/milano_pavement_2026'
root = base / 'extracted'
root.mkdir(exist_ok=True)
archive = base / 'Pavement_Crack_Segmentation_Dataset.zip'
with zipfile.ZipFile(archive) as z:
    needed = sum(i.file_size for i in z.infolist())
    if shutil.disk_usage(base).free < needed + 512 * 1024**2:
        raise SystemExit('Insufficient extraction headroom')
    for member in z.infolist():
        dest = (root / member.filename).resolve()
        if not dest.is_relative_to(root.resolve()):
            raise SystemExit('Unsafe archive path')
    for member in z.infolist():
        target = root / member.filename
        if not member.is_dir() and (not target.exists() or target.stat().st_size != member.file_size):
            z.extract(member, root)
print('Extraction complete', flush=True)
report = {'splits': {}, 'issues': [], 'source': 'https://zenodo.org/records/20040434'}
hashes = defaultdict(list)
records = []
for split in ('train', 'val', 'test'):
    images = {p.stem: p for p in (root/split/'images').glob('*') if p.is_file()}
    masks = {p.stem: p for p in (root/split/'masks').glob('*') if p.is_file()}
    values, modes, dimensions = Counter(), Counter(), Counter()
    empty = 0
    if images.keys() != masks.keys():
        report['issues'].append({'split': split, 'unpaired_images': sorted(images.keys()-masks.keys()), 'unpaired_masks': sorted(masks.keys()-images.keys())})
    for name in sorted(images.keys() & masks.keys()):
        ip, mp = images[name], masks[name]
        try:
            with Image.open(ip) as im, Image.open(mp) as mask:
                im.load(); mask.load()
                if im.size != mask.size:
                    report['issues'].append({'split': split, 'name': name, 'dimension_mismatch': [im.size, mask.size]})
                array = np.asarray(mask.convert('L'))
                unique = np.unique(array)
                values.update(map(int, unique))
                modes[mask.mode] += 1
                dimensions[str(im.size)] += 1
                empty += int(not np.any(array))
            digest = hashlib.sha256(ip.read_bytes()).hexdigest()
            hashes[digest].append(f'{split}/{name}')
            records.append({'split': split, 'image': ip.relative_to(root).as_posix(), 'mask': mp.relative_to(root).as_posix(), 'image_sha256': digest})
        except Exception as exc:
            report['issues'].append({'split': split, 'name': name, 'error': str(exc)})
    report['splits'][split] = {'images': len(images), 'masks': len(masks), 'empty_masks': empty, 'mask_values_image_counts': dict(values), 'mask_modes': dict(modes), 'image_dimensions': dict(dimensions)}
    print(f'{split}: {len(images)} images / {len(masks)} masks audited', flush=True)
report['cross_split_exact_duplicates'] = [v for v in hashes.values() if len({x.split('/')[0] for x in v}) > 1]
report['notes'] = ['Publisher splits preserved. Site independence and near-duplicates have not been verified.', 'Foreground label semantics and vegetation coverage need review before training.', 'No training or model activation performed.']
(base/'dataset_audit.json').write_text(json.dumps(report, indent=2))
(base/'image_manifest.json').write_text(json.dumps(records, indent=2))
# Review training examples only, leaving validation and test visuals unopened.
train = [x for x in records if x['split'] == 'train']
selected = [train[i] for i in np.linspace(0, len(train)-1, min(12, len(train)), dtype=int)]
sheet = Image.new('RGB', (960, 4*190), 'white')
draw = ImageDraw.Draw(sheet)
for idx, row in enumerate(selected):
    with Image.open(root/row['image']) as im, Image.open(root/row['mask']) as mask:
        photo = ImageOps.contain(im.convert('RGB'), (155, 165))
        label = ImageOps.contain(mask.convert('RGB'), (155, 165), Image.Resampling.NEAREST)
        x, y = (idx % 3)*320, (idx//3)*190
        sheet.paste(photo, (x, y+20)); sheet.paste(label, (x+160, y+20))
        draw.text((x+3,y+3), Path(row['image']).stem, fill='black')
sheet.save(base/'training_samples.jpg')
print(json.dumps(report, indent=2), flush=True)
