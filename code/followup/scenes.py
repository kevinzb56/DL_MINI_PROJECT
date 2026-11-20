"""Audited external scenes and paired broad-scene metrics, with no implied labels."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from followup.adaptation import sha, region
from src.atomic import write_json
from src.model_factory import build_model


def inside(root, name):
    path = (Path(root)/name).resolve()
    if not path.is_relative_to(Path(root).resolve()): raise ValueError('Scene path escapes root')
    return path


def binary(path):
    with Image.open(path) as image: array = np.asarray(image)
    if array.ndim != 2 or not set(np.unique(array)).issubset({0, 255}):
        raise ValueError('Scene labels must be binary single-channel 0/255')
    return array > 0


def audit_scenes(path, root, pilot):
    data = json.loads(Path(path).read_text())
    if data.get('protocol') != 'followup_external_scenes_v1' or not data.get('records'):
        raise ValueError('Expected nonempty external scene manifest')
    if not data.get('usage_terms') or not data.get('physical_site_provenance'):
        raise ValueError('Usage terms and physical-site provenance required')
    if any('REPLACE' in data[k] for k in ('usage_terms','physical_site_provenance')):
        raise ValueError('Replace template provenance with actual collection/permission records')
    known = {r['image_sha256'] for r in pilot['records']}
    seen = set(); hashes = set(); sites = {}; groups = {}
    for row in data['records']:
        if row['split'] not in ('scene_train', 'scene_validation', 'scene_test'):
            raise ValueError('Unknown scene split')
        if row['image_id'] in seen or row['image_sha256'] in hashes:
            raise ValueError('Duplicate scene image')
        seen.add(row['image_id']); hashes.add(row['image_sha256'])
        if row['image_sha256'] in known: raise ValueError('Previously inspected pilot image is not independent')
        if row.get('annotation_complete') is not True or row.get('annotation_origin') not in ('human', 'existing_dataset'):
            raise ValueError('Complete real annotations required; predictions are not labels')
        for key, ledger in [('site_id', sites), ('group_id', groups)]:
            if not row.get(key): raise ValueError('Physical site and capture group required')
            if row[key] in ledger and ledger[row[key]] != row['split']:
                raise ValueError('Site/capture leakage across scene partitions')
            ledger[row[key]] = row['split']
        for field in ('image', 'mask', 'road_mask'):
            asset = inside(root, row[field])
            if sha(asset) != row[field+'_sha256']: raise ValueError('Scene checksum mismatch')
        with Image.open(inside(root, row['image'])) as image: size = image.size
        mask = binary(inside(root, row['mask'])); road = binary(inside(root, row['road_mask']))
        if mask.shape != (size[1], size[0]) or road.shape != mask.shape:
            raise ValueError('Image and masks must align at original resolution')
        if np.any(mask & ~road): raise ValueError('Crack annotation outside annotated road region')
    return data


def mask_metrics(predicted, truth, road):
    tp = int((predicted & truth).sum()); fp = int((predicted & ~truth).sum()); fn = int((~predicted & truth).sum())
    return {'dice': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            'iou': tp/(tp+fp+fn) if tp+fp+fn else 1.,
            'recall': tp/(tp+fn) if tp+fn else None,
            'nonroad_fp_fraction': float(predicted[~road].mean()) if (~road).any() else None,
            'positive': bool(truth.any())}


def require_scene_size(data,requirements):
    for split,minimum in requirements.items():
        rows=[r for r in data['records'] if r['split']==split]
        if len(rows)<minimum['images'] or len({r['site_id'] for r in rows})<minimum['sites']:
            raise ValueError(f"Insufficient {split}: need {minimum['images']} images from {minimum['sites']} sites")


@torch.no_grad()
def score_scenes(state, data, root, split, device='cpu', mode='resize'):
    from inspection.advanced import tiled_probability
    config = state['config'].get('model_config', state['config'])
    model = build_model(config).to(device).eval(); model.load_state_dict(state['model'])
    rows = []
    for row in data['records']:
        if row['split'] != split: continue
        with Image.open(inside(root, row['image'])) as image: image = image.convert('RGB')
        if mode == 'tiles':
            if str(device) != 'cpu': raise ValueError('Audited tiled pipeline currently CPU only')
            from inspection.engine import probability as native_probability
            probability = tiled_probability(lambda tile: native_probability(model,tile,config['image_size']),
                                            image, config['image_size'], min(64,config['image_size']//4))[0]
        elif mode == 'resize':
            small = np.array(image.resize((config['image_size'],)*2, Image.Resampling.BILINEAR), dtype=np.float32)/255
            x = torch.from_numpy(small.transpose(2,0,1)).unsqueeze(0).to(device)
            probability = region(model(x)).sigmoid()[0,0].cpu().numpy()
            probability = np.asarray(Image.fromarray(probability).resize(image.size, Image.Resampling.BILINEAR))
        else: raise ValueError('Unknown sampling mode')
        if not np.isfinite(probability).all(): raise ValueError('Nonfinite scene prediction')
        rows.append({'image_id': row['image_id'], 'site_id': row['site_id'], 'group_id': row['group_id'],
                     **mask_metrics(probability >= .5, binary(inside(root,row['mask'])), binary(inside(root,row['road_mask'])))})
    if not rows: raise ValueError('No scenes for '+split)
    summary = {}
    for key in ('dice','iou','recall','nonroad_fp_fraction'):
        values = [r[key] for r in rows if r[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
        summary[key+'_n'] = len(values)
    return {'summary': summary, 'per_image': rows, 'n': len(rows), 'sampling': mode}


def scene_gate(parent, candidate, gates):
    p, c = parent['summary'], candidate['summary']
    assessable = (p['nonroad_fp_fraction'] is not None and p['nonroad_fp_fraction'] > 0
                  and p['recall'] is not None and c['recall'] is not None)
    reduction = 1-c['nonroad_fp_fraction']/p['nonroad_fp_fraction'] if assessable else None
    passed = bool(assessable and reduction >= gates['min_relative_nonroad_fp_reduction']
                  and c['recall'] >= p['recall']-gates['max_recall_drop_points']/100
                  and c['dice'] >= p['dice']-gates['max_dice_drop_points']/100)
    return {'passed': passed, 'assessable': assessable, 'relative_nonroad_fp_reduction': reduction,
            'recall_change_points': 100*(c['recall']-p['recall']) if assessable else None,
            'scope': 'Paired images, positive-image macro recall; no road truth is supplied to inference.'}
