"""Compare every fixed Phase 3-5 reference on one saved inspection image.

No training, checkpoint selection, activation, or research-score updates occur.
Coverage is a diagnostic count, not accuracy without a ground-truth mask.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from inspection.engine import infer, render, save_png
from inspection.store import Store


def compare(workspace, image_id, output):
    store = Store(workspace)
    record = store.get('image', image_id)
    path = store.root / 'images' / image_id / 'original.png'
    image = Image.open(path).convert('RGB')
    output.mkdir(parents=True, exist_ok=False)
    save_png(output / 'original.png', image)
    versions = [v for v in store.list('version') if v.get('phase') in (3, 4, 5) and v.get('project') is None]
    rows = []
    for version in versions:
        probability, config, elapsed = infer(store, version, image)
        views = render(image, probability, .5, .5)
        name = ('region_base' if version['phase'] == 3 else 'boundary_base') if version['phase'] != 5 else version['label'].replace(' · ', '_')
        name = name.replace(' ', '_')
        folder = output / name
        folder.mkdir()
        for view, rendered in views.items():
            save_png(folder / (view + '.png'), rendered)
        np.save(folder / 'probability.npy', probability, allow_pickle=False)
        mask = probability >= .5
        row = {'label': version['label'], 'version_id': version['id'], 'checkpoint_sha256': version['sha256'],
               'phase': version['phase'], 'output': name, 'crack_pixels': int(mask.sum()),
               'predicted_area_percent': float(mask.mean()*100),
               'upper_45_percent_predicted_fraction': float(mask[:round(image.height*.45)].mean()*100),
               'preprocessing_size': config['image_size'], 'latency_ms': elapsed}
        if version['phase'] == 5:
            group, run = version['label'].split(' · ')
            draw, k = run.removeprefix('draw').split('_k')
            row.update(model=group, draw=int(draw), k=int(k))
        else:
            row.update(model='control' if version['phase'] == 3 else 'boundary', draw=None, k=0)
        rows.append(row)
        print(name, f'predicted area={row["predicted_area_percent"]:.2f}%', flush=True)
    summaries = []
    for model in ('control', 'boundary'):
        for k in (0, 1, 5, 10):
            group = [r for r in rows if r['model'] == model and r['k'] == k]
            expected = 1 if k == 0 else 5
            if len(group) != expected:
                raise ValueError(f'Expected {expected} references for {model}, K={k}; found {len(group)}')
            summaries.append({'model': model, 'k': k, 'runs': len(group),
                              'mean_predicted_area_percent': statistics.mean(r['predicted_area_percent'] for r in group),
                              'min_predicted_area_percent': min(r['predicted_area_percent'] for r in group),
                              'max_predicted_area_percent': max(r['predicted_area_percent'] for r in group),
                              'mean_upper_region_predicted_percent': statistics.mean(r['upper_45_percent_predicted_fraction'] for r in group)})
    try:
        heading = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
        small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 13)
    except OSError:
        heading = small = ImageFont.load_default()

    def sheet(selected, filename, title, columns=4):
        cell_w, cell_h = 260, 410
        canvas = Image.new('RGB', (columns*cell_w, 75 + ((len(selected)+columns-1)//columns)*cell_h), '#11181c')
        draw = ImageDraw.Draw(canvas)
        draw.text((15, 12), title, fill='#f7b282', font=heading)
        draw.text((15, 40), 'Same original image; threshold 0.50, opacity 0.50. No ground truth: visual comparison only.', fill='#b4c5cc', font=small)
        for index, row in enumerate(selected):
            x = (index % columns)*cell_w; y = 75 + (index//columns)*cell_h
            draw.text((x+10,y+4), row.get('label','Original'), fill='#eeeeee', font=heading)
            if row.get('output'):
                content = Image.open(output / row['output'] / 'overlay.png').convert('RGB')
                note = f'Predicted image area: {row["predicted_area_percent"]:.1f}%'
            else:
                content = image.copy(); note = '480 x 640; no segmentation'
            content.thumbnail((240, 340))
            canvas.paste(content, (x+10, y+32))
            draw.text((x+10,y+365), note, fill='#a8bbc6', font=small)
        save_png(output / filename, canvas)

    for model in ('control', 'boundary'):
        selected = sorted((r for r in rows if r['model'] == model), key=lambda r:(r['k'], -1 if r['draw'] is None else r['draw']))
        sheet(selected, model + '_all_draws.png', f'{model}: base plus every frozen 1/5/10-shot adaptation')
    representative = [{'label':'Original image'}]
    representative.extend(r for r in rows if r['k'] == 0)
    representative.extend(sorted((r for r in rows if r['model']=='boundary' and r['draw']==0), key=lambda r:r['k']))
    sheet(representative, 'comparison.png', 'Original, both bases, and boundary adaptations (draw 0 fixed in advance)', columns=3)
    report = {'image': record, 'normalized_image_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'threshold': .5, 'opacity': .5, 'models_compared': len(rows), 'results': rows, 'summaries': summaries,
              'limits': ['No ground-truth mask: no Dice, IoU, precision or recall calculated.',
                         'Predicted area is not accuracy; less predicted area can also mean more missed cracks.',
                         'Upper 45 percent is a geometric diagnostic window containing mixed scene content, not a labeled background mask.',
                         'Draw 0 is a fixed illustration, not a selected best checkpoint. All draws are retained.',
                         'These adaptations used the original dataset supports, not corrections on this photograph.']}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('artifacts/phase6/workspace'))
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.workspace, args.image_id, args.output)
