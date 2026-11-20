"""Audit final Milano evidence and build a concise, traceable research report."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'output/milano_v1/recovery/milano_recovery_litecrack_ba_20261110T184426004641Z.zip'
PREP=ROOT/'data/external/milano_prepared_v1'
OUT=ROOT/'output/milano_v1'
OUT.mkdir(parents=True,exist_ok=True)

archive_sha=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
assert archive_sha==ARCHIVE.with_suffix('.zip.sha256').read_text().split()[0]
with zipfile.ZipFile(ARCHIVE) as z:
    assert z.testzip() is None
    def read_json(name): return json.loads(z.read(name))
    comparison=read_json('artifacts/comparison.json')
    manifests={m:read_json(f'artifacts/{m}/manifest.json') for m in ('litecrack','litecrack_ba')}
    assert manifests['litecrack']==manifests['litecrack_ba']
    manifest=manifests['litecrack']; manifest_sha=hashlib.sha256(z.read('artifacts/litecrack/manifest.json')).hexdigest()
    histories={m:read_json(f'artifacts/{m}/history.json') for m in ('litecrack','litecrack_ba')}
    rows={m:read_json(f'artifacts/{m}/source_holdout_per_image.json') for m in ('litecrack','litecrack_ba')}
    configs={m:read_json(f'artifacts/{m}/config.json') for m in ('litecrack','litecrack_ba')}
    statuses={m:read_json(f'artifacts/{m}/status.json') for m in ('litecrack','litecrack_ba')}
    best_steps={m:read_json(f'artifacts/{m}/checkpoints/best.json')['global_step'] for m in ('litecrack','litecrack_ba')}
    runtime=read_json('artifacts/runtime_rebuild.json')
    rebuilt=read_json('artifacts/rebuilt_environment_packages.json')
    for model in rows:
        assert statuses[model]['state']=='training_complete'
        assert len(rows[model])==231
        for key in ('dice','iou','precision','recall','boundary_f1'):
            macro=float(np.mean([r[key] for r in rows[model]]))
            recorded=comparison[model]['results']['source_holdout'][key]
            assert abs(macro-recorded)<2e-6,(model,key,macro,recorded)
        for pointer_name in ('latest.json','best.json'):
            pointer=read_json(f'artifacts/{model}/checkpoints/{pointer_name}')
            target=f"artifacts/{model}/checkpoints/{pointer['file']}"
            assert hashlib.sha256(z.read(target)).hexdigest()==pointer['sha256']

source_rows={r['image_id']:r for r in manifest['records']}
assert len(source_rows)==2209
assert set(r['image_id'] for r in rows['litecrack'])==set(r['image_id'] for r in rows['litecrack_ba'])

# Refine only the broad average-hash candidate list with a second grayscale thumbnail screen.
audit=json.loads((PREP/'preparation_audit.json').read_text())
thumbs={}
def thumb(image_id):
    if image_id not in thumbs:
        row=source_rows['milano/'+image_id]
        with Image.open(PREP/row['image']) as im:
            thumbs[image_id]=np.asarray(im.convert('L').resize((64,64),Image.Resampling.LANCZOS),dtype=np.float32)
    return thumbs[image_id]

similar=[]
for pair in audit['similarity_candidates']:
    a,b=thumb(pair['a']),thumb(pair['b']); x=a-a.mean(); y=b-b.mean()
    corr=float((x*y).sum()/(np.sqrt((x*x).sum()*(y*y).sum())+1e-9))
    rmse=float(np.sqrt(np.mean((a-b)**2)))
    ids=pair['a'].split('/'),pair['b'].split('/')
    similar.append({'image_a':pair['a'],'image_b':pair['b'],
        'average_hash_distance_8x8':pair['average_hash_distance'],
        'grayscale_64x64_pearson':corr,'grayscale_64x64_rmse':rmse,
        'filename_ids_consecutive':ids[0][1][2:].isdigit() and ids[1][1][2:].isdigit()
            and abs(int(ids[0][1][2:])-int(ids[1][1][2:]))<=2})
similar.sort(key=lambda r:(-r['grayscale_64x64_pearson'],r['grayscale_64x64_rmse']))
strong=[r for r in similar if r['grayscale_64x64_pearson']>=.95]
strong_test_pairs=[r for r in strong if '/test/' in '/'+r['image_a'] or '/test/' in '/'+r['image_b']]
with (OUT/'milano_similarity_candidates.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(similar[0])); writer.writeheader(); writer.writerows(similar)

# Review sheet is a visual triage aid, not an automatic split correction.
selected=similar[:24]
sheet=Image.new('RGB',(960,len(selected)*150),'white'); draw=ImageDraw.Draw(sheet)
for j,pair in enumerate(selected):
    for col,key in enumerate(('image_a','image_b')):
        row=source_rows['milano/'+pair[key]]
        with Image.open(PREP/row['image']) as im:
            tile=im.convert('RGB').resize((320,120),Image.Resampling.LANCZOS)
        sheet.paste(tile,(col*320,j*150+24))
    draw.text((645,j*150+24),f"r={pair['grayscale_64x64_pearson']:.3f}  d={pair['average_hash_distance_8x8']}\n{pair['image_a']}\n{pair['image_b']}",fill='black')
sheet.save(OUT/'milano_similarity_review.jpg',quality=92)

empty={}
for row in manifest['records']:
    if row['split']!='source_holdout': continue
    with Image.open(PREP/row['mask']) as mask: empty[row['image_id']]=not np.asarray(mask).any()
assert sum(empty.values())==45

summary={}
for model in ('litecrack','litecrack_ba'):
    vals=comparison[model]['results']['source_holdout']
    per=rows[model]
    positive=[r for r in per if not empty[r['image_id']]]
    negatives=[r for r in per if empty[r['image_id']]]
    summary[model]={**vals,'train_epochs':len(histories[model]),
      'best_validation_dice':max(r['validation']['dice'] for r in histories[model]),
      'best_checkpoint_validation_step':best_steps[model],
      'positive_test_images':len(positive),'empty_test_masks':len(negatives),
      'positive_only_macro_dice':float(np.mean([r['dice'] for r in positive])),
      'empty_mask_predicted_empty_fraction':float(np.mean([r['dice']==1. for r in negatives]))}

delta={k:summary['litecrack_ba'][k]-summary['litecrack'][k]
       for k in ('dice','iou','precision','recall','boundary_f1','positive_only_macro_dice')}
out_report={'archive':ARCHIVE.name,'archive_sha256':archive_sha,'manifest_sha256':manifest_sha,
    'models':summary,'litecrack_ba_minus_litecrack':delta,
    'similarity':{'average_hash_candidates':len(similar),'strong_review_candidates_corr_ge_0_95':len(strong),
        'near_identical_corr_ge_0_99':sum(r['grayscale_64x64_pearson']>=.99 for r in similar),
        'strong_candidates_crossing_test':len(strong_test_pairs),
        'filename_ids_consecutive_within_2':sum(r['filename_ids_consecutive'] for r in strong),
        'method':'Candidates from 64-bit 8x8 average-hash Hamming distance <=3; re-ranked by global Pearson correlation and RMSE of 64x64 grayscale thumbnails; manual review still required',
        'candidate_rows':len(similar),'images_loaded':len(thumbs)},
    'runtime':{'original_record':runtime['original_runtime'],'rebuilt_record':runtime['rebuilt_runtime'],
        'rebuilt_cudnn':rebuilt['cudnn_version'],'cross_build_bitwise_equivalence_claimed':False},
    'limitations':['One training seed (42)','Publisher test split, not confirmed site independent',
        'Similarity pairs are heuristic flags; no site metadata is available to adjudicate them',
        'No road masks, so road-only false-positive area and vegetation performance cannot be measured',
        'Publisher taxonomy includes crack or pavement distress','No external human annotation study']}
(OUT/'milano_final_report.json').write_text(json.dumps(out_report,indent=2)+'\n',encoding='utf-8')

def pct(v): return f'{100*v:.2f}%'
text=f'''# Milano pavement segmentation: completed experiment

**Status:** both models trained and the publisher test split evaluated. This is a single-seed exploratory result, not an independent-site validation.

## Protocol

Dataset: [Asphalt Pavement Crack Segmentation Dataset for Mobile Road Distress Assessment](https://doi.org/10.5281/zenodo.20040434), with publisher train/validation/test splits of 1,480/498/231 images. The publisher describes binary crack or pavement-distress labels. The frozen manifest SHA256 is `{manifest_sha}`. Images were resized to 256x256 at load time. Both models used seed 42, random initialization, batch 8, BCE+Dice, learning rate 1e-4, and validation patience 15. LiteCrack-BA used boundary-loss weight 0.2. The best validation-Dice checkpoint for each model was evaluated once at threshold 0.5. Per-image metrics were macro-averaged across all 231 test images; the reported aggregates were independently recomputed from the archived per-image results.

## Results

| Model | Epochs | Best validation Dice | Test Dice | Test IoU | Precision | Recall | Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LiteCrack | {summary['litecrack']['train_epochs']} | {pct(summary['litecrack']['best_validation_dice'])} | {pct(summary['litecrack']['dice'])} | {pct(summary['litecrack']['iou'])} | {pct(summary['litecrack']['precision'])} | {pct(summary['litecrack']['recall'])} | {pct(summary['litecrack']['boundary_f1'])} |
| LiteCrack-BA | {summary['litecrack_ba']['train_epochs']} | {pct(summary['litecrack_ba']['best_validation_dice'])} | {pct(summary['litecrack_ba']['dice'])} | {pct(summary['litecrack_ba']['iou'])} | {pct(summary['litecrack_ba']['precision'])} | {pct(summary['litecrack_ba']['recall'])} | {pct(summary['litecrack_ba']['boundary_f1'])} |

LiteCrack-BA improves macro test Dice by **{100*delta['dice']:.2f} percentage points** and boundary F1 by **{100*delta['boundary_f1']:.2f} points**. It raises precision by {100*delta['precision']:.2f} points and lowers recall by {abs(100*delta['recall']):.2f} points. This overall Dice gain is concentrated in empty-mask images: the test split includes 45 such images, and the fraction predicted entirely empty was {pct(summary['litecrack']['empty_mask_predicted_empty_fraction'])} for LiteCrack and {pct(summary['litecrack_ba']['empty_mask_predicted_empty_fraction'])} for LiteCrack-BA. On the 186 nonempty-mask images, macro Dice was {pct(summary['litecrack']['positive_only_macro_dice'])} and {pct(summary['litecrack_ba']['positive_only_macro_dice'])}, respectively; BA is lower there by {abs(100*delta['positive_only_macro_dice']):.2f} points. Report both figures; the aggregate alone would hide this tradeoff.

## Split-overlap screen

The initial 8x8 average-hash screen flagged {len(similar):,} cross-split pairs. A second 64x64 grayscale correlation screen found **{len(strong)} pairs at correlation >=0.95** and **{out_report['similarity']['near_identical_corr_ge_0_99']} at >=0.99**; {out_report['similarity']['strong_candidates_crossing_test']} stronger candidates cross the test split. All {out_report['similarity']['filename_ids_consecutive_within_2']} stronger candidates have consecutive filename IDs within two frames. Visual inspection of the contact sheet shows closely matching adjacent road views, including pavement markings and vegetation. This raises a concrete risk of adjacent-frame leakage. The data contain no capture-group metadata to establish which images share a sequence or site, so the flagged pairs are not certified duplicates. The refined list is [`milano_similarity_candidates.csv`](milano_similarity_candidates.csv); review sheet: [`milano_similarity_review.jpg`](milano_similarity_review.jpg). The publisher test score should remain exploratory until provenance and candidate pairs are adjudicated. No records were removed or relabeled after evaluation.

## Limits and next work

The dataset supplies no verified physical-site or capture-group metadata, and no road-region masks. It therefore cannot measure road-only false-positive area or establish vegetation performance. Its labels include pavement distress, not necessarily cracks alone. The result uses one seed and is exploratory.

Before making a field-performance claim, collect permissioned, original-resolution road scenes from documented physical sites and capture groups; provide complete human crack masks and road-region masks, including explicit non-road hard negatives; split by site before tuning; freeze a site-separated test set; then evaluate both frozen checkpoints and report per-site scores and non-road false-positive area. The intake skeleton is [`configs/followup_scenes.template.json`](../../configs/followup_scenes.template.json); its required provenance and minimum counts are in [`docs/NEXT_GPU_RUN.md`](../../docs/NEXT_GPU_RUN.md). See the practical [`collection guide`](../../data/followup_external/COLLECTION_GUIDE.md). Current intake has zero scenes and no filled site/usage metadata.

## Reproducibility

Verified recovery archive: `{ARCHIVE.name}`; SHA256 `{archive_sha}`. It contains both validation-best checkpoints, training histories, per-image test metrics, and the comparison. Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8, and two Tesla T4 devices were used. The isolated Python build stamp differs from the earlier environment and is recorded in the archive; cross-build bitwise identity is not claimed.

Artifacts: [`milano_final_report.json`](milano_final_report.json), [`milano_similarity_candidates.csv`](milano_similarity_candidates.csv), [`milano_similarity_review.jpg`](milano_similarity_review.jpg), [`milano_recovery_litecrack_ba_20261110T184426004641Z.zip`](recovery/{ARCHIVE.name}), and the frozen data manifest under `data/external/milano_prepared_v1/manifest.json`.
'''
(OUT/'FINAL_REPORT.md').write_text(text,encoding='utf-8')
print(json.dumps({'report':str(OUT/'FINAL_REPORT.md'),'archive_sha256':archive_sha,
 'results':summary,'delta':delta,'similarity_candidates':len(similar),
 'strong_review_candidates':len(strong),'near_identical':out_report['similarity']['near_identical_corr_ge_0_99'],
 'images_loaded':len(thumbs)},indent=2),flush=True)
