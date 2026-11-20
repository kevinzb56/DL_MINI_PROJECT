"""Evaluate the immutable Phase 3 region-only checkpoint, without any tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image,ImageDraw
from src.atomic import write_json
from src.dataset import CrackDataset,verify_files
from src.model_factory import build_model
from scripts.experiments.train_baseline import evaluate


@torch.no_grad()
def preview(model,dataset,device,output,split):
    indices=random.Random(42).sample(range(len(dataset)),min(6,len(dataset)))
    sheet=Image.new('RGB',(1024,280*len(indices)),'white')
    draw=ImageDraw.Draw(sheet)
    for j,index in enumerate(indices):
        image,truth=dataset[index]
        probability=model(image[None].to(device)).sigmoid()[0,0].cpu().numpy()
        prediction=(probability>=.5).astype(np.uint8)*255
        original=Image.fromarray((image.numpy().transpose(1,2,0)*255).astype(np.uint8))
        target=Image.fromarray((truth[0].numpy()*255).astype(np.uint8))
        mask=Image.fromarray(prediction)
        overlay=Image.composite(Image.blend(original,Image.new('RGB',original.size,'red'),.55),original,mask)
        for column,view in enumerate((original,target.convert('RGB'),mask.convert('RGB'),overlay)):
            sheet.paste(view.resize((256,256)),(column*256,j*280+24))
        draw.text((2,j*280+4),dataset.rows[index]['image_id']+' | image / truth / prediction / overlay',fill='black')
        mask.save(output/f'{split}_preview_mask_{j}.png')
    sheet.save(output/f'{split}_predictions.jpg',quality=92)
    return [dataset.rows[index]['image_id'] for index in indices]


def run_evaluation(project,data_root,output):
    project,output=Path(project),Path(output)
    if output.exists(): raise ValueError('Output exists; use a new directory and preserve prior results')
    manifest_path=project/'data/manifests/pilot_v1.json'
    raw_manifest=manifest_path.read_bytes()
    manifest_hash=hashlib.sha256(raw_manifest).hexdigest()
    if manifest_hash!=manifest_path.with_suffix('.json.sha256').read_text().split()[0]:
        raise ValueError('Manifest hash mismatch')
    manifest=json.loads(raw_manifest)
    verify_files(manifest['records'],data_root)
    run=project/'artifacts/phase3/litecrack'
    pointer=json.loads((run/'checkpoints/best.json').read_text())
    path=(run/'checkpoints'/pointer['file']).resolve()
    if path.parent!=(run/'checkpoints').resolve(): raise ValueError('Unsafe checkpoint path')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=pointer['sha256']:
        raise ValueError('Checkpoint hash mismatch')
    # This is the trusted snapshot from the already-verified Phase 3 archive.
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['manifest_hash']!=manifest_hash: raise ValueError('Checkpoint used different splits')
    config=state['config']
    if config['architecture']!='litecrack': raise ValueError('Expected region-only LiteCrack')
    if not torch.cuda.is_available(): raise RuntimeError('Enable the Kaggle GPU runtime')
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    device=torch.device('cuda')
    model=build_model(config).to(device)
    model.load_state_dict(state['model'])
    validation=CrackDataset(manifest,data_root,'source_validation',config['image_size'])
    replay,_=evaluate(model,validation,device,config['batch_size'])
    recorded=json.loads((run/'evaluation.json').read_text())['results']['source_validation']
    for metric in ('dice','iou','precision','recall'):
        if abs(replay[metric]-recorded[metric])>1e-5:
            raise ValueError('Validation replay mismatch; stopping before holdout/query evaluation')
    output.mkdir(parents=True)
    results,preview_ids={},{}
    for split in ('source_holdout','target_query'):
        dataset=CrackDataset(manifest,data_root,split,config['image_size'])
        metrics,rows=evaluate(model,dataset,device,config['batch_size'])
        results[split]=metrics
        write_json(output/f'litecrack_{split}_per_image.json',rows)
        preview_ids[split]=preview(model,dataset,device,output,split)
    report={'manifest_sha256':manifest_hash,'architecture':'litecrack',
            'checkpoint_sha256':pointer['sha256'],'best_epoch':pointer['epoch'],
            'threshold':.5,'validation_replay':replay,'results':results,
            'preview_ids':preview_ids,'preview_selection':'seed 42, six samples per split; not cherry-picked',
            'scope':'Fixed region-only model; no training, adaptation or threshold tuning; exploratory target query previously used'}
    write_json(output/'evaluation.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); run_evaluation(a.project,a.data_root,a.output)
