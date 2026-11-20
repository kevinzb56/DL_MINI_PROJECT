"""Final baseline evaluation: fixed validation-best weights and threshold, no tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from src.atomic import write_json
from src.dataset import CrackDataset,verify_files
from src.models import SegmentationUNet
from scripts.experiments.train_baseline import evaluate


def run_evaluation(project,data_root,output):
    project=Path(project); output=Path(output)
    if output.exists(): raise ValueError('Evaluation output exists; preserve it and use a new directory')
    manifest_path=project/'data/manifests/pilot_v1.json'
    manifest_bytes=manifest_path.read_bytes()
    manifest_hash=hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_hash!=manifest_path.with_suffix('.json.sha256').read_text().split()[0]:
        raise ValueError('Manifest checksum mismatch')
    manifest=json.loads(manifest_bytes)
    verify_files(manifest['records'],data_root)
    if not torch.cuda.is_available(): raise RuntimeError('Use the verified Kaggle GPU runtime')
    output.mkdir(parents=True)
    summary={}
    for architecture in ('unet','r2au'):
        run=project/'artifacts/phase2'/architecture
        pointer=json.loads((run/'checkpoints/best.json').read_text())
        path=(run/'checkpoints'/pointer['file']).resolve()
        if path.parent!=(run/'checkpoints').resolve(): raise ValueError('Unsafe checkpoint path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=pointer['sha256']:
            raise ValueError('Checkpoint checksum mismatch')
        # Complete recovery files use pickle. Only load the trusted local run.
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['manifest_hash']!=manifest_hash: raise ValueError('Checkpoint uses different split')
        config=state['config']
        device=torch.device('cuda')
        model=SegmentationUNet(architecture,config['base_channels'],config.get('normalization','none')).to(device)
        model.load_state_dict(state['model'])
        validation=CrackDataset(manifest,data_root,'source_validation',config['image_size'])
        replay,_=evaluate(model,validation,device,config['batch_size'])
        expected=json.loads((run/'evaluation.json').read_text())['results']['source_validation']
        for key in ('dice','iou','precision','recall'):
            if abs(replay[key]-expected[key])>1e-5:
                raise ValueError(f'{architecture} validation replay mismatch: {key}; no query evaluation performed for this model')
        results={}
        for split in ('source_holdout','target_query'):
            dataset=CrackDataset(manifest,data_root,split,config['image_size'])
            metrics,rows=evaluate(model,dataset,device,config['batch_size'])
            results[split]=metrics
            write_json(output/(architecture+'_'+split+'_per_image.json'),rows)
        summary[architecture]={'checkpoint_sha256':pointer['sha256'],'best_epoch':pointer['epoch'],
            'validation_replay':replay,'threshold':.5,'results':results}
        write_json(output/'evaluation.json',{'manifest_sha256':manifest_hash,'models':summary,
            'scope':'Fixed stabilized baselines; no training or threshold tuning; exploratory pilot query previously used in v1'})
        print(json.dumps({architecture:summary[architecture]},indent=2),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); run_evaluation(a.project,a.data_root,a.output)
