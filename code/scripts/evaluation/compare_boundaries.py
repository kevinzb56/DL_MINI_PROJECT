"""Paired evaluation of frozen region-only and boundary-aware checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from src.atomic import write_json
from src.dataset import CrackDataset,verify_files
from src.model_factory import build_model
from scripts.experiments.train_baseline import evaluate


def compare(control,treatment,data_root,output):
    control,treatment,output=Path(control),Path(treatment),Path(output)
    if output.exists(): raise ValueError('Preserve prior evaluation; choose a new output directory')
    manifest_bytes=(treatment/'data/manifests/pilot_v1.json').read_bytes()
    if manifest_bytes!=(control/'data/manifests/pilot_v1.json').read_bytes(): raise ValueError('Split mismatch')
    manifest_hash=hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_hash!=(treatment/'data/manifests/pilot_v1.json.sha256').read_text().split()[0]: raise ValueError('Manifest checksum mismatch')
    manifest=json.loads(manifest_bytes); verify_files(manifest['records'],data_root)
    if not torch.cuda.is_available(): raise RuntimeError('Enable Kaggle GPU')
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    device=torch.device('cuda'); models={}; identities={}
    for name,project,relative in [('control',control,'artifacts/phase3/litecrack'),('boundary',treatment,'artifacts/phase4/litecrack_ba')]:
        run=project/relative
        pointer=json.loads((run/'checkpoints/best.json').read_text())
        path=(run/'checkpoints'/pointer['file']).resolve()
        if path.parent!=(run/'checkpoints').resolve(): raise ValueError('Unsafe checkpoint path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=pointer['sha256']: raise ValueError('Checkpoint hash mismatch')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['manifest_hash']!=manifest_hash: raise ValueError('Checkpoint split mismatch')
        config=state['config']; model=build_model(config).to(device); model.load_state_dict(state['model'])
        expected_arch='litecrack' if name=='control' else 'litecrack_ba'
        if config['architecture']!=expected_arch: raise ValueError('Unexpected architecture')
        dataset=CrackDataset(manifest,data_root,'source_validation',config['image_size'])
        replay,_=evaluate(model,dataset,device,config['batch_size'])
        original=json.loads((run/'evaluation.json').read_text())['results']['source_validation']
        for metric in ('dice','iou','precision','recall'):
            if abs(replay[metric]-original[metric])>1e-5: raise ValueError('Validation replay mismatch: '+name)
        models[name]=(model,config)
        identities[name]={'checkpoint_sha256':pointer['sha256'],'best_epoch':pointer['epoch'],'validation_replay':replay}
    output.mkdir(parents=True)
    summary={'manifest_sha256':manifest_hash,'checkpoints':identities,'threshold':.5,'splits':{},
             'boundary_definition':'3x3 morphological gradient; replicated borders; symmetric many-to-one matching at two-pixel Chebyshev radius; both-empty=1, one-empty=0',
             'scope':'Single-seed exploratory pilot; fixed weights, no training or threshold tuning'}
    for split in ('source_validation','source_holdout','target_query'):
        rows_by_model={}; scores={}
        for name,(model,config) in models.items():
            dataset=CrackDataset(manifest,data_root,split,config['image_size'])
            scores[name],rows=evaluate(model,dataset,device,config['batch_size'])
            rows_by_model[name]={r['image_id']:r for r in rows}
            write_json(output/f'{name}_{split}_per_image.json',rows)
        if rows_by_model['control'].keys()!=rows_by_model['boundary'].keys(): raise ValueError('Unpaired images')
        differences=[{'image_id':key,**{metric:rows_by_model['boundary'][key][metric]-rows_by_model['control'][key][metric]
                     for metric in ('dice','iou','precision','recall','boundary_f1')}} for key in rows_by_model['control']]
        delta={metric:100*(scores['boundary'][metric]-scores['control'][metric]) for metric in ('dice','iou','boundary_f1')}
        summary['splits'][split]={'scores':scores,'boundary_minus_control_percentage_points':delta,
            'boundary_target_met_on_point_estimates':delta['boundary_f1']>=2 and delta['dice']>=-1}
        write_json(output/f'{split}_paired_differences.json',differences)
        write_json(output/'comparison.json',summary)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('control','treatment','data-root','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); compare(a.control,a.treatment,a.data_root,a.output)
