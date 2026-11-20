"""Run the preregistered two-model, five-draw, three-budget adaptation matrix."""
import argparse,hashlib,json,os,sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from src.atomic import write_json
from src.dataset import CrackDataset,verify_files
from src.model_factory import build_model
from scripts.experiments.adapt_fewshot import adapt
from scripts.experiments.train_baseline import evaluate

def run(control,treatment,data_root,settings_path,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    settings=json.loads(Path(settings_path).read_text())
    raw=(Path(treatment)/'data/manifests/pilot_v1.json').read_bytes()
    if raw!=(Path(control)/'data/manifests/pilot_v1.json').read_bytes():raise ValueError('Split mismatch')
    manifest=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    verify_files(manifest['records'],data_root)
    device=torch.device(settings['device'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    parents={};identity={}
    for name,project,relative in [('control',control,'artifacts/phase3/litecrack'),('boundary',treatment,'artifacts/phase4/litecrack_ba')]:
        folder=Path(project)/relative
        pointer=json.loads((folder/'checkpoints/best.json').read_text())
        path=(folder/'checkpoints'/pointer['file']).resolve()
        if path.parent!=(folder/'checkpoints').resolve() or hashlib.sha256(path.read_bytes()).hexdigest()!=pointer['sha256']:raise ValueError('Parent checkpoint invalid')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['manifest_hash']!=digest:raise ValueError('Parent split mismatch')
        parents[name]=state;identity[name]=pointer['sha256']
    protocol={'settings':settings,'manifest_sha256':digest,'parents':identity,'support_draws':manifest['support_draws'],
              'label_provenance':'Existing fully annotated public acquisition masks; no measured human correction time'}
    protocol_path=output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise ValueError('Protocol changed; use a new experiment')
    write_json(protocol_path,protocol)
    # Complete every adaptation job before exposing query results.
    for name,state in parents.items():
        for draw_id in settings['draws']:
            draw=next(d for d in manifest['support_draws'] if d['draw']==draw_id)
            for k in settings['budgets']:
                folder=output/name/f'draw{draw_id}_k{k}'
                result=adapt(state,identity[name],manifest,digest,data_root,draw['supports'][str(k)],settings,
                    draw['seed'],folder,resume=(folder/'checkpoints/latest.json').exists())
                if result is None:return
                del result
                print(f'Adaptation saved: {name}, draw {draw_id}, K={k}',flush=True)
    summary={'protocol':protocol,'static':{},'adapted':[]}
    for name,state in parents.items():
        configs=state['config'];model=build_model(configs).to(device);model.load_state_dict(state['model'])
        def score(model,label):
            result={}
            for split in ('source_validation','source_holdout','target_query'):
                dataset=CrackDataset(manifest,data_root,split,configs['image_size'])
                metrics,rows=evaluate(model,dataset,device,configs['batch_size'])
                result[split]=metrics;write_json(output/f'{label}_{split}.json',rows)
            return result
        summary['static'][name]=score(model,name+'_k0')
        for draw_id in settings['draws']:
            for k in settings['budgets']:
                folder=output/name/f'draw{draw_id}_k{k}'
                pointer=json.loads((folder/'checkpoints/latest.json').read_text())
                checkpoint=folder/'checkpoints'/pointer['file']
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=pointer['sha256']:raise ValueError('Adapted checkpoint corrupted')
                adapted=torch.load(checkpoint,map_location='cpu',weights_only=False)
                model.load_state_dict(adapted['model'])
                metrics=score(model,f'{name}_draw{draw_id}_k{k}')
                summary['adapted'].append({'model':name,'draw':draw_id,'k':k,'checkpoint_sha256':pointer['sha256'],
                    'results':metrics,'target_dice_gain_points':100*(metrics['target_query']['dice']-summary['static'][name]['target_query']['dice']),
                    'source_dice_change_points':100*(metrics['source_holdout']['dice']-summary['static'][name]['source_holdout']['dice'])})
                write_json(output/'results.json',summary)
    write_json(output/'status.json',{'state':'complete','adaptation_runs':len(summary['adapted'])})
    print('Complete: 30 adaptations, static references and fixed-checkpoint evaluations',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('control','treatment','data-root','settings','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.control,a.treatment,a.data_root,a.settings,a.output)
