"""Isolated retention/background studies and matched multi-seed source retraining."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from followup.adaptation import sha,code_identity
from followup.study import checkpoint,original_parents,develop,confirm,freeze,score
from followup.scenes import audit_scenes,score_scenes,scene_gate,require_scene_size
from src.atomic import write_json
from src.dataset import verify_files
from src.model_factory import build_model


def preflight(config,manifest_path,data_root):
    manifest=json.loads(Path(manifest_path).read_text());counts={};verified={};missing=[];bad=[]
    for row in manifest['records']:
        split=row['split'];counts[split]=counts.get(split,0)+1
        okay=True
        for field in ('image','mask'):
            path=(Path(data_root)/row[field]).resolve()
            if not path.is_relative_to(Path(data_root).resolve()):raise ValueError('Data path escapes root')
            if not path.is_file():missing.append(row['image_id']+':'+field);okay=False
            elif sha(path)!=row[field+'_sha256']:bad.append(row['image_id']+':'+field);okay=False
        if okay:verified[split]=verified.get(split,0)+1
    return {'cuda_available':torch.cuda.is_available(),'torch':torch.__version__,
            'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'manifest_matches':sha(manifest_path)==config['manifest_sha256'],
            'counts':counts,'verified_pairs':verified,'missing_files':len(missing),'mismatched_files':len(bad),
            'missing_examples':missing[:5],'mismatch_examples':bad[:5],
            'free_disk_gib':round(shutil.disk_usage(ROOT).free/1024**3,2),
            'ready_for_gpu_retraining':bool(torch.cuda.is_available() and not missing and not bad and sha(manifest_path)==config['manifest_sha256']),
            'independent_scene_data':'Must be supplied with complete labels and physical-site provenance.',
            'human_effort':'Requires real timed annotation sessions; not derived from imported labels.'}


def source_train(config,manifest,data_root,output,only_seed=None):
    from scripts.experiments.train_baseline import train
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    templates={n:json.loads((ROOT/'configs'/f'{n}.json').read_text()) for n in config['source_models']}
    freeze(output/'protocol.json',{'config':config,'templates':templates,'manifest_sha256':sha(manifest),
            'code':code_identity(),'selection':'source_validation only; no query/holdout evaluation during training'})
    entries=[]
    for seed in config['training_seeds']:
        for name,template in templates.items():
            run=output/f'seed{seed}'/name
            parent_config={**template,'seed':seed,'device':config['device'],
                           'protocol':'followup_multiseed_source_v1','evaluation_splits':['source_validation']}
            state=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
            if state.get('state')!='training_complete' and (only_seed is None or seed==only_seed):
                if shutil.disk_usage(output).free<8*1024**3:
                    raise RuntimeError('Need 8 GiB headroom before another source run; export recovery and restore in fresh storage')
                train(parent_config,manifest,data_root,run,resume=(run/'checkpoints/latest.json').exists(),final_evaluation=False)
                state=json.loads((run/'status.json').read_text())
                if state.get('state')!='training_complete':return
            if state.get('state')!='training_complete':continue
            _,digest=checkpoint(run/'checkpoints','best')
            entries.append({'name':name,'seed':seed,'checkpoint_dir':(run/'checkpoints').relative_to(output).as_posix(),
                            'which':'best','sha256':digest})
            write_json(output/'parents.json',{'protocol':'followup_multiseed_source_v1','entries':entries,'complete':False,'manifest_sha256':sha(manifest)})
    write_json(output/'parents.json',{'protocol':'followup_multiseed_source_v1','entries':entries,
               'complete':len(entries)==len(templates)*len(config['training_seeds']),'manifest_sha256':sha(manifest)})


def indexed_parents(path,manifest_sha,include_baseline=False):
    index=json.loads(Path(path).read_text())
    if not index.get('complete') or index['manifest_sha256']!=manifest_sha:raise ValueError('Complete multi-seed parent matrix required')
    parents={}
    for row in index['entries']:
        if not include_baseline and row['name'] not in ('litecrack','litecrack_ba'):continue
        state,digest=checkpoint(Path(path).parent/row['checkpoint_dir'],row['which'])
        if digest!=row['sha256'] or state['manifest_hash']!=manifest_sha:raise ValueError('Parent index changed')
        name=f"{row['name']}_seed{row['seed']}"
        if name in parents:raise ValueError('Duplicate parent')
        parents[name]=(state,digest)
    if not parents:raise ValueError('No compatible parents')
    return parents


def freeze_scene(config,manifest,parents,extra,development,model_name,mode,output):
    development=Path(development)
    selection=json.loads((development/'selection.json').read_text())
    protocol=json.loads((development/'protocol.json').read_text())
    if selection['protocol_sha256']!=sha(development/'protocol.json') or protocol['code']!=code_identity():
        raise ValueError('Selection provenance changed')
    if protocol['config']!=config or protocol['manifest_sha256']!=config['manifest_sha256']:
        raise ValueError('Scene protocol differs from development')
    if protocol['extra_manifest_sha256']!=extra['sha256'] or protocol['parents']!={n:h for n,(_,h) in parents.items()}:
        raise ValueError('Scene parent/data differs from development')
    chosen=selection['selections'][model_name]['condition']
    if chosen is None:raise ValueError('No candidate passed development gates')
    state,digest=checkpoint(development/model_name/chosen['name']/'checkpoints')
    data=audit_scenes(extra['manifest'],extra['root'],manifest)
    base=score_scenes(parents[model_name][0],data,extra['root'],'scene_validation',config['device'],mode)
    candidate=score_scenes(state,data,extra['root'],'scene_validation',config['device'],mode)
    gate=scene_gate(base,candidate,config['broad_scene_gates'])
    if not gate['passed']:raise ValueError('Scene validation gate failed; test remains unopened')
    freeze(output,{'config':config,'code':code_identity(),'scene_manifest_sha256':extra['sha256'],
           'selection_sha256':sha(development/'selection.json'),'parent_name':model_name,
           'parent_sha256':parents[model_name][1],'candidate_sha256':digest,
           'checkpoint_dir':str((development/model_name/chosen['name']/'checkpoints').resolve()),
           'mode':mode,'validation_gate':gate,'validation_parent':base,'validation_candidate':candidate})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['preflight','source-train','source-evaluate','develop','confirm','scene-freeze','scene-test'])
    p.add_argument('--config',type=Path,default=ROOT/'configs/followup_v1.json')
    p.add_argument('--manifest',type=Path,default=ROOT/'data/manifests/pilot_v1.json')
    p.add_argument('--data-root',type=Path,default=ROOT/'data/raw')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/followup_v1')
    p.add_argument('--archive-root',type=Path,default=ROOT)
    p.add_argument('--parents-index',type=Path)
    p.add_argument('--only-seed',type=int,choices=[42,73,101])
    p.add_argument('--development',type=Path)
    p.add_argument('--scene-manifest',type=Path);p.add_argument('--scene-root',type=Path)
    p.add_argument('--scene-lock',type=Path);p.add_argument('--parent-name')
    p.add_argument('--mode',choices=['resize','tiles'],default='resize')
    a=p.parse_args();config=json.loads(a.config.read_text());manifest=json.loads(a.manifest.read_text())
    if a.stage=='preflight':
        report=preflight(config,a.manifest,a.data_root);write_json(a.output/'preflight.json',report)
        print(json.dumps(report,indent=2));return
    if sha(a.manifest)!=config['manifest_sha256']:raise ValueError('Frozen pilot manifest changed')
    if config['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU needed: connect the Kaggle CUDA kernel')
    torch.set_num_threads(config['threads'])
    # Full original data integrity is required before any study execution.
    verify_files(manifest['records'],a.data_root)
    extra=None
    if a.scene_manifest:
        if not a.scene_root:raise ValueError('--scene-root required')
        audited=audit_scenes(a.scene_manifest,a.scene_root,manifest)
        require_scene_size(audited,config['scene_requirements'])
        extra={'manifest':a.scene_manifest,'root':a.scene_root,'sha256':sha(a.scene_manifest)}
    if a.stage=='source-train':source_train(config,a.manifest,a.data_root,a.output,a.only_seed);return
    if a.parents_index:
        index=json.loads(a.parents_index.read_text())
        wanted={(name,seed) for name in config['source_models'] for seed in config['training_seeds']}
        actual={(r['name'],r['seed']) for r in index['entries']}
        if index.get('protocol')!='followup_multiseed_source_v1' or actual!=wanted or len(index['entries'])!=len(wanted):
            raise ValueError('Matched complete predeclared multi-seed matrix required')
    parents=indexed_parents(a.parents_index,config['manifest_sha256'],a.stage=='source-evaluate') if a.parents_index else original_parents(a.archive_root,config['manifest_sha256'])
    if a.stage=='source-evaluate':
        if not a.parents_index:raise ValueError('Source evaluation requires the complete new multi-seed parent index')
        freeze(a.output/'protocol.json',{'config':config,'code':code_identity(),'parents':{n:h for n,(_,h) in parents.items()}})
        result={}
        for name,(state,_) in parents.items():
            model=build_model(state['config']).to(config['device']);model.load_state_dict(state['model'])
            result[name]={s:score(model,manifest,a.data_root,s,state['config']['image_size'],config['device']) for s in ('source_holdout','target_query')}
            write_json(a.output/'results.json',{'scope':'Repeated training seeds; reused target query remains exploratory','results':result})
    elif a.stage=='develop':develop(parents,manifest,config['manifest_sha256'],a.data_root,config,a.output,extra)
    elif a.stage=='confirm':
        if not a.development:raise ValueError('--development required')
        confirm(parents,manifest,config['manifest_sha256'],a.data_root,config,a.development,a.output,extra)
    elif a.stage=='scene-freeze':
        if not all((extra,a.development,a.parent_name,a.scene_lock)):raise ValueError('Scenes, development, parent name and scene lock required')
        freeze_scene(config,manifest,parents,extra,a.development,a.parent_name,a.mode,a.scene_lock)
    elif a.stage=='scene-test':
        if not extra or not a.scene_lock:raise ValueError('Scene data and previously frozen scene lock required')
        lock=json.loads(a.scene_lock.read_text())
        if lock['config']!=config or lock['code']!=code_identity() or lock['scene_manifest_sha256']!=extra['sha256']:
            raise ValueError('Test lock config/code/data changed')
        state,digest=checkpoint(lock['checkpoint_dir'])
        parent=parents[lock['parent_name']]
        if digest!=lock['candidate_sha256'] or parent[1]!=lock['parent_sha256']:raise ValueError('Locked models changed')
        data=audit_scenes(extra['manifest'],extra['root'],manifest)
        base=score_scenes(parent[0],data,extra['root'],'scene_test',config['device'],lock['mode'])
        candidate=score_scenes(state,data,extra['root'],'scene_test',config['device'],lock['mode'])
        freeze(a.output/'scene_test.json',{'scene_lock_sha256':sha(a.scene_lock),'parent':base,'candidate':candidate,
               'gate':scene_gate(base,candidate,config['broad_scene_gates']),
               'scope':'Reserved supplied physical sites; provenance is supplied, not independently certified.'})


if __name__=='__main__':main()
