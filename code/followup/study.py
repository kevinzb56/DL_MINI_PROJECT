"""Validation-only search followed by a locked adaptation/evaluation matrix."""
import json
from pathlib import Path
import zipfile
import torch
from followup.adaptation import adapt, sha, code_identity
from scripts.release_integrity import MODEL_ARCHIVES
from scripts.project_layout import archive_path
from scripts.experiments.train_baseline import evaluate
from src.atomic import write_json
from src.dataset import CrackDataset, verify_files
from src.model_factory import build_model


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value: raise ValueError('Frozen study changed: use a new output directory')
    else: write_json(path, value)


def checkpoint(folder, which='latest'):
    folder = Path(folder).resolve(); pointer = json.loads((folder/(which+'.json')).read_text())
    path = (folder/pointer['file']).resolve()
    if path.parent != folder or sha(path) != pointer['sha256']: raise ValueError('Checkpoint integrity failure')
    return torch.load(path, map_location='cpu', weights_only=False), pointer['sha256']


def original_parents(root, manifest_sha):
    result = {}
    import io
    pinned={3:'9ac52e68c86938f826ddf9bf5176ce8abba830c4ecb3cdd01dd35823777c1fa2',
            4:'786fbf654af8ff1b0e139328ac5d4f121ce9c215dd504988054ae21502f661f6'}
    for phase, name in [(3,'litecrack'), (4,'litecrack_ba')]:
        archive = archive_path(root, f'phase{phase}_results.zip')
        if not archive.exists():
            # Extracted original inputs are accepted with pinned checkpoint and manifest identities.
            state,digest=checkpoint(Path(root)/f'artifacts/phase{phase}/{name}/checkpoints','best')
            if digest!=pinned[phase] or state['manifest_hash']!=manifest_sha:raise ValueError('Unrecognized extracted original parent')
            result[name]=(state,digest)
            continue
        if sha(archive) != MODEL_ARCHIVES[archive.name]: raise ValueError('Original archive changed')
        with zipfile.ZipFile(archive) as z:
            prefix = f'artifacts/phase{phase}/{name}/checkpoints/'
            pointer = json.loads(z.read(prefix+'best.json'))
            if pointer['sha256']!=pinned[phase]:raise ValueError('Unexpected original parent')
            if Path(pointer['file']).name != pointer['file']: raise ValueError('Unsafe checkpoint pointer')
            data = z.read(prefix+pointer['file'])
            import hashlib
            if hashlib.sha256(data).hexdigest() != pointer['sha256']: raise ValueError('Parent checksum failure')
            state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
            if state['manifest_hash'] != manifest_sha: raise ValueError('Parent manifest mismatch')
            result[name] = (state,pointer['sha256'])
    return result


def gate(source_parent, source, target_parent, target, config):
    drop = 100*(source_parent-source); gain = 100*(target-target_parent)
    return {'passed': drop <= config['max_source_drop_points'] and gain > config['min_target_gain_points'],
            'source_change_points': -drop, 'target_gain_points': gain}


def choose(rows):
    viable = [r for r in rows if r['gate']['passed'] and r['condition']['distill_weight'] > 0
              and r.get('scene_gate',{}).get('passed',True)]
    return max(viable, key=lambda r: (r['gate']['target_gain_points'], r['gate']['source_change_points'], r['condition']['name'])) if viable else None


def score(model, manifest, root, split, size, device, ids=None):
    subset = {'records':[r for r in manifest['records'] if ids is None or r['image_id'] in ids]}
    data = CrackDataset(subset,root,split,size)
    summary, rows = evaluate(model,data,torch.device(device),2)
    return {'summary':summary,'per_image':rows}


def settings(config, condition):
    return {**config['adaptation'], **condition, 'device':config['device'], 'threads':config['threads']}


def develop(parents, manifest, manifest_sha, root, config, output, extra=None):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    freeze(output/'protocol.json', {'config':config, 'manifest_sha256':manifest_sha,
           'parents':{n:h for n,(_,h) in parents.items()}, 'code':code_identity(),
           'extra_manifest_sha256': extra['sha256'] if extra else None,
           'selection':'Source validation and remaining acquisition only; no holdout/query access.'})
    draw = next(d for d in manifest['support_draws'] if d['draw']==config['screen_draw'])
    ids = draw['supports'][str(config['screen_k'])]
    validation = {r['image_id'] for r in manifest['records'] if r['split']=='target_acquisition'}-set(ids)
    if not validation: raise ValueError('Separate acquisition validation required')
    selections = {}; all_rows = []
    scenes = None
    if extra:
        from followup.scenes import audit_scenes,score_scenes,scene_gate,require_scene_size
        scenes=audit_scenes(extra['manifest'],extra['root'],manifest)
        require_scene_size(scenes,config.get('scene_requirements',{}))
        if not all(any(r['split']==s for r in scenes['records']) for s in ('scene_train','scene_validation','scene_test')):
            raise ValueError('Background follow-up needs site-separated train, validation and test scenes')
    for name,(state,parent_sha) in parents.items():
        model = build_model(state['config']).to(config['device']); model.load_state_dict(state['model'])
        size=state['config']['image_size']
        base_source=score(model,manifest,root,'source_validation',size,config['device'])
        base_target=score(model,manifest,root,'target_acquisition',size,config['device'],validation)
        base_scene=score_scenes(state,scenes,extra['root'],'scene_validation',config['device']) if scenes else None
        rows=[]
        for condition in config['conditions']:
            run=output/name/condition['name']
            model=adapt(state,parent_sha,manifest,manifest_sha,root,ids,settings(config,condition),draw['seed'],run,
                        resume=(run/'checkpoints/latest.json').exists(),extra=extra if condition['distill_weight']>0 else None)
            if model is None: return None
            source=score(model,manifest,root,'source_validation',size,config['device'])
            target=score(model,manifest,root,'target_acquisition',size,config['device'],validation)
            row={'model':name,'condition':condition,'source':source,'target':target,
                 'gate':gate(base_source['summary']['dice'],source['summary']['dice'],
                             base_target['summary']['dice'],target['summary']['dice'],config['gates'])}
            if scenes:
                candidate_state={'config':state['config'],'model':model.state_dict()}
                row['scenes']=score_scenes(candidate_state,scenes,extra['root'],'scene_validation',config['device'])
                row['scene_gate']=scene_gate(base_scene,row['scenes'],config['broad_scene_gates'])
            rows.append(row);all_rows.append(row);write_json(output/'development_results.json',all_rows)
            print(name,condition['name'],json.dumps(row['gate']),flush=True)
        best=choose(rows)
        selections[name]={'condition':best['condition'] if best else None,
                          'base_source':base_source,'base_target':base_target,
                          'support_ids':ids,'validation_ids':sorted(validation),
                          'status':'selected' if best else 'no_admissible_candidate'}
    lock={'protocol_sha256':sha(output/'protocol.json'),'selections':selections,
          'scope':'Hyperparameters selected on acquisition/source validation. Old query remains exploratory.'}
    freeze(output/'selection.json',lock)
    return lock


def confirm(parents, manifest, manifest_sha, root, config, development, output, extra=None):
    development=Path(development);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    lock=json.loads((development/'selection.json').read_text())
    protocol=json.loads((development/'protocol.json').read_text())
    expected={'config':config,'manifest_sha256':manifest_sha,'parents':{n:h for n,(_,h) in parents.items()},
              'code':code_identity(),'extra_manifest_sha256':extra['sha256'] if extra else None,
              'selection':'Source validation and remaining acquisition only; no holdout/query access.'}
    if protocol!=expected or sha(development/'protocol.json')!=lock['protocol_sha256']:
        raise ValueError('Development protocol/code/parents changed after selection')
    freeze(output/'protocol.json',{'development_protocol':protocol,'selection_sha256':sha(development/'selection.json'),
           'test_scope':'Previously inspected frozen holdout/query; exploratory, not independent confirmation.'})
    jobs=[]
    for name,(state,parent_sha) in parents.items():
        chosen=lock['selections'][name]['condition']
        if chosen is None:
            write_json(output/(name+'_unmet.json'),{'state':'no_admissible_candidate','no_gate_relaxation':True})
            continue
        conditions=[next(c for c in config['conditions'] if c['name']==n) for n in ('no_replay','replay')]+[chosen]
        for draw in manifest['support_draws']:
            if draw['draw'] not in config['draws']:continue
            for k in config['budgets']:
                ids=draw['supports'][str(k)]
                for condition in conditions:
                    run=output/name/f"draw{draw['draw']}_k{k}"/condition['name']
                    # The no-replay/replay controls never receive external background examples.
                    candidate_extra=extra if condition['distill_weight']>0 else None
                    model=adapt(state,parent_sha,manifest,manifest_sha,root,ids,settings(config,condition),draw['seed'],run,
                                resume=(run/'checkpoints/latest.json').exists(),extra=candidate_extra)
                    if model is None:return None
                    del model
                    jobs.append((name,draw['draw'],k,condition,run))
    # No test metrics are exposed until the entire locked adaptation matrix completes.
    result={'scope':'exploratory reused-query study','selection_sha256':sha(development/'selection.json'),
            'static':{},'adapted':[],'training_seeds_are_separate_parents':True}
    for name,(state,_) in parents.items():
        model=build_model(state['config']).to(config['device']);model.load_state_dict(state['model'])
        result['static'][name]={split:score(model,manifest,root,split,state['config']['image_size'],config['device'])
                                for split in ('source_holdout','target_query')}
    for name,draw,k,condition,run in jobs:
        state,digest=checkpoint(run/'checkpoints')
        model=build_model(state['config']['model_config']).to(config['device']);model.load_state_dict(state['model'])
        scores={split:score(model,manifest,root,split,state['config']['model_config']['image_size'],config['device'])
                for split in ('source_holdout','target_query')}
        base=result['static'][name]
        result['adapted'].append({'model':name,'draw':draw,'k':k,'condition':condition['name'],
            'checkpoint_sha256':digest,'scores':scores,'gate':gate(base['source_holdout']['summary']['dice'],
            scores['source_holdout']['summary']['dice'],base['target_query']['summary']['dice'],
            scores['target_query']['summary']['dice'],config['gates'])})
        write_json(output/'results.json',result)
    write_json(output/'results.json',result)
    write_json(output/'status.json',{'state':'complete','adaptations':len(jobs),
               'unmet_models':[n for n in parents if lock['selections'][n]['condition'] is None]})
    return result
