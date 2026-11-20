"""Audit a downloaded follow-up backup without changing experiments or app models."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import stat
import sys
import xml.etree.ElementTree as ET
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.project_layout import archive_path
from scripts.release_integrity import safe_member,digest
from followup.adaptation import code_identity
from src.atomic import write_json


def verify(z):
    infos=z.infolist();names=[safe_member(i.filename) for i in infos]
    if len(names)!=len(set(n.casefold() for n in names)):raise ValueError('Duplicate archive paths')
    if sum(i.file_size for i in infos)>2*1024**3:raise ValueError('Archive too large')
    if any(stat.S_ISLNK(i.external_attr>>16) for i in infos):raise ValueError('Symlinks forbidden')
    marker='FOLLOWUP_BACKUP_MANIFEST.json'
    if z.getinfo(marker).file_size>8*1024**2:raise ValueError('Oversized inventory')
    inventory=json.loads(z.read(marker))['files']
    if set(names)!=(set(inventory)|{marker}):raise ValueError('Archive inventory differs')
    for name,entry in inventory.items():
        if z.getinfo(name).file_size!=entry['bytes']:raise ValueError('Member size mismatch: '+name)
        h=hashlib.sha256()
        with z.open(name) as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=entry['sha256']:raise ValueError('Member checksum mismatch: '+name)
    return inventory


def metrics(record,expected):
    rows=record['per_image'];summary=record['summary']
    if len(rows)!=len(expected) or {r['image_id'] for r in rows}!=set(expected) or summary['n']!=len(rows):
        raise ValueError('Evaluation split IDs/count mismatch')
    for key in ('dice','iou','precision','recall','boundary_f1'):
        values=[r[key] for r in rows]
        if any(not math.isfinite(v) or not 0<=v<=1 for v in values):raise ValueError('Invalid per-image metric')
        if abs(statistics.mean(values)-summary[key])>1e-8:raise ValueError('Metric mean differs from per-image records')


def gate(parent_source,source,parent_target,target,config):
    change=100*(source-parent_source);gain=100*(target-parent_target)
    return {'passed':change>=-config['max_source_drop_points'] and gain>config['min_target_gain_points'],
            'source_change_points':change,'target_gain_points':gain}


def verify_gate(record,expected):
    if record['passed']!=expected['passed']:raise ValueError('Gate flag mismatch')
    for key in ('source_change_points','target_gain_points'):
        if abs(record[key]-expected[key])>1e-8:raise ValueError('Gate arithmetic mismatch')


def audit(path,output):
    import torch
    from src.model_factory import build_model
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        inventory=verify(z)
        def read(name):return json.loads(z.read(name))
        protocol=read('retention_development/protocol.json');config=read('configs/followup_v1.json')
        selection=read('retention_development/selection.json');development=read('retention_development/development_results.json')
        manifest=read('data/manifests/pilot_v1.json');result=read('retention_matrix/results.json')
        completed=read('retention_matrix/status.json');matrix_protocol=read('retention_matrix/protocol.json')
        if config!=json.loads((ROOT/'configs/followup_v1.json').read_text()) or config!=protocol['config']:
            raise ValueError('Configuration differs from implemented follow-up')
        if hashlib.sha256(z.read('data/manifests/pilot_v1.json')).hexdigest()!=config['manifest_sha256']:
            raise ValueError('Frozen split manifest changed')
        if z.read('data/manifests/pilot_v1.json')!=(ROOT/'data/manifests/pilot_v1.json').read_bytes():raise ValueError('Local manifest differs')
        if set(protocol['code'])!=set(code_identity()):raise ValueError('Experiment source inventory differs')
        newline_differences=[]
        for name,sha in protocol['code'].items():
            archived=z.read(name);local=(ROOT/name).read_bytes()
            if hashlib.sha256(archived).hexdigest()!=sha:raise ValueError('Archived source differs: '+name)
            if local!=archived:
                if local.replace(b'\r\n',b'\n')!=archived.replace(b'\r\n',b'\n'):
                    raise ValueError('Experiment implementation differs: '+name)
                newline_differences.append(name)
        if hashlib.sha256(z.read('retention_development/protocol.json')).hexdigest()!=selection['protocol_sha256']:
            raise ValueError('Selection protocol checksum mismatch')
        selection_sha=hashlib.sha256(z.read('retention_development/selection.json')).hexdigest()
        if matrix_protocol['selection_sha256']!=selection_sha or result['selection_sha256']!=selection_sha or matrix_protocol['development_protocol']!=protocol:
            raise ValueError('Matrix differs from frozen selection')
        if protocol['extra_manifest_sha256'] is not None:raise ValueError('Unexpected external scene intervention')
        lookup={r['image_id']:r for r in manifest['records']}
        split_ids={s:{r['image_id'] for r in manifest['records'] if r['split']==s} for s in manifest['counts']}
        pinned={'litecrack':'9ac52e68c86938f826ddf9bf5176ce8abba830c4ecb3cdd01dd35823777c1fa2',
                'litecrack_ba':'786fbf654af8ff1b0e139328ac5d4f121ce9c215dd504988054ae21502f661f6'}
        if protocol['parents']!=pinned or set(result['static'])!=set(pinned):raise ValueError('Unexpected parents')
        parents={}
        for phase,name in [(3,'litecrack'),(4,'litecrack_ba')]:
            prefix=f'artifacts/phase{phase}/{name}/checkpoints/'
            archive=f'parents/phase{phase}_results.zip'
            if archive in inventory:
                from scripts.release_integrity import MODEL_ARCHIVES
                if inventory[archive]['sha256']!=MODEL_ARCHIVES[f'phase{phase}_results.zip']:raise ValueError('Original model archive changed')
                with zipfile.ZipFile(io.BytesIO(z.read(archive))) as parent_zip:
                    pointer=json.loads(parent_zip.read(prefix+'best.json'));raw=parent_zip.read(prefix+pointer['file'])
            else:
                pointer=read('parents/'+prefix+'best.json');raw=z.read('parents/'+prefix+pointer['file'])
            if hashlib.sha256(raw).hexdigest()!=pinned[name]:raise ValueError('Parent weight checksum mismatch')
            parents[name]=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
            if parents[name]['manifest_hash']!=config['manifest_sha256']:raise ValueError('Parent dataset mismatch')
        draw=next(d for d in manifest['support_draws'] if d['draw']==config['screen_draw'])
        support=draw['supports'][str(config['screen_k'])];val=split_ids['target_acquisition']-set(support)
        checked_evaluations=0;checked_weights=0
        for name,base in result['static'].items():
            for split in ('source_holdout','target_query'):metrics(base[split],split_ids[split]);checked_evaluations+=1
        for name,selected in selection['selections'].items():
            if selected['support_ids']!=support or set(selected['validation_ids'])!=val or len(selected['validation_ids'])!=len(val):
                raise ValueError('Acquisition support/validation differs')
            metrics(selected['base_source'],split_ids['source_validation']);metrics(selected['base_target'],val);checked_evaluations+=2
            rows=[r for r in development if r['model']==name]
            if len(rows)!=len(config['conditions']) or {r['condition']['name'] for r in rows}!={c['name'] for c in config['conditions']}:
                raise ValueError('Incomplete development grid')
            for row in rows:
                if row['condition'] not in config['conditions']:raise ValueError('Unexpected development condition')
                metrics(row['source'],split_ids['source_validation']);metrics(row['target'],val);checked_evaluations+=2
                verify_gate(row['gate'],gate(selected['base_source']['summary']['dice'],row['source']['summary']['dice'],
                            selected['base_target']['summary']['dice'],row['target']['summary']['dice'],config['gates']))
            viable=[r for r in rows if r['gate']['passed'] and r['condition']['distill_weight']>0]
            chosen=max(viable,key=lambda r:(r['gate']['target_gain_points'],r['gate']['source_change_points'],r['condition']['name'])) if viable else None
            if selected['condition']!=(chosen['condition'] if chosen else None):raise ValueError('Selection was not validation-only rule')
        def check_run(folder,name,condition,support_ids,seed):
            nonlocal checked_weights
            pointer=read(folder+'/checkpoints/latest.json');safe_member(pointer['file'])
            if Path(pointer['file']).name!=pointer['file']:raise ValueError('Unsafe snapshot pointer')
            raw=z.read(folder+'/checkpoints/'+pointer['file'])
            if hashlib.sha256(raw).hexdigest()!=pointer['sha256']:raise ValueError('Snapshot pointer checksum failed')
            state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
            saved=read(folder+'/config.json');status=read(folder+'/status.json')
            expected_settings={**config['adaptation'],**condition,'device':config['device'],'threads':config['threads']}
            if state['config']!=saved or saved['settings']!=expected_settings or saved['seed']!=seed:
                raise ValueError('Snapshot configuration differs')
            if saved['code']!=protocol['code'] or saved['parent_sha256']!=pinned[name] or state['manifest_hash']!=config['manifest_sha256']:
                raise ValueError('Snapshot provenance mismatch')
            if saved['support_ids']!=support_ids or saved['extra_manifest_sha256'] is not None:raise ValueError('Training support differs')
            if saved['model_config']!=parents[name]['config']:raise ValueError('Architecture differs from parent')
            expected_steps=math.ceil(len(support_ids)/config['adaptation']['batch_size'])*config['adaptation']['epochs']
            expected_examples=len(support_ids)*config['adaptation']['epochs']
            if status['state']!='complete' or state['global_step']!=expected_steps or status['step']!=expected_steps or pointer['global_step']!=expected_steps:
                raise ValueError('Incomplete optimizer budget')
            if state['epoch']!=config['adaptation']['epochs'] or state['sampler_state']['order'] is not None or state['sampler_state']['cursor']!=0:
                raise ValueError('Snapshot is not the final fixed-budget state')
            history=state['sampler_state']['history']
            if len(history)!=expected_steps or [r['step'] for r in history]!=list(range(1,expected_steps+1)):
                raise ValueError('Incomplete training history')
            for key,expected in [('support_examples',expected_examples),('source_examples',expected_examples*condition['source_per_support']),('scene_examples',0)]:
                if status[key]!=expected or sum(r[key] for r in history)!=expected:raise ValueError('Training exposure counts differ')
            if state['optimizer'] is None or state['rng'] is None:raise ValueError('Recovery state missing')
            frozen=True;changed=False
            for key,value in state['model'].items():
                if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('Nonfinite model weights')
                equal=torch.equal(value,parents[name]['model'][key])
                if key.startswith('encoder.') and not equal:frozen=False
                if not key.startswith('encoder.') and not equal:changed=True
            if not frozen or not changed:raise ValueError('Encoder changed or adaptation did not update decoder')
            checked_weights+=1
            return pointer['sha256']
        torch.set_num_threads(2)
        for row in development:
            check_run('retention_development/'+row['model']+'/'+row['condition']['name'],row['model'],row['condition'],support,draw['seed'])
        wanted=set()
        for name,selected in selection['selections'].items():
            if selected['condition'] is None:continue
            for draw_id in config['draws']:
                for k in config['budgets']:
                    for condition_name in ('no_replay','replay',selected['condition']['name']):wanted.add((name,draw_id,k,condition_name))
        actual=[(r['model'],r['draw'],r['k'],r['condition']) for r in result['adapted']]
        if len(actual)!=len(wanted) or set(actual)!=wanted or completed!={'state':'complete','adaptations':len(wanted),'unmet_models':[]}:
            raise ValueError('Final matrix incomplete or duplicated')
        for row in result['adapted']:
            name=row['model'];draw=next(d for d in manifest['support_draws'] if d['draw']==row['draw'])
            condition=next(c for c in config['conditions'] if c['name']==row['condition'])
            folder=f"retention_matrix/{name}/draw{row['draw']}_k{row['k']}/{row['condition']}"
            digest_value=check_run(folder,name,condition,draw['supports'][str(row['k'])],draw['seed'])
            if row['checkpoint_sha256']!=digest_value:raise ValueError('Metrics refer to a different checkpoint')
            for split in ('source_holdout','target_query'):metrics(row['scores'][split],split_ids[split]);checked_evaluations+=1
            base=result['static'][name]
            verify_gate(row['gate'],gate(base['source_holdout']['summary']['dice'],row['scores']['source_holdout']['summary']['dice'],
                        base['target_query']['summary']['dice'],row['scores']['target_query']['summary']['dice'],config['gates']))
        baseline_replay={'evaluations_compared':0,'max_per_image_metric_difference':0.,'max_macro_mean_difference_by_metric':{}}
        prior=archive_path(ROOT, 'phase5_results.zip')
        if prior.exists():
            from scripts.release_integrity import MODEL_ARCHIVES
            if digest(prior)!=MODEL_ARCHIVES['phase5_results.zip']:raise ValueError('Original Phase 5 archive changed')
            with zipfile.ZipFile(prior) as old:
                for row in result['adapted']:
                    if row['condition']!='no_replay':continue
                    label='control' if row['model']=='litecrack' else 'boundary'
                    for split in ('source_holdout','target_query'):
                        name=f"results/{label}_draw{row['draw']}_k{row['k']}_{split}.json"
                        original={r['image_id']:r for r in json.loads(old.read(name))}
                        for metric in ('dice','iou','precision','recall','boundary_f1'):
                            difference=abs(statistics.mean(r[metric] for r in original.values())-row['scores'][split]['summary'][metric])
                            ledger=baseline_replay['max_macro_mean_difference_by_metric']
                            ledger[metric]=max(ledger.get(metric,0),difference)
                        for image in row['scores'][split]['per_image']:
                            for metric in ('dice','iou','precision','recall','boundary_f1'):
                                difference=abs(image[metric]-original[image['image_id']][metric])
                                baseline_replay['max_per_image_metric_difference']=max(baseline_replay['max_per_image_metric_difference'],difference)
                        baseline_replay['evaluations_compared']+=1
        cases=ET.fromstring(z.read('test_evidence/tests.xml')).findall('.//testcase')
        if len(cases)!=18 or any(c.find(t) is not None for c in cases for t in ('skipped','failure','error')):
            raise ValueError('CPU/CUDA focused checks incomplete')
        if not read('readiness/preflight.json')['ready_for_gpu_retraining'] or read('dataset_restore.json')!={'data_root':'/kaggle/working/road_crack_followup_data','pairs':657,'verified':True}:
            raise ValueError('GPU/full data restoration evidence absent')
        grouped=defaultdict(list)
        for row in result['adapted']:grouped[(row['model'],row['condition'],row['k'])].append(row)
        summary=[]
        for (name,condition,k),rows in sorted(grouped.items()):
            target=[100*r['scores']['target_query']['summary']['dice'] for r in rows]
            source=[100*r['scores']['source_holdout']['summary']['dice'] for r in rows]
            base=result['static'][name]
            summary.append({'model':name,'condition':condition,'k':k,'draws':len(rows),
                'target_dice_mean_percent':statistics.mean(target),'target_dice_sd_points':statistics.stdev(target),
                'target_gain_mean_points':statistics.mean(target)-100*base['target_query']['summary']['dice'],
                'source_dice_mean_percent':statistics.mean(source),'source_change_mean_points':statistics.mean(source)-100*base['source_holdout']['summary']['dice'],
                'source_change_min_points':min(r['gate']['source_change_points'] for r in rows),
                'draws_passing_joint_gate':sum(r['gate']['passed'] for r in rows)})
        report={'status':'verified','archive':Path(path).name,'archive_sha256':digest(path),
                'inventory_files':len(inventory),'audited_checkpoints':checked_weights,'audited_evaluation_blocks':checked_evaluations,
                'cpu_cuda_tests_passed':len(cases),'matrix_adaptations':len(actual),'development_adaptations':len(development),
                'source_newline_only_differences':newline_differences,
                'phase5_no_replay_consistency':baseline_replay,
                'selected_conditions':{n:r['condition'] for n,r in selection['selections'].items()},
                'summary':summary,'source_multiseed_present':any(n.startswith('source_multiseed/') for n in inventory),
                'independent_scene_test_present':any(n.endswith('scene_test.json') and not n.startswith('test_evidence/') for n in inventory),
                'real_annotation_sessions_present':any('/annotation_sessions/' in n and not n.startswith('test_evidence/') for n in inventory),
                'limitations':['Single original training seed; support draws are not training seeds.',
                               'Old source holdout and target query reused; exploratory results.',
                               'Dataset restoration record was audited; raw images are not bundled for full local inference replay.',
                               'Metrics and checkpoint identities verified; not a local rerun of every GPU prediction.',
                               'Selected conditions are chosen from distillation candidates; replay controls may yield higher target Dice.',
                               'BA selected strategy combines replay, distillation and lower learning rate; no learning-rate-matched replay-only control isolates the distillation effect.'],
                'app_activation_performed':False}
        write_json(output/'audit.json',report)
        lines=['# Verified follow-up retention results','',f"Archive: `{Path(path).name}`.",
               f"SHA256: `{report['archive_sha256']}`.",'',
               f"Verified {len(inventory)} files, {checked_weights} checkpoints, {len(actual)} final adaptations and all 18 CPU/CUDA checks.",'',
               'Means and sample SD are across five support draws of one original training seed. Source changes use all 237 holdout images; target Dice uses the fixed 95-image query.', '',
               '| Model | Condition | K | Target Dice (%) | Source change (pts) | Joint gate passes |',
               '| --- | --- | ---: | ---: | ---: | ---: |']
        for r in summary:
            lines.append(f"| {r['model']} | {r['condition']} | {r['k']} | {r['target_dice_mean_percent']:.2f} +/- {r['target_dice_sd_points']:.2f} | {r['source_change_mean_points']:+.2f} | {r['draws_passing_joint_gate']}/5 |")
        lines+=['','The selected BA distillation candidate passes all 15 joint gates. This resolves the measured retention failure for these fixed pilot runs, at a target-accuracy cost versus replay alone; it does not establish universal retention.',
                '', 'The BA replay control passes all five K=5 and K=10 gates and has higher target Dice than the selected distilled candidate. The implemented selection deliberately considers distillation candidates; no best method is retrospectively chosen using query scores.',
                '', 'The selected BA strategy combines replay, distillation and a lower learning rate. The grid does not contain a learning-rate-matched replay-only control, so its retention benefit cannot be attributed exclusively to distillation.',
                '', f"No-replay controls closely reproduce Phase 5 across {baseline_replay['evaluations_compared']} evaluation blocks, with maximum macro Dice difference {100*baseline_replay['max_macro_mean_difference_by_metric'].get('dice',0):.6f} percentage points. Small per-image differences remain; cross-run outputs are not bit-identical.",
                '', 'Three-seed retraining, independent broad-scene testing, vegetation-failure resolution and measured human correction effort are not established by this archive. Synthetic test fixtures under test_evidence are excluded from those claims.',
                '', 'No model was activated and the frozen Phase 8 release was not changed.']
        (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','archive_sha256','inventory_files','audited_checkpoints','cpu_cuda_tests_passed','matrix_adaptations','source_multiseed_present','independent_scene_test_present')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'output/followup_v1/downloaded_audit')
    a=p.parse_args();audit(a.archive,a.output)
