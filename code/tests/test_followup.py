import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from followup.adaptation import adapt,sha,select_rows,distillation_loss
from followup.scenes import audit_scenes,scene_gate,mask_metrics,score_scenes
from followup.study import choose,gate,develop,confirm
from src.model_factory import build_model


@pytest.fixture
def pilot(tmp_path):
    torch.set_num_threads(1)
    rows=[];rng=np.random.default_rng(91)
    for split,n in [('source_train',3),('source_validation',2),('target_acquisition',4),('target_query',1),('source_holdout',1)]:
        for i in range(n):
            name=f'{split}_{i}';image=tmp_path/(name+'.png');mask=tmp_path/(name+'_mask.png')
            Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(image)
            truth=np.zeros((32,32),np.uint8);truth[:,4+i:7+i]=255;Image.fromarray(truth).save(mask)
            rows.append({'image_id':name,'split':split,'image':image.name,'mask':mask.name,
                         'image_sha256':sha(image),'mask_sha256':sha(mask)})
    return {'records':rows,'support_draws':[{'draw':0,'seed':17,'supports':{'1':['target_acquisition_0'],'2':['target_acquisition_0','target_acquisition_1']}}]}


def parent(architecture='litecrack_ba'):
    torch.manual_seed(88)
    config={'architecture':architecture,'encoder_width_mult':.5,'image_size':32,'boundary_weight':.2}
    return {'config':config,'model':copy.deepcopy(build_model(config).state_dict())}


def settings(device='cpu'):
    return {'source_per_support':1,'distill_weight':2.,'learning_rate':.0001,
            'epochs':2,'batch_size':1,'gradient_clip_norm':1.,'device':device,'threads':1}


@pytest.mark.parametrize('architecture',['litecrack','litecrack_ba'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_followup_exact_resume_and_teacher_frozen(tmp_path,pilot,architecture,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    state=parent(architecture);before=copy.deepcopy(state['model'])
    ids=['target_acquisition_0','target_acquisition_1']
    full=adapt(state,'fixture',pilot,'fixture',tmp_path,ids,settings(device),42,tmp_path/'full')
    assert adapt(state,'fixture',pilot,'fixture',tmp_path,ids,settings(device),42,tmp_path/'resumed',stop_after_steps=1) is None
    resumed=adapt(state,'fixture',pilot,'fixture',tmp_path,ids,settings(device),42,tmp_path/'resumed',resume=True)
    for k,v in full.state_dict().items():
        assert torch.equal(v,resumed.state_dict()[k])
        assert torch.equal(state['model'][k],before[k])
        if k.startswith('encoder.'):assert torch.equal(v.cpu(),before[k])
    assert any(not torch.equal(v.cpu(),before[k]) for k,v in full.state_dict().items() if not k.startswith('encoder.'))
    assert json.loads((tmp_path/'full/status.json').read_text())['source_examples']==4


def test_split_and_checksum_guards(tmp_path,pilot):
    with pytest.raises(ValueError,match='split violation'):select_rows(pilot,['target_query_0'],{'target_acquisition'})
    with pytest.raises(ValueError,match='split violation'):select_rows(pilot,['source_validation_0'],{'source_train'})
    (tmp_path/'target_acquisition_0.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):adapt(parent(),'fixture',pilot,'fixture',tmp_path,['target_acquisition_0'],settings(),42,tmp_path/'run')


def test_kl_matches_parent_and_teacher_has_no_gradient():
    teacher=torch.tensor([[-2.,0.,2.]],requires_grad=True)
    student=teacher.detach().clone().requires_grad_()
    assert float(distillation_loss(student,teacher).detach())==pytest.approx(0,abs=1e-6)
    shifted=student+1;loss=distillation_loss(shifted,teacher);loss.backward()
    assert float(loss.detach())>0 and teacher.grad is None and student.grad is not None


def test_selection_cannot_reward_unchanged_parent_or_forgetting():
    gates={'max_source_drop_points':2.,'min_target_gain_points':0.}
    assert not gate(.8,.8,.4,.4,gates)['passed']
    assert not gate(.8,.77,.4,.6,gates)['passed']
    rows=[{'condition':{'name':'fail','distill_weight':1},'gate':gate(.8,.77,.4,.6,gates)}]
    assert choose(rows) is None
    rows.append({'condition':{'name':'valid','distill_weight':1},'gate':gate(.8,.79,.4,.5,gates)})
    assert choose(rows)['condition']['name']=='valid'
    rows[-1]['scene_gate']={'passed':False};assert choose(rows) is None


def scenes(tmp_path,pilot):
    rows=[]
    for i,split in enumerate(['scene_train','scene_validation','scene_test']):
        image=tmp_path/f'scene{i}.png';mask=tmp_path/f'scene{i}_mask.png';road=tmp_path/f'scene{i}_road.png'
        Image.fromarray(np.full((32,32,3),40+i,np.uint8)).save(image)
        truth=np.zeros((32,32),np.uint8);truth[16:18,8:20]=255;Image.fromarray(truth).save(mask)
        roi=np.zeros((32,32),np.uint8);roi[12:,:]=255;Image.fromarray(roi).save(road)
        row={'image_id':f'scene{i}','split':split,'site_id':f'site{i}','group_id':f'capture{i}',
             'annotation_complete':True,'annotation_origin':'human'}
        for field,path in [('image',image),('mask',mask),('road_mask',road)]:row[field]=path.name;row[field+'_sha256']=sha(path)
        rows.append(row)
    data={'protocol':'followup_external_scenes_v1','usage_terms':'Synthetic test fixture only',
          'physical_site_provenance':'Fixture grouping only; not real field evidence','records':rows}
    path=tmp_path/'scenes.json';path.write_text(json.dumps(data));return path,data


def test_external_scene_site_and_label_guards(tmp_path,pilot):
    path,data=scenes(tmp_path,pilot);assert len(audit_scenes(path,tmp_path,pilot)['records'])==3
    data['records'][1]['site_id']='site0';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='leakage'):audit_scenes(path,tmp_path,pilot)
    path,data=scenes(tmp_path,pilot);data['records'][0]['image_sha256']=pilot['records'][0]['image_sha256'];path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='independent'):audit_scenes(path,tmp_path,pilot)
    path,data=scenes(tmp_path,pilot);data['records'][0]['annotation_origin']='prediction';path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='annotations'):audit_scenes(path,tmp_path,pilot)


def test_scene_study_requires_predeclared_sample_sizes(tmp_path,pilot):
    from followup.scenes import require_scene_size
    _,data=scenes(tmp_path,pilot)
    with pytest.raises(ValueError,match='Insufficient scene_test'):
        require_scene_size(data,{'scene_test':{'images':30,'sites':5}})


def test_no_cracks_prediction_cannot_pass_scene_gate():
    truth=np.zeros((4,4),bool);truth[2,1]=True;road=np.zeros_like(truth);road[2:]=True
    baseline=truth.copy();baseline[0,0]=True
    base={'summary':mask_metrics(baseline,truth,road)}
    zero={'summary':mask_metrics(np.zeros_like(truth),truth,road)}
    gates={'min_relative_nonroad_fp_reduction':.25,'max_recall_drop_points':2.,'max_dice_drop_points':2.}
    assert not scene_gate(base,zero,gates)['passed']
    assert scene_gate(base,{'summary':mask_metrics(truth,truth,road)},gates)['passed']


@pytest.mark.parametrize('mode',['resize','tiles'])
def test_scene_inference_keeps_original_alignment(tmp_path,pilot,mode):
    path,data=scenes(tmp_path,pilot)
    result=score_scenes(parent(),data,tmp_path,'scene_validation','cpu',mode)
    assert result['n']==1 and result['summary']['nonroad_fp_fraction_n']==1
    assert 0<=result['summary']['dice']<=1


def test_background_training_exact_recovery(tmp_path,pilot):
    path,data=scenes(tmp_path,pilot);extra={'manifest':path,'root':tmp_path,'sha256':sha(path)}
    state=parent();ids=['target_acquisition_0','target_acquisition_1']
    full=adapt(state,'fixture',pilot,'fixture',tmp_path,ids,settings(),42,tmp_path/'full',extra=extra)
    adapt(state,'fixture',pilot,'fixture',tmp_path,ids,settings(),42,tmp_path/'partial',stop_after_steps=1,extra=extra)
    recovered=adapt(state,'fixture',pilot,'fixture',tmp_path,ids,settings(),42,tmp_path/'partial',resume=True,extra=extra)
    assert all(torch.equal(v,recovered.state_dict()[k]) for k,v in full.state_dict().items())
    assert json.loads((tmp_path/'full/status.json').read_text())['scene_examples']==4


def test_validation_only_search_and_protocol_lock(tmp_path,pilot,monkeypatch):
    import followup.study as study
    config={'adaptation':{'epochs':1,'batch_size':2,'gradient_clip_norm':1.},'device':'cpu','threads':1,
            'screen_draw':0,'screen_k':1,'draws':[0],'budgets':[1],
            'conditions':[{'name':'no_replay','source_per_support':0,'distill_weight':0.,'learning_rate':.0001},
                          {'name':'replay','source_per_support':1,'distill_weight':0.,'learning_rate':.0001},
                          {'name':'distill','source_per_support':1,'distill_weight':1.,'learning_rate':.0001}],
            'gates':{'max_source_drop_points':2.,'min_target_gain_points':0.}}
    state=parent();parents={'ba':(state,'fixture')};seen=[];real=study.score
    def spy(*args,**kwargs):
        seen.append(args[3]);assert args[3] not in ('target_query','source_holdout')
        return real(*args,**kwargs)
    monkeypatch.setattr(study,'score',spy)
    lock=develop(parents,pilot,'fixture',tmp_path,config,tmp_path/'dev')
    assert seen and lock['selections']['ba']['validation_ids']==['target_acquisition_1','target_acquisition_2','target_acquisition_3']
    altered=copy.deepcopy(config);altered['gates']['max_source_drop_points']=20.
    with pytest.raises(ValueError,match='Frozen study'):develop(parents,pilot,'fixture',tmp_path,altered,tmp_path/'dev')


def test_annotation_timer_rejects_imports_and_repeated_finish(tmp_path,pilot,monkeypatch):
    import scripts.operations.followup_annotation as timer
    ticks=iter([1_000_000_000,2_000_000_000])
    monkeypatch.setattr(timer.time,'monotonic_ns',lambda:next(ticks))
    from scripts.operations.followup_annotation import start,finish
    session=start(tmp_path/'sessions',tmp_path/'target_acquisition_0.png','fixture','fixture','task')
    with pytest.raises(ValueError,match='Imported'):finish(session,tmp_path/'target_acquisition_0_mask.png','existing_dataset')
    result=finish(session,tmp_path/'target_acquisition_0_mask.png','human_corrected')
    assert result['elapsed_seconds']==1. and result['image_sha256']==sha(tmp_path/'target_acquisition_0.png')
    with pytest.raises(ValueError,match='already ended'):finish(session,tmp_path/'target_acquisition_0_mask.png','human_corrected')


def test_backup_contains_both_pointers_and_refuses_overwrite(tmp_path,pilot):
    from scripts.operations.package_followup import package
    import zipfile
    run=tmp_path/'run'
    adapt(parent(),'fixture',pilot,'fixture',tmp_path,['target_acquisition_0'],settings(),42,run)
    backup=tmp_path/'backup.zip';report=package(run,backup)
    with zipfile.ZipFile(backup) as z:
        inventory=json.loads(z.read('FOLLOWUP_BACKUP_MANIFEST.json'))['files']
        pointer=json.loads(z.read('checkpoints/latest.json'))
        assert 'checkpoints/'+pointer['file'] in inventory
    assert report['sha256']==sha(backup)
    with pytest.raises(FileExistsError):package(run,backup)


def test_summary_does_not_pool_support_draws_as_training_seeds():
    from scripts.evaluation.summarize_followup import summarize
    rows=[]
    for seed,values in [(42,[.4,.6]),(73,[.6,.8]),(101,[.5,.7])]:
        for draw,value in enumerate(values):
            rows.append({'model':f'litecrack_seed{seed}','condition':'distill','draw':draw,'k':5,
                         'scores':{s:{'summary':{'dice':value}} for s in ('source_holdout','target_query')}})
    result=summarize({'adapted':rows})['rows']
    assert all(r['training_seeds']==3 for r in result)
    assert all(r['mean_of_seed_means']==pytest.approx(.6) for r in result)
    assert all(r['training_seed_sd']==pytest.approx(.1) for r in result)


def test_confirmation_matrix_finishes_before_test_scores(tmp_path,pilot,monkeypatch):
    import followup.study as study
    config={'adaptation':{'epochs':1,'batch_size':2,'gradient_clip_norm':1.},'device':'cpu','threads':1,
            'screen_draw':0,'screen_k':1,'draws':[0],'budgets':[1],
            'conditions':[{'name':'no_replay','source_per_support':0,'distill_weight':0.,'learning_rate':.0001},
                          {'name':'replay','source_per_support':1,'distill_weight':0.,'learning_rate':.0001},
                          {'name':'distill','source_per_support':1,'distill_weight':1.,'learning_rate':.0001}],
            'gates':{'max_source_drop_points':2.,'min_target_gain_points':0.}}
    state=parent();parents={'ba':(state,'fixture')}
    # Force a selected fixture to exercise the final path, without claiming real gate success.
    monkeypatch.setattr(study,'choose',lambda rows:rows[-1])
    develop(parents,pilot,'fixture',tmp_path,config,tmp_path/'dev')
    real=study.score;seen=[]
    def spy(*args,**kwargs):
        if args[3] in ('target_query','source_holdout'):
            statuses=list((tmp_path/'matrix').rglob('status.json'))
            assert len(statuses)==3 and all(json.loads(p.read_text())['state']=='complete' for p in statuses)
            seen.append(args[3])
        return real(*args,**kwargs)
    monkeypatch.setattr(study,'score',spy)
    result=confirm(parents,pilot,'fixture',tmp_path,config,tmp_path/'dev',tmp_path/'matrix')
    assert len(result['adapted'])==3 and seen
