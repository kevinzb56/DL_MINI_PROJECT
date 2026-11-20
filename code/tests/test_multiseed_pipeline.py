import copy
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import pytest
import torch
from scripts.experiments.multiseed_pipeline import verify_backup,restore,latest_backup,validate_source_index,execute,WorkerStillRunning
from scripts.operations.followup_study_inputs import initialize,freeze_manifest,annotation_report
from scripts.release_integrity import digest


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def make_backup(path,files,created='2026-11-01T00:00:00Z'):
    inventory={n:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for n,v in files.items()}
    with zipfile.ZipFile(path,'x') as z:
        for name,value in files.items():z.writestr(name,value)
        z.writestr('FOLLOWUP_BACKUP_MANIFEST.json',json.dumps({'created':created,'files':inventory}))


def test_restore_supports_zip_and_extracted_inputs(tmp_path):
    plan={'protocol':'fixture'};files={'MULTISEED_PLAN.json':json.dumps(plan).encode(),'checkpoints/state.pt':b'fixture'}
    archive=tmp_path/'multiseed_1.zip';make_backup(archive,files)
    assert restore(archive,tmp_path/'restored',plan)['files']==2
    extracted=tmp_path/'extracted'
    with zipfile.ZipFile(archive) as z:z.extractall(extracted)
    assert restore(extracted,tmp_path/'restored2',plan)['source_form']=='directory'
    assert (tmp_path/'restored2/checkpoints/state.pt').read_bytes()==b'fixture'


def test_restore_detects_all_conflicts_before_writing(tmp_path):
    plan={'protocol':'fixture'};archive=tmp_path/'backup.zip'
    make_backup(archive,{'MULTISEED_PLAN.json':json.dumps(plan).encode(),'first.json':b'first','last.json':b'last'})
    destination=tmp_path/'restore';destination.mkdir();(destination/'last.json').write_bytes(b'newer')
    with pytest.raises(ValueError,match='Existing result differs'):restore(archive,destination,plan)
    assert not (destination/'first.json').exists() and (destination/'last.json').read_bytes()==b'newer'


@pytest.mark.parametrize('name',['../escape','/absolute','C:drive','name.'])
def test_restore_rejects_unsafe_backup_paths(tmp_path,name):
    archive=tmp_path/'bad.zip';make_backup(archive,{name:b'unsafe'})
    with pytest.raises(ValueError,match='Unsafe'):verify_backup(archive)


def test_backup_hash_tamper_and_protocol_mismatch_rejected(tmp_path):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'x') as z:
        z.writestr('MULTISEED_PLAN.json','{}')
        z.writestr('FOLLOWUP_BACKUP_MANIFEST.json',json.dumps({'files':{'MULTISEED_PLAN.json':{'bytes':2,'sha256':'0'*64}}}))
    with pytest.raises(ValueError,match='checksum'):verify_backup(archive)
    archive=tmp_path/'other.zip';make_backup(archive,{'MULTISEED_PLAN.json':b'{}'})
    with pytest.raises(ValueError,match='different multi-seed'):restore(archive,tmp_path/'target',{'protocol':'other'})


def test_latest_backup_selects_only_matching_study(tmp_path):
    plan={'protocol':'fixture'};other={'protocol':'other'};paths=[]
    for i,p in enumerate([plan,plan,other]):
        path=tmp_path/f'multiseed_{i}.zip';make_backup(path,{'MULTISEED_PLAN.json':json.dumps(p).encode()},f'2026-11-0{i+1}T00:00:00Z');paths.append(path)
    assert latest_backup(paths,plan)==paths[1]


@pytest.fixture
def source_matrix(tmp_path):
    config={'source_models':['litecrack','litecrack_ba','stabilized_r2au'],'training_seeds':[42,73,101],
            'manifest_sha256':'fixture','device':'cpu'}
    rows=[]
    for name in config['source_models']:
        write(tmp_path/'configs'/f'{name}.json',{'architecture':name,'epochs':1,'device':'cpu','seed':42})
    for seed in config['training_seeds']:
        for name in config['source_models']:
            folder=tmp_path/'source_multiseed'/f'seed{seed}'/name/'checkpoints';folder.mkdir(parents=True)
            saved={'architecture':name,'epochs':1,'device':'cpu','seed':seed,
                   'protocol':'followup_multiseed_source_v1','evaluation_splits':['source_validation']}
            torch.save({'config':saved,'manifest_hash':'fixture','model':{'fixture':torch.zeros(1)}},folder/'best.pt')
            sha=digest(folder/'best.pt');write(folder/'best.json',{'file':'best.pt','sha256':sha})
            write(folder.parent/'status.json',{'state':'training_complete'})
            write(folder.parent/'config.json',saved)
            rows.append({'name':name,'seed':seed,'checkpoint_dir':folder.relative_to(tmp_path/'source_multiseed').as_posix(),'sha256':sha,'which':'best'})
    write(tmp_path/'source_multiseed/parents.json',{'protocol':'followup_multiseed_source_v1','manifest_sha256':'fixture','entries':rows,'complete':True})
    return tmp_path,config


def test_parent_index_requires_all_nine_matched_runs(source_matrix):
    project,config=source_matrix
    assert len(validate_source_index(project,config)['entries'])==9
    path=project/'source_multiseed/parents.json';index=json.loads(path.read_text());index['entries'].pop();write(path,index)
    with pytest.raises(ValueError,match='All nine'):validate_source_index(project,config)


def test_parent_index_rejects_duplicate_seed_and_wrong_checkpoint(source_matrix):
    project,config=source_matrix;path=project/'source_multiseed/parents.json';index=json.loads(path.read_text())
    original=copy.deepcopy(index);index['entries'][1]=copy.deepcopy(index['entries'][0]);write(path,index)
    with pytest.raises(ValueError,match='Duplicate'):validate_source_index(project,config)
    write(path,original)
    folder=project/'source_multiseed'/original['entries'][0]['checkpoint_dir'];(folder/'best.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='checksum'):validate_source_index(project,config)


def test_safe_stop_still_exports_and_never_marks_stage_complete(source_matrix,monkeypatch):
    import scripts.experiments.multiseed_pipeline as pipeline
    project,config=source_matrix;monkeypatch.setattr(pipeline,'verify_project',lambda _:config)
    path=project/'source_multiseed/parents.json';index=json.loads(path.read_text());index['entries']=[r for r in index['entries'] if r['seed']!=42];index['complete']=False;write(path,index)
    exports=[]
    with pytest.raises(RuntimeError,match='stopped safely'):
        execute(project,project/'raw','train',42,project.parent,worker=lambda *_:None,exporter=lambda *args:exports.append(args))
    assert len(exports)==1 and not (project/'COMPLETED_seed42.json').exists()


def test_source_block_exports_after_recording_completion(source_matrix,monkeypatch):
    import scripts.experiments.multiseed_pipeline as pipeline
    project,config=source_matrix;monkeypatch.setattr(pipeline,'verify_project',lambda _:config)
    def export(*_):assert (project/'COMPLETED_seed42.json').exists()
    execute(project,project/'raw','train',42,project.parent,worker=lambda *_:None,exporter=export)


def test_no_export_while_worker_might_still_be_running(source_matrix,monkeypatch):
    import scripts.experiments.multiseed_pipeline as pipeline
    project,config=source_matrix;monkeypatch.setattr(pipeline,'verify_project',lambda _:config);exports=[]
    def worker(*_):raise WorkerStillRunning('fixture')
    with pytest.raises(WorkerStillRunning):execute(project,project/'raw','train',42,project.parent,worker=worker,exporter=lambda *args:exports.append(args))
    assert not exports


def test_empty_data_intake_never_claims_a_real_study(tmp_path):
    report=initialize(tmp_path);assert report['status']=='awaiting_real_data'
    with (tmp_path/'scene_index.csv').open() as stream:assert not list(csv.DictReader(stream))
    assert annotation_report(tmp_path/'annotation_sessions')['complete_sessions']==0
    with pytest.raises(ValueError,match='No real scene'):freeze_manifest(tmp_path,tmp_path/'scene_index.csv',tmp_path/'usage_terms.txt',tmp_path/'physical_sites.txt',tmp_path/'frozen.json')
    assert not (tmp_path/'frozen.json').exists()


def test_annotation_imports_never_become_human_effort(tmp_path):
    session={'id':'one','started_utc':'fixture','state':'complete','annotation_origin':'existing_dataset'}
    write(tmp_path/'one.json',session)
    with pytest.raises(ValueError,match='Imported labels'):annotation_report(tmp_path)
