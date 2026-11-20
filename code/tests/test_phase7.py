"""Phase 7 algorithm invariants, native/ONNX parity and replay recovery."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from inspection.advanced import entropy,rank_images,tiled_probability
from inspection.engine import infer,png_bytes
from inspection.portable import export_model
from inspection.replay import selected_source
from inspection.store import identity
from scripts.phase7_worker import run as replay_run
from tests.test_inspection import setup,upload,correction
from src.atomic import write_json


def test_entropy_and_tile_alignment():
    assert entropy(np.array([.5]))[0]==pytest.approx(1)
    assert entropy(np.array([0.,1.])).max()<.00001
    pixels=np.random.default_rng(7).integers(0,256,(49,77,3),dtype=np.uint8)
    image=Image.fromarray(pixels)
    values,meta=tiled_probability(lambda tile:np.asarray(tile)[...,0]/255,image,32,8)
    assert values.shape==(49,77)
    np.testing.assert_allclose(values,pixels[...,0]/255,atol=2e-7)
    assert meta['tiles']==6 and meta['pad_right']==3 and meta['pad_bottom']==7
    small,_=tiled_probability(lambda tile:np.ones((32,32),dtype=np.float32)*.6,Image.new('RGB',(9,13)),32,8)
    np.testing.assert_allclose(small,.6,atol=2e-7)
    with pytest.raises(ValueError,match='limit'):
        tiled_probability(lambda _:None,image,16,15,max_tiles=2)


def test_ranking_excludes_labels_query_and_validation(setup):
    client,store,project,base,version,_=setup
    a,b,c,d=[upload(client,base,seed) for seed in (11,12,13,14)]
    correction(client,base,b['id'])
    c['training_eligible']=False;c['frozen_splits']=['target_query'];store.put('image',c,project['id'])
    store.put('job',{'id':identity(),'validation_ids':[d['id']]},project['id'])
    one=rank_images(store,project['id'],version,'entropy')
    assert [r['image_id'] for r in one['ranked']]==[a['id']]
    assert 0<=one['ranked'][0]['score']<=1
    assert rank_images(store,project['id'],version,'random',42)==rank_images(store,project['id'],version,'random',42)
    response=client.post(base+'/rankings',json={'version_id':version['id']})
    assert response.status_code==201
    assert response.json()['pool_size']==1


def test_portable_parity_runtime_and_tiled_api(setup):
    client,store,project,base,version,_=setup
    image=upload(client,base,21)
    report=export_model(store,project['id'],version)
    assert report['max_probability_difference']<=.001
    assert report['audited_source_validation'] is False
    assert len(report['checks'])==4
    for mode in ('resize','tiled'):
        response=client.post(base+'/predict',json={'image_id':image['id'],'version_id':version['id'],'inference_mode':mode,'runtime':'onnx'})
        assert response.status_code==201,response.text
        value=response.json();assert value['inference']['runtime']=='onnx'
        assert value['inference']['mode']==mode and value['width']==48 and value['height']==32
    exported=client.get(base+f'/onnx/{report["id"]}/download')
    assert exported.status_code==200
    other=client.post('/api/projects',json={'name':'Other'}).json()
    assert client.get(f'/api/projects/{other["id"]}/onnx/{report["id"]}/download').status_code==404
    (store.root/'onnx_exports'/report['id']/'model.onnx').write_bytes(b'corrupted')
    assert client.get(base+f'/onnx/{report["id"]}/download').status_code==400


def fake_source_buffer(store):
    folder=store.folder('source_replay');rows=[]
    for index,split in enumerate(['source_train','source_train','source_validation']):
        image=png_bytes(Image.new('RGB',(48,32),(70+index,90,100)))
        mask=png_bytes(Image.new('L',(48,32),255))
        (folder/f'{index}.png').write_bytes(image);(folder/f'{index}_mask.png').write_bytes(mask)
        rows.append({'image_id':f'fixture-source-{index}','split':split,'image':f'{index}.png','mask':f'{index}_mask.png',
                     'image_sha256':hashlib.sha256(image).hexdigest(),'mask_sha256':hashlib.sha256(mask).hexdigest()})
    value={'records':rows};write_json(folder/'manifest.json',value)
    return folder,value


def test_replay_exact_resume_and_validation_isolation(setup,monkeypatch):
    client,store,project,base,version,_=setup
    a,b=upload(client,base,31),upload(client,base,32)
    white=np.full((32,48),255,dtype=np.uint8)
    correction(client,base,a['id'],array=white);correction(client,base,b['id'],array=white)
    folder,buffer=fake_source_buffer(store)
    monkeypatch.setattr('inspection.replay.verified_buffer',lambda _: (folder,buffer))
    body={'parent_id':version['id'],'support_ids':[a['id']],'validation_ids':[b['id']],'epochs':2,'source_replay':True}
    first=client.post(base+'/jobs',json=body)
    assert first.status_code==201,first.text
    first=first.json();replay_run(store.root,first['id'])
    second=client.post(base+'/jobs',json=body).json()
    replay_run(store.root,second['id'],stop_after_steps=1)
    assert store.get('job',second['id'])['state']=='cancelled'
    replay_run(store.root,second['id'])
    def state(job):
        p=store.root/'jobs'/job['id']/'checkpoints'
        return torch.load(p/json.loads((p/'latest.json').read_text())['file'],weights_only=False,map_location='cpu')
    full,resumed=state(first),state(second)
    assert full['sampler_state']['history']==resumed['sampler_state']['history']
    for key in full['model']:assert torch.equal(full['model'][key],resumed['model'][key])
    seen={iid for row in resumed['sampler_state']['history'] for iid in row['source_replay_ids']}
    assert seen<={'fixture-source-0','fixture-source-1'} and 'fixture-source-2' not in seen
    job=store.get('job',second['id'])
    assert job['state']=='completed' and job['validation']['encoder_unchanged']
    assert job['validation']['source_validation']['passed']
    assert store.get('project',project['id'])['active_version']==version['id']
    assert len(job['validation']['source_validation']['images'])==1


def test_source_selection_excludes_holdout_and_query():
    manifest=json.loads((Path(__file__).resolve().parents[1]/'data/manifests/pilot_v1.json').read_text())
    rows=selected_source(manifest)
    assert rows==selected_source(manifest)
    assert len({r['image_id'] for r in rows})==24
    assert sum(r['split']=='source_train' for r in rows)==16
    assert sum(r['split']=='source_validation' for r in rows)==8
    assert not any(r['split'] in ('source_holdout','target_query','target_acquisition') for r in rows)


def test_failed_portable_gate_never_registers_export(setup,monkeypatch):
    client,store,project,base,version,_=setup
    class InvalidRuntime:
        def run(self,names,inputs):
            value=inputs['image'];return [np.full((1,1,value.shape[2],value.shape[3]),.5,dtype=np.float32)]
    monkeypatch.setattr('onnxruntime.InferenceSession',lambda *args,**kwargs:InvalidRuntime())
    with pytest.raises(ValueError,match='parity gate'):
        export_model(store,project['id'],version)
    assert store.list('onnx',project['id'])==[]
