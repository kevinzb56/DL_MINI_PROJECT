"""Real-data functional validation of Phase 7, isolated from research results."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import urllib.parse
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
from PIL import Image
import torch
from fastapi.testclient import TestClient
from inspection.api import create_app
from inspection.advanced import infer_advanced
from inspection.engine import load_state,model_config,render,save_png
from inspection.portable import export_model
from inspection.store import identity,now
from scripts.prepare_inspection_models import prepare as prepare_models
from scripts.prepare_source_replay import prepare as prepare_replay
from scripts.inspection_worker import run as no_replay_run
from scripts.phase7_worker import run as replay_run
from src.atomic import write_json
from src.model_factory import build_model


def main():
    root=Path(__file__).resolve().parents[2]
    output=root/'output/phase7';output.mkdir(exist_ok=True)
    workspace=root/'artifacts/phase7/validation-workspace'
    prepare_models(root,workspace);prepare_replay(workspace)
    app=create_app(workspace,start_workers=False);store=app.state.store;client=TestClient(app)
    project=client.post('/api/projects',json={'name':'Phase 7 real-data functional validation'}).json()
    base=f'/api/projects/{project["id"]}'
    versions={v['phase']:v for v in store.list('version') if v['phase'] in (3,4)}
    manifest=json.loads((root/'data/manifests/pilot_v1.json').read_text())
    supports=manifest['support_draws'][0]['supports']['5']
    remaining=sorted((r['image_id'] for r in manifest['records'] if r['split']=='target_acquisition' and r['image_id'] not in supports))[:5]
    selected=supports+remaining
    inputs=root/'artifacts/phase7/validation-inputs';inputs.mkdir(exist_ok=True)
    ids={};pairs=[]
    for iid in selected:
        row=next(r for r in manifest['records'] if r['image_id']==iid)
        content={}
        for field in ('image','mask'):
            target=inputs/(row[field+'_sha256']+Path(row[field]).suffix)
            if not target.exists():
                relative=row[field].removeprefix('CrackMap/')
                url='https://raw.githubusercontent.com/ikatsamenis/CrackMap/bd8fbd7789993cf278e6118836071f2ed32acffd/'+urllib.parse.quote(relative)
                with urllib.request.urlopen(url,timeout=30) as response:data=response.read()
                if hashlib.sha256(data).hexdigest()!=row[field+'_sha256']:raise ValueError('Pinned target checksum mismatch')
                target.write_bytes(data)
            data=target.read_bytes()
            if hashlib.sha256(data).hexdigest()!=row[field+'_sha256']:raise ValueError('Existing target bytes changed')
            content[field]=data
        upload=client.post(base+'/images',files={'file':(Path(row['image']).name,content['image'])})
        if upload.status_code!=201:raise ValueError(upload.text)
        ids[iid]=upload.json()['id'];pairs.append((ids[iid],content['mask']))
    rankings={}
    for policy in ('random','entropy'):
        response=client.post(base+'/rankings',json={'version_id':versions[4]['id'],'policy':policy})
        if response.status_code!=201:raise ValueError(response.text)
        rankings[policy]=response.json()
    for iid,mask in pairs:
        saved=client.post(base+f'/images/{iid}/corrections?complete=true',files={'file':('mask.png',mask)})
        if saved.status_code!=201:raise ValueError(saved.text)
    reports=[]
    for phase in (3,4):
        for replay in (False,True):
            body={'parent_id':versions[phase]['id'],'support_ids':[ids[i] for i in supports],
                  'validation_ids':[ids[i] for i in remaining],'epochs':1,'seed':42,'source_replay':replay}
            response=client.post(base+'/jobs',json=body)
            if response.status_code!=201:raise ValueError(response.text)
            job=response.json();(replay_run if replay else no_replay_run)(workspace,job['id'])
            final=store.get('job',job['id']);reports.append(final)
            if final['state']!='completed':raise ValueError('Functional replay job failed')
            write_json(output/'real_replay_jobs.json',reports)
    exports=[]
    for phase in (3,4):
        report=export_model(store,project['id'],versions[phase]);exports.append(report)
        write_json(output/f'onnx_phase{phase}_parity.json',report)
    # Fresh native-only model-pass timing, excluding IO, model loading and resizing.
    benchmark=[]
    torch.set_num_threads(2)
    for phase in (3,4):
        state=load_state(store,versions[phase]);config=model_config(state)
        model=build_model(config).eval();model.load_state_dict(state['model'])
        x=torch.zeros(1,3,config['image_size'],config['image_size'])
        with torch.inference_mode():
            for _ in range(20):model(x)
            samples=[]
            for _ in range(200):
                start=time.perf_counter();model(x);samples.append((time.perf_counter()-start)*1000)
        benchmark.append({'phase':phase,'checkpoint_sha256':versions[phase]['sha256'],'runtime':str(torch.__version__),
                          'threads':2,'warmups':20,'passes':200,'p50_ms':float(np.percentile(samples,50)),
                          'p95_ms':float(np.percentile(samples,95)),'gate_p95_le_200ms':bool(np.percentile(samples,95)<=200),
                          'scope':'CPU model execution only, FP32, batch 1, 256 square; excludes loading/preprocessing/rendering'})
    # Wide road-photo diagnostic uses no labels or adaptation on that photograph.
    wide=next((p for p in (root/'artifacts/phase6/workspace/images').glob('*/original.png') if Image.open(p).size==(480,640)),None)
    diagnostics=[]
    if wide:
        image=Image.open(wide).convert('RGB')
        for mode in ('resize','tiled'):
            values,_,elapsed,details=infer_advanced(store,versions[4],image,mode)
            for name,view in render(image,values,.5,.5).items():save_png(output/f'wide_{mode}_{name}.png',view)
            diagnostics.append({'mode':mode,'predicted_area_percent':float((values>=.5).mean()*100),'latency_ms':elapsed,'inference':details,'accuracy':'Not measured: no ground truth'})
    report={'scope':'Phase 7 functional real-data pilot; not a new full E6/E7 benchmark or independent query result.',
            'workspace':str(workspace),'project_id':project['id'],'target_support_ids':supports,'target_local_validation_ids':remaining,
            'protocol':'Draw 0 K=5, one support pass, source validation subset 8; fixed before results',
            'rankings':rankings,'jobs':reports,'onnx_exports':exports,'benchmark':benchmark,'wide_scene_diagnostics':diagnostics,
            'hardware':{'platform':platform.platform(),'processor':platform.processor(),'python':platform.python_version()},
            'limits':['No annotation time was measured; existing dataset labels were used.',
                      'Replay smoke uses one draw, one seed and one support pass; no claim of general retention improvement.',
                      'Tiling is not road-region segmentation and may retain background false positives.']}
    if platform.system()=='Windows':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            report['hardware']['cpu_model']=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    write_json(output/'real_validation.json',report)
    from scripts.evaluation.evaluate_phase7_retention import evaluate
    evaluate(output/'real_validation.json')
    print(json.dumps({'jobs':[(j['settings'].get('source_replay',False),j['validation']['passed']) for j in reports],
                      'onnx_max_errors':[e['max_probability_difference'] for e in exports],'benchmark':benchmark},indent=2))


if __name__=='__main__':main()
