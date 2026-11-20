"""Portable Phase 3 region-only compact-model experiment."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[2]
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
    for directory in ('src','scripts','tests','configs'):
        for path in sorted((root/directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.json') and path.name not in ('build_phase3_notebook.py','build_phase4_notebook.py'):
                z.write(path,path.relative_to(root).as_posix())
    for name in ('data/manifests/pilot_v1.json','data/manifests/pilot_v1.json.sha256',
                 'data/manifests/baseline_registry_v1.json','requirements.txt'):
        z.write(root/name,name)
payload=buffer.getvalue()
cells=[]
def add(kind,text):
    cell={'cell_type':kind,'id':f'phase4-{len(cells)}','metadata':{},'source':text.splitlines(True)}
    if kind=='code': cell.update(execution_count=None,outputs=[])
    cells.append(cell)
add('markdown','''# Phase 3 â€” LiteCrack region-only baseline

This notebook embeds code, configs, frozen splits and the Phase 2 baseline registry.
Use a Kaggle GPU; enable Internet if datasets need restoring. MobileNetV3-Small
uses width multiplier 0.5, weights=None and GroupNorm, with a separable decoder.
An auxiliary boundary head adds 0.2 times boundary BCE to the region objective. No pretrained weights or target-query tuning.

The runtime must already contain compatible torch/torchvision. Do not upgrade
Kaggle's working GPU stack blindly. CPU/CUDA exact recovery, spatial shape,
gradient and parameter-budget tests gate the real training run.
''')
add('code',f'''import os,sys,json,hashlib,io,base64,zipfile,subprocess
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
project=Path('/kaggle/working/road_crack_phase4_v1')
project.mkdir(exist_ok=True)
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={hashlib.sha256(payload).hexdigest()!r}
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    for name in z.namelist():
        path=(project/name).resolve(); assert path.is_relative_to(project.resolve())
        if path.exists(): assert path.read_bytes()==z.read(name), 'Existing source differs: preserve and use a new project directory'
    z.extractall(project)
os.chdir(project)
import torch,torchvision
assert torch.cuda.is_available(), 'Enable Kaggle GPU'
print('GPU:',torch.cuda.get_device_name(0),'torch:',torch.__version__,'torchvision:',torchvision.__version__)
manifest=project/'data/manifests/pilot_v1.json'
restore_report=project/'artifacts/dataset_restore.json'
subprocess.run([sys.executable,'scripts/data/restore_datasets.py','--manifest',str(manifest),
    '--sources','configs/data_sources.json','--destination','/kaggle/working/road_crack_phase1/data/raw',
    '--report',str(restore_report)],check=True)
data_root=Path(json.loads(restore_report.read_text())['data_root'])
evidence=project/'artifacts/phase4'
evidence.mkdir(parents=True,exist_ok=True)
''')
add('code','''import xml.etree.ElementTree as ET
result=subprocess.run([sys.executable,'-m','pytest','-q','-rA','--junitxml='+str(evidence/'tests.xml')],
    text=True,capture_output=True)
(evidence/'tests.txt').write_text(result.stdout+result.stderr)
print(result.stdout,result.stderr)
assert result.returncode==0, 'Tests failed: do not start training'
cases=ET.parse(evidence/'tests.xml').findall('.//testcase')
recovery=[c for c in cases if 'test_actual_trainer_mid_epoch_resume' in c.attrib.get('name','')]
assert len(recovery)==8 and all(c.find('skipped') is None for c in recovery), 'All four model CPU/CUDA recovery tests must run'
compact=[c for c in cases if 'test_compact_' in c.attrib.get('name','')]
assert len(compact)==2 and all(c.find('skipped') is None for c in compact)
(evidence/'requirements.kaggle.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
subprocess.run([sys.executable,'scripts/experiments/environment.py'],check=True)
sys.path.insert(0,str(project))
from src.model_factory import build_model
config=json.loads((project/'configs/litecrack_ba.json').read_text())
model=build_model(config)
parameters=sum(p.numel() for p in model.parameters())
registry=json.loads((project/'data/manifests/baseline_registry_v1.json').read_text())
baseline_parameters=2110029
assert parameters<=config['max_parameters'] and parameters<=baseline_parameters/2
print('LiteCrack parameters:',parameters,'R2AU parameters:',baseline_parameters,'reduction:',1-parameters/baseline_parameters)
del model
''')
add('markdown','''## Train one fixed configuration

Same seed, split, resolution, region objective, optimizer, clipping and stopping
budget as the stabilized baselines. Validation selects the checkpoint. Target
query and source holdout are not evaluated in this development run. No claim of
superiority follows from a smaller parameter count or one seed.

Every epoch and 30 optimizer steps saves immutable recovery state. Rerunning
resumes the same configuration. Historical snapshots are never pruned. Kernel
termination can lose work since the latest saved checkpoint; download outputs.
''')
add('code','''import shutil,signal
assert shutil.disk_usage('/kaggle/working').free>2*1024**3, 'Need at least 2 GiB free'
run=evidence/'litecrack_ba'
command=[sys.executable,'scripts/experiments/train_baseline.py','--config','configs/litecrack_ba.json',
    '--manifest',str(manifest),'--data-root',str(data_root),'--run-dir',str(run)]
if (run/'checkpoints/latest.json').exists(): command.append('--resume')
def execute_logged(command,path):
    with open(path,'a',buffering=1) as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        print('Worker PID:',process.pid,flush=True)
        try:
            for line in process.stdout: print(line,end='',flush=True); log.write(line)
            exit_code=process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGTERM); process.wait(timeout=120); raise
    if exit_code: raise RuntimeError('Worker failed: '+str(path))
execute_logged(command,evidence/'training.log')
print((run/'status.json').read_text())
if (run/'evaluation.json').exists(): print((run/'evaluation.json').read_text())
''')
add('code','''assert json.loads((run/'status.json').read_text())['state']=='training_complete', 'Export safe-stop state before resuming'
for device in ['cuda','cpu']:
    execute_logged([sys.executable,'scripts/evaluation/profile_baseline.py','--run-dir',str(run),'--device',device],
                   evidence/('profile_'+device+'.log'))
''')
add('markdown','''## Export before ending the session

Run even if an earlier cell failed. The ZIP includes code, configs, tests, metrics,
and latest/best snapshots. Other historical snapshots remain on Kaggle; no files
are deleted. Download through Kaggle Files and save a persistent notebook output.
The working directory alone is not an external backup.
''')
add('code','''backup=Path('/kaggle/working/phase4_results.zip')
selected={p for p in project.rglob('*') if p.is_file() and p.suffix!='.pt'
          and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
folder=evidence/'litecrack_ba/checkpoints'
for which in ['latest','best']:
    pointer=folder/(which+'.json')
    if pointer.exists():
        entry=json.loads(pointer.read_text()); path=(folder/entry['file']).resolve()
        assert path.parent==folder.resolve()
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        selected.add(path)
with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(selected): z.write(p,p.relative_to(project).as_posix())
digest=hashlib.sha256(backup.read_bytes()).hexdigest()
backup.with_suffix('.zip.sha256').write_text(digest+'  '+backup.name+'\\n')
print('Download:',backup)
print('SHA256:',digest)
''')
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
for c in cells:
    if c['cell_type']=='code': compile(''.join(c['source']),c['id'],'exec')
out=root/'output/kaggle/Road_Crack_Phase4_Boundary.ipynb'
out.write_text(json.dumps(notebook,indent=2),encoding='utf-8')
print(out)
print('Notebook code-cell syntax checks passed')
