"""Self-contained Phase 2 training notebook; no remote access credentials."""
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
        for p in sorted((root/directory).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json') and p.name!='build_phase2_notebook.py':
                z.write(p,p.relative_to(root).as_posix())
    for name in ('data/manifests/pilot_v1.json','data/manifests/pilot_v1.json.sha256','requirements.txt'):
        z.write(root/name,name)
payload=buffer.getvalue(); digest=hashlib.sha256(payload).hexdigest()
cells=[]
def add(kind,text):
    cell={'cell_type':kind,'id':f'phase2-{len(cells):02d}','metadata':{},'source':text.splitlines(True)}
    if kind=='code': cell.update(execution_count=None,outputs=[])
    cells.append(cell)

add('markdown','''# Phase 2 — Separate validation-only stabilization run

Run on the connected Kaggle GPU kernel, in order. This notebook embeds the source
and frozen pilot manifest. Initial runs use seed 42 and 16 base channels; they are
pilot baselines, not exact paper reproductions. Only one GPU is used deliberately.
The original Dice-only runs collapsed and are preserved. This v2 run adds
GroupNorm, BCE+Dice, gradient clipping and a lower learning rate. Holdout/query
scores are excluded from this diagnostic run. No improvement is claimed yet.

Recovery tests must pass before long training. Checkpoints are local to the live
session until you download/save outputs. Keep the session running until backup
is verified. No snapshots are automatically deleted.
''')
add('code',f'''import base64, hashlib, io, json, os, sys, zipfile
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={digest!r}
project=Path('/kaggle/working/road_crack_phase2_v2')
project.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    for name in z.namelist():
        path=(project/name).resolve()
        assert path.is_relative_to(project.resolve())
        if path.exists(): assert path.read_bytes()==z.read(name), 'Code differs: preserve this run and use a new project directory'
    z.extractall(project)
os.chdir(project)
manifest=project/'data/manifests/pilot_v1.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==manifest.with_suffix('.json.sha256').read_text().split()[0]
import subprocess
restoration_report=project/'artifacts/dataset_restore.json'
subprocess.run([sys.executable,'scripts/data/restore_datasets.py',
    '--manifest',str(manifest),'--sources','configs/data_sources.json',
    '--destination','/kaggle/working/road_crack_phase1/data/raw',
    '--report',str(restoration_report)],check=True)
data_root=Path(json.loads(restoration_report.read_text())['data_root'])
import torch
assert torch.cuda.is_available(), 'Enable Kaggle GPU'
print('GPU:',torch.cuda.get_device_name(0),'Devices:',torch.cuda.device_count(),'Torch:',torch.__version__)
print('Frozen splits:',json.loads(manifest.read_text())['counts'])
''')
add('code','''import subprocess, xml.etree.ElementTree as ET
evidence=project/'artifacts/phase2'
evidence.mkdir(parents=True,exist_ok=True)
test_run=subprocess.run([sys.executable,'-m','pytest','-q','-rA',
    '--junitxml='+str(evidence/'tests.xml')],text=True,capture_output=True)
(evidence/'tests.txt').write_text(test_run.stdout+test_run.stderr)
print(test_run.stdout,test_run.stderr)
assert test_run.returncode==0, 'Recovery tests failed; do not start training'
cases=ET.parse(evidence/'tests.xml').findall('.//testcase')
resume_cases=[c for c in cases if 'test_actual_trainer_mid_epoch_resume' in c.attrib.get('name','')]
assert len(resume_cases)==4 and all(c.find('skipped') is None for c in resume_cases), 'Both CPU and CUDA recovery tests must execute'
(evidence/'requirements.kaggle.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
subprocess.run([sys.executable,'scripts/experiments/environment.py'],check=True)
print('Actual U-Net and R2AU trainer recovery tests passed on CPU and CUDA.')
''')
add('markdown','''## Training

The following cell starts both pilot baselines (up to 100 epochs each, early
stopping patience 15). Training is FP32, deterministic, batch eight, 256 × 256.
Checkpoints are saved at each epoch and every 30 steps. Rerunning resumes an
existing checkpoint; configurations and dataset hashes must match exactly.

Interrupt requests handled by the training process stop at a complete optimizer
step. A forced kernel kill can lose progress since the latest saved checkpoint.
To stop from another cell/terminal, send SIGTERM to the printed worker PID; avoid
shutting down the kernel/session. If disk space runs low, export then resume on
adequate persistent storage; older snapshots are preserved.
''')
add('code','''import shutil
assert shutil.disk_usage('/kaggle/working').free>8*1024**3, 'Need at least 8 GiB free before the two pilot runs'
def execute_logged(arguments,log_path):
    with open(log_path,'a',buffering=1) as log:
        process=subprocess.Popen(arguments,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        print('Training worker PID:',process.pid,flush=True)
        try:
            for line in process.stdout:
                print(line,end='',flush=True); log.write(line)
            code=process.wait()
        except KeyboardInterrupt:
            import signal
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=120)
            raise
    if code: raise RuntimeError('Worker failed; inspect '+str(log_path))
for architecture in ['unet','r2au']:
    run=project/'artifacts/phase2'/architecture
    command=[sys.executable,'scripts/experiments/train_baseline.py','--config','configs/stabilized_'+architecture+'.json',
             '--manifest',str(manifest),'--data-root',str(data_root),'--run-dir',str(run)]
    if (run/'checkpoints/latest.json').exists(): command.append('--resume')
    execute_logged(command,evidence/(architecture+'_console.log'))
    status=json.loads((run/'status.json').read_text())
    if status['state']!='training_complete':
        print('Stopped safely; export checkpoints before resuming.'); break
    print('Completed',architecture,json.loads((run/'evaluation.json').read_text()))
''')
add('code','''# Profile each completed validation-best model on the same GPU and CPU.
for architecture in ['unet','r2au']:
    run=evidence/architecture
    if not (run/'evaluation.json').exists(): continue
    for device in ['cuda','cpu']:
        execute_logged([sys.executable,'scripts/evaluation/profile_baseline.py','--run-dir',str(run),
                        '--device',device],evidence/(architecture+'_profile.log'))
''')
add('markdown','''## Export before ending the session

Run this cell even after an earlier failure. It exports code, manifest, tests,
logs, evaluation, and the latest/best checkpoint snapshots for each model. Other
historical snapshots remain on Kaggle without deletion. Download the ZIP and
checksum from Kaggle Files/Output, and retain a persistent Kaggle output version.
This cell does not itself verify that an external backup was downloaded.
''')
add('code','''import zipfile,hashlib
backup=Path('/kaggle/working/phase2_stabilization_results.zip')
selected=set()
for path in project.rglob('*'):
    if not path.is_file() or '__pycache__' in path.parts or '.pytest_cache' in path.parts: continue
    if path.suffix=='.pt': continue
    selected.add(path)
for architecture in ['unet','r2au']:
    folder=evidence/architecture/'checkpoints'
    for which in ['latest','best']:
        pointer=folder/(which+'.json')
        if pointer.exists():
            entry=json.loads(pointer.read_text())
            path=folder/entry['file']
            assert path.resolve().parent==folder.resolve()
            assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
            selected.add(path)
with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(selected): z.write(path,path.relative_to(project).as_posix())
digest=hashlib.sha256(backup.read_bytes()).hexdigest()
backup.with_suffix('.zip.sha256').write_text(digest+'  '+backup.name+'\\n')
print('Download:',backup)
print('SHA256:',digest)
print('Historical snapshots remain on Kaggle. Latest and best are included in this ZIP.')
''')
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'}},'cells':cells}
out=root/'output/kaggle/Road_Crack_Phase2_Stabilization.ipynb'
out.write_text(json.dumps(notebook,indent=2),encoding='utf-8')
for c in cells:
    if c['cell_type']=='code': compile(''.join(c['source']),c['id'],'exec')
print(out)
print('Notebook code-cell syntax checks passed')
