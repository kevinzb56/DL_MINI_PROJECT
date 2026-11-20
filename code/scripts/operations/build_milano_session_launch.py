"""Build credential-free launch/status code for an existing Kaggle session."""
import ast
import argparse
import base64
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--resume-package',type=Path)
parser.add_argument('--python-executable')
parser.add_argument('--allow-python-build-metadata-change',action='store_true')
args=parser.parse_args()
out=root/'output/kaggle'
notebook=json.loads((out/'Road_Crack_Milano_v1.ipynb').read_text())
code=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
export_now='''import hashlib, json, zipfile
from pathlib import Path
ROOT=Path('/kaggle/working/cracklab_milano_v1')
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): h.update(block)
    return h.hexdigest()
'''+code[2]+'''
for arch in ('litecrack','litecrack_ba'):
    if (ROOT/'artifacts'/arch/'checkpoints/latest.json').exists():
        export_recovery(arch)
        print('Manual recovery export completed',flush=True)
        break
else:
    raise RuntimeError('No complete checkpoint yet')
'''
ast.parse(export_now)
(out/'milano_export_now.py').write_text(export_now,encoding='utf-8')
bootstrap=code[0].replace("list(Path('/kaggle/input').rglob('Milano_Prepared_v1.zip'))",
                        "[Path('/kaggle/working/Milano_Prepared_v1.zip')]")
source_bootstrap=bootstrap.split('archives=')[0]
if args.resume_package:
    package=args.resume_package
    digest=hashlib.sha256(package.read_bytes()).hexdigest()
    bootstrap=bootstrap.split('archives=')[0]+f'''
import urllib.request, time
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): h.update(block)
    return h.hexdigest()
restore=Path('/kaggle/working')/{package.name!r}
assert file_sha(restore)=={digest!r}, 'Restoration package checksum mismatch'
with zipfile.ZipFile(restore) as z:
    assert z.testzip() is None, 'Restoration package corrupt'
    for name in z.namelist():
        target=(ROOT/name).resolve()
        assert target.is_relative_to(ROOT.resolve()), 'Unsafe restore path'
        if target.exists():
            assert target.read_bytes()==z.read(name), 'Restore conflicts with existing state'
    z.extractall(ROOT)
DATA=ROOT/'data'; manifest_path=DATA/'manifest.json'
identity=json.loads((ROOT/'restoration_identity.json').read_text())
assert file_sha(manifest_path)==identity['manifest_sha256']
manifest=json.loads(manifest_path.read_text())
assert manifest['status']=='FROZEN_MILANO_V1'
# Fetch original publisher bytes, while retaining exact normalized masks from the backup pack.
source=Path('/kaggle/working/milano_publisher_source.zip')
if not source.exists():
    partial=source.with_suffix('.zip.partial')
    url='https://zenodo.org/records/20040434/files/Pavement_Crack_Segmentation_Dataset.zip?download=1'
    for attempt in range(3):
        try:
            size=0
            with urllib.request.urlopen(url,timeout=45) as response, partial.open('wb') as f:
                for block in iter(lambda:response.read(4*1024**2),b''):
                    f.write(block); size+=len(block)
                    if size%(64*1024**2)==0: print('Publisher download bytes:',size,flush=True)
            partial.replace(source); break
        except Exception as exc:
            print('Publisher download retry:',type(exc).__name__,attempt+1,flush=True)
            if attempt==2: raise
            time.sleep(5)
h=hashlib.md5()
with source.open('rb') as f:
    for block in iter(lambda:f.read(1024**2),b''): h.update(block)
assert h.hexdigest()=='4b530eba05708a195355a0064e520db4', 'Publisher archive checksum mismatch'
with zipfile.ZipFile(source) as z:
    for row in manifest['records']:
        target=(DATA/row['image']).resolve()
        assert target.is_relative_to(DATA.resolve())
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(row['image']) as incoming, target.open('wb') as outgoing:
                shutil.copyfileobj(incoming,outgoing)
from src.dataset import verify_files
verify_files(manifest['records'],DATA)
print('Restored exact data and checkpoint state:',len(manifest['records']),'pairs',flush=True)
'''
tests=(root/'tests/test_phase2.py').read_text()
preflight=source_bootstrap+f"\n(ROOT/'tests').mkdir(exist_ok=True)\n(ROOT/'tests/test_phase2.py').write_text({tests!r})\n"+'''
result=subprocess.run([sys.executable,'-m','pytest','tests/test_phase2.py','-q','--tb=short',
    '-p','no:cacheprovider','--basetemp',str(ROOT/'gpu_preflight_test_tmp')],
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
print(result.stdout,flush=True)
assert result.returncode==0, 'GPU recovery preflight failed'
print('Remote CPU/CUDA trainer and recovery preflight passed',flush=True)
'''
ast.parse(preflight)
(out/'milano_gpu_tests.py').write_text(preflight,encoding='utf-8')
runtime_check=code[1]
if args.allow_python_build_metadata_change:
    original="if lock.exists(): assert json.loads(lock.read_text())==runtime, 'Recovery runtime changed'"
    replacement="""if lock.exists():
    expected=json.loads(lock.read_text())
    if expected!=runtime:
        # Rebuilt interpreter must retain the exact Python patch and all locked packages.
        assert {k:v for k,v in expected.items() if k!='python'}=={k:v for k,v in runtime.items() if k!='python'}, 'Recovery package or GPU runtime changed'
        assert expected['python'].split()[0]==runtime['python'].split()[0], 'Python patch version changed'
        evidence={'original_runtime':expected,'rebuilt_runtime':runtime,
                  'scope':'Exact locked versions restored; Python build provenance differs; original runtime.json retained. Cross-build bitwise equivalence not asserted.'}
        evidence_path=lock.parent/'runtime_rebuild.json'
        if evidence_path.exists(): assert json.loads(evidence_path.read_text())==evidence, 'Rebuilt runtime changed again'
        else: evidence_path.write_text(json.dumps(evidence,indent=2))
        print('Recorded rebuilt Python provenance; pinned versions match',flush=True)"""
    if original not in runtime_check: raise ValueError('Expected runtime guard missing')
    runtime_check=runtime_check.replace(original,replacement)
worker=bootstrap+'\n'+runtime_check+'''
(ROOT/'tests').mkdir(exist_ok=True)
'''+f"(ROOT/'tests/test_phase2.py').write_text({tests!r})\n"+'''
test=subprocess.run([sys.executable,'-m','pytest','tests/test_phase2.py','-q',
    '--tb=short','-p','no:cacheprovider','--basetemp',str(ROOT/'gpu_recovery_test_tmp')])
assert test.returncode==0, 'GPU recovery tests failed; full training blocked'
(ROOT/'artifacts/gpu_tests.json').write_text(json.dumps({'state':'passed',
    'command':'pytest tests/test_phase2.py; CPU and CUDA model recovery tests'}))
'''+code[2]+'''
import threading, time, traceback
backup_stop=threading.Event()
def periodic_backup():
    while not backup_stop.wait(60):
        try:
            for arch in ('litecrack','litecrack_ba'):
                if (ROOT/'artifacts'/arch/'checkpoints/latest.json').exists():
                    export_recovery(arch)
                    break
        except Exception as error:
            print('PERIODIC BACKUP FAILED:',type(error).__name__,str(error),flush=True)
threading.Thread(target=periodic_backup,daemon=True).start()
try:
'''+''.join('    '+line+'\n' for line in ('\n'.join(code[3:])).splitlines())+'''
finally:
    backup_stop.set()
'''
ast.parse(worker)
if args.resume_package:
    intro=f'''# Resume Milano training in a fresh Kaggle notebook

Enable GPU and Internet. This notebook expects `{args.resume_package.name}`
in `/kaggle/working/` or attached as an intact ZIP through a private input.
If it is not there, send the new Jupyter connection URL so the verified package
can be transferred. Never paste the connection URL into notebook code.

The package restores exact masks, data identity, latest/best weights, optimizer,
scheduler and RNG/sampler state. Source images are downloaded and checksummed.
The original runtime must match; mismatches stop rather than alter recovery state.
Run the preparation cell, then the training cell. LiteCrack resumes first;
LiteCrack-BA follows. Validation selects checkpoints; test results remain separate.
Recovery exports are written every minute once the restored worker starts training.
Download the ZIP and SHA256 exports, or keep the local backup watcher running.
'''
    preparation=f'''from pathlib import Path
import shutil
destination=Path('/kaggle/working')/{args.resume_package.name!r}
if not destination.exists():
    candidates=list(Path('/kaggle/input').rglob({args.resume_package.name!r}))
    assert len(candidates)==1, 'Transfer the verified resume ZIP into this session before running'
    shutil.copyfile(candidates[0],destination)
print('Recovery package available:',destination.name)
'''
    ast.parse(preparation)
    notebook={'nbformat':4,'nbformat_minor':4,
              'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
              'cells':[{'cell_type':'markdown','metadata':{},'source':intro.splitlines(True)},
                       *[{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                          'source':value.splitlines(True)} for value in (preparation,worker)]]}
    resume_name='Road_Crack_'+args.resume_package.stem+'.ipynb'
    (out/resume_name).write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
    print('Resume notebook:',out/resume_name)
encoded=base64.b64encode(worker.encode()).decode()
launch=f'''import base64, json, os, subprocess, sys
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'Enable GPU'
root=Path('/kaggle/working')
pidfile=root/'milano_worker_pid.json'
if pidfile.exists():
    pid=json.loads(pidfile.read_text())['pid']
    try:
        os.kill(pid,0)
        raise RuntimeError('Existing Milano worker is running; inspect instead of launching twice')
    except ProcessLookupError:
        pass
worker=root/'milano_session_worker.py'
content=base64.b64decode({encoded!r})
if worker.exists() and worker.read_bytes()!=content:
    raise RuntimeError('Existing worker code differs; preserve and use a new study directory')
worker.write_bytes(content)
with (root/'milano_session.log').open('ab') as log:
    child=subprocess.Popen([{args.python_executable!r} or sys.executable,'-u',str(worker)],stdout=log,stderr=subprocess.STDOUT,
                           cwd=root,start_new_session=True)
pidfile.write_text(json.dumps({{'pid':child.pid,'log':'milano_session.log'}},indent=2))
print('Launched GPU preflight and training worker:',child.pid)
'''
ast.parse(launch)
(out/'milano_session_launch.py').write_text(launch,encoding='utf-8')
status='''import json, os
from pathlib import Path
root=Path('/kaggle/working')
pidfile=root/'milano_worker_pid.json'
if pidfile.exists():
    pid=json.loads(pidfile.read_text())['pid']
    try: os.kill(pid,0); alive=True
    except ProcessLookupError: alive=False
    print('Worker PID:',pid,'alive:',alive)
for run in (root/'cracklab_milano_v1/artifacts').glob('*'):
    if not run.is_dir(): continue
    for filename in ('environment.json','status.json','failure.json','checkpoints/latest.json','checkpoints/best.json'):
        p=run/filename
        if p.exists(): print(run.name,filename,p.read_text())
log=root/'milano_session.log'
if log.exists():
    with log.open('rb') as f:
        f.seek(max(0,log.stat().st_size-9000))
        print('LOG TAIL:',f.read().decode(errors='replace'))
backups=sorted(root.glob('milano_recovery_*.zip'))
print('Recovery ZIPs:',[p.name for p in backups[-5:]])
'''
ast.parse(status)
(out/'milano_session_status.py').write_text(status,encoding='utf-8')
print('Built and syntax-checked remote launch, worker and status scripts')
