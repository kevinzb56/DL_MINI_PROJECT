"""Rebuild the pinned training stack separately from Kaggle's default runtime."""
import ast
import base64
import json
from pathlib import Path

root=Path(__file__).resolve().parents[2]
out=root/'output/kaggle'
training=(out/'milano_session_launch.py').read_text()
setup='''import json, os, shutil, subprocess, sys
from pathlib import Path
root=Path('/kaggle/working')
status=root/'milano_environment_status.json'
env=dict(os.environ)
env['UV_PYTHON_INSTALL_DIR']=str(root/'milano_python')
env['UV_NO_CACHE']='1'
def phase(name):
    status.write_text(json.dumps({'state':name},indent=2))
    print('ENVIRONMENT:',name,flush=True)
def run(command):
    print('RUN:',command,flush=True)
    subprocess.run(command,env=env,check=True)
try:
    assert shutil.disk_usage(root).free>13*1024**3, 'Insufficient space for isolated GPU stack'
    phase('installing_python_3.13.15')
    run(['uv','python','install','3.13.15'])
    interpreter=subprocess.check_output(['uv','python','find','--managed-python','3.13.15'],env=env,text=True).strip()
    runtime=root/'milano_runtime_cu128'
    python=runtime/'bin/python'
    if not python.exists(): run(['uv','venv','--python',interpreter,str(runtime)])
    phase('installing_pinned_packages')
    run(['uv','pip','install','--python',str(python),'--no-cache',
         '--index-url','https://download.pytorch.org/whl/cu128',
         '--extra-index-url','https://pypi.org/simple','--index-strategy','unsafe-best-match',
         'torch==2.11.0+cu128','torchvision==0.26.0+cu128',
         'numpy==2.1.3','Pillow==13.0.0','pytest>=8,<10','ipython'])
    phase('verifying_isolated_gpu_runtime')
    check="""import sys,json,torch,torchvision,numpy,PIL
assert sys.version.split()[0]=='3.13.15'
assert torch.__version__=='2.11.0+cu128' and torch.version.cuda=='12.8'
assert torchvision.__version__=='0.26.0+cu128'
assert numpy.__version__=='2.1.3' and PIL.__version__=='13.0.0'
assert torch.cuda.is_available() and torch.cuda.device_count()==2
assert torch.cuda.get_device_name()=='Tesla T4'
x=torch.ones((64,64),device='cuda'); assert float((x@x).sum())==262144.0
print(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
 'torchvision':torchvision.__version__,'numpy':numpy.__version__,'pillow':PIL.__version__,
 'gpu':torch.cuda.get_device_name()},indent=2),flush=True)
"""
    run([str(python),'-c',check])
    phase('launching_restored_training_worker')
'''+''.join('    '+line+'\n' for line in training.splitlines())+'''
    phase('training_worker_launched')
except Exception as error:
    status.write_text(json.dumps({'state':'failed','type':type(error).__name__,'error':str(error)},indent=2))
    raise
'''
ast.parse(setup)
encoded=base64.b64encode(setup.encode()).decode()
launch=f'''import base64, json, os, subprocess, sys
from pathlib import Path
root=Path('/kaggle/working')
pidfile=root/'milano_environment_pid.json'
if pidfile.exists():
    pid=json.loads(pidfile.read_text())['pid']
    try:
        os.kill(pid,0)
        raise RuntimeError('Environment worker already exists; inspect before retrying')
    except ProcessLookupError: pass
script=root/'milano_environment_setup.py'
script.write_bytes(base64.b64decode({encoded!r}))
with (root/'milano_environment_setup.log').open('ab') as log:
    worker=subprocess.Popen([sys.executable,'-u',str(script)],stdout=log,stderr=subprocess.STDOUT,
        cwd=root,start_new_session=True)
pidfile.write_text(json.dumps({{'pid':worker.pid}},indent=2))
print('Isolated environment setup launched:',worker.pid,flush=True)
'''
ast.parse(launch)
(out/'milano_environment_launch.py').write_text(launch,encoding='utf-8')
monitor='''import json,time
from pathlib import Path
root=Path('/kaggle/working')
for index in range(90):
    status=root/'milano_environment_status.json'
    state=json.loads(status.read_text()) if status.exists() else {'state':'not_started'}
    print(json.dumps(state),flush=True)
    log=root/'milano_environment_setup.log'
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-2200))
            print(f.read().decode(errors='replace'),flush=True)
    if state['state']=='failed': raise RuntimeError('Isolated environment setup failed; inspect log')
    if state['state']=='training_worker_launched': break
    time.sleep(20)
else: raise RuntimeError('Environment monitoring limit reached')
'''
ast.parse(monitor)
(out/'milano_environment_monitor.py').write_text(monitor,encoding='utf-8')
print('Isolated environment launcher and monitor syntax checked')
