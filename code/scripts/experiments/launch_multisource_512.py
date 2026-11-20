"""Credential stays in process memory while package preparation finishes."""
import getpass, hashlib, json, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
base=getpass.getpass('Kaggle connection (hidden): ').strip()
out=ROOT/'output/multisource_512_pilot'
package=out/'Training_512_Pilot.zip'
def bridge(args):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/experiments/kaggle_session.py'),*args],
                          input=base+'\n',text=True,capture_output=True,cwd=ROOT)
    print(result.stdout,flush=True)
    if result.returncode: raise RuntimeError('Kaggle operation failed; inspect status before retry')
def status(state):
    (out/'launch_status.json').write_text(json.dumps(dict(state=state),indent=2))
status('waiting_for_package')
while not package.with_suffix('.zip.sha256').exists(): time.sleep(10)
digest=hashlib.sha256(package.read_bytes()).hexdigest()
assert digest==package.with_suffix('.zip.sha256').read_text().split()[0]
status('uploading')
bridge(['--upload',str(package)])
code=f'''from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
root=Path('/kaggle/working/cracklab_multisource_512_pilot')
p=root.parent/'Training_512_Pilot.zip'
assert hashlib.sha256(p.read_bytes()).hexdigest()=={digest!r}
assert not root.exists(), 'Inspect existing worker before relaunch'
root.mkdir()
with zipfile.ZipFile(p) as z:
    assert z.testzip() is None
    for name in z.namelist():
        target=(root/name).resolve(); assert target.is_relative_to(root.resolve())
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(z.read(name))
with open(root.parent/'multisource_512_pilot.log','w') as log:
    worker=subprocess.Popen([sys.executable,'-u','scripts/experiments/run_multisource_full.py'],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'worker.json').write_text(json.dumps(dict(pid=worker.pid)))
print('Full training worker started:',worker.pid)
'''
launcher=out/'remote_launch.py'; launcher.write_text(code)
bridge(['--code-file',str(launcher),'--capture',str(out/'launch.txt')])
status('worker_launched_backup_watch_active')
bridge(['--watch-recovery','--recovery-prefix','multisource_full_recovery_',
        '--completion-path','cracklab_multisource_512_pilot/artifacts/run/status.json',
        '--completion-state','training_complete','--poll-seconds','30',
        '--output',str(out/'recovery')])
status('training_complete_final_backup_verified')
