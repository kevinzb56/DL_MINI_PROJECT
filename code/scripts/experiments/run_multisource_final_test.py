"""Frozen official-test evaluation orchestration; never trains or tunes."""
import getpass,hashlib,json,subprocess,sys,time,zipfile,argparse,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'output/multisource_test_v1'
parser=argparse.ArgumentParser()
parser.add_argument('--skip-upload',action='store_true',help='Reuse already uploaded files; their hashes are still checked remotely')
args=parser.parse_args()
if importlib.util.find_spec('websocket') is None:
 raise SystemExit('Missing websocket-client in '+sys.executable+'. Install with: "'+sys.executable+'" -m pip install websocket-client')
base=getpass.getpass('Kaggle connection (hidden): ').strip()
def status(state): (out/'status.json').write_text(json.dumps(dict(state=state)))
def bridge(args):
 result=subprocess.run([sys.executable,'scripts/experiments/kaggle_session.py',*args],cwd=ROOT,input=base+'\n',text=True,capture_output=True)
 print(result.stdout,flush=True)
 with (out/'session_operations.log').open('a',encoding='utf-8') as log:
  log.write(json.dumps(dict(operation=args[0],returncode=result.returncode))+'\n'+result.stdout+'\n')
 if result.returncode:
  (out/'status.json').write_text(json.dumps(dict(state='failed',operation=args[0],detail=result.stdout.strip() or 'Session connection failed')))
  raise RuntimeError('Session operation failed')
status('preparing_test_inputs')
package=out/'Full_Training_v1.zip'
while not (out/'Full_Training_v1.zip.sha256').exists(): time.sleep(10)
# Builder is invoked with explicit output Test_Inputs.zip; wait on its sidecar.

assert hashlib.sha256(package.read_bytes()).hexdigest()==package.with_suffix('.zip.sha256').read_text().split()[0]
with zipfile.ZipFile(package) as archive:
 assert archive.testzip() is None
 manifest=json.loads(archive.read('manifest.json'))
 assert len(manifest['records'])==496
 assert all(r['publisher_split']=='test' and r['split']=='source_holdout' for r in manifest['records'])
 assert json.loads(archive.read('config.json'))['image_size']==512
parent=ROOT/'output/multisource_512_continuation/recovery/multisource_full_recovery_1794909934657617149.zip'
assert hashlib.sha256(parent.read_bytes()).hexdigest()==parent.with_suffix('.zip.sha256').read_text().split()[0]
status('uploading')
if not args.skip_upload:
 for p in (package,parent): bridge(['--upload',str(p)])
checks={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (package,parent)}
code="from pathlib import Path\nimport hashlib,zipfile\nroot=Path('/kaggle/working/cracklab_multisource_final_test')\nassert not root.exists(), 'Inspect existing test results before rerun'\nroot.mkdir()\nchecks="+repr(checks)+"\nfor name,digest in checks.items():\n p=root.parent/name\n assert hashlib.sha256(p.read_bytes()).hexdigest()==digest\n with zipfile.ZipFile(p) as z:\n  assert z.testzip() is None\n  for n in z.namelist():\n   target=(root/n).resolve();assert target.is_relative_to(root.resolve())\n   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(n))\n"
# Recovery contains the training manifest: restore the test manifest after its extraction.
code+="with zipfile.ZipFile(root.parent/'Full_Training_v1.zip') as z:\n (root/'manifest.json').write_bytes(z.read('manifest.json'))\n"
code+=(ROOT/'output/kaggle/multisource_final_test.py').read_text()
remote=out/'remote_evaluate.py';remote.write_text(code)
status('evaluating_frozen_test')
bridge(['--code-file',str(remote),'--capture',str(out/'evaluation.txt')])
bridge(['--download','multisource_final_test.zip','--output',str(out/'results.zip')])
with zipfile.ZipFile(out/'results.zip') as z:
 assert z.testzip() is None
 rows=json.loads(z.read('per_image.json'));assert len(rows)==496 and all(r['split']=='source_holdout' for r in rows)
 for name in ('summary.json','per_image.json'): (out/name).write_bytes(z.read(name))
status('complete_verified_locally')
