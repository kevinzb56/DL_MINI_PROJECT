"""Build the next GPU notebook from the exact, audited retention workflow source."""
import base64
import hashlib
import json
from pathlib import Path
import zipfile
import zlib
ROOT=Path(__file__).resolve().parents[2]


def build():
    audit=json.loads((ROOT/'output/followup_v1/downloaded_audit/audit.json').read_text())
    archive=ROOT/audit['archive']
    if audit['status']!='verified' or hashlib.sha256(archive.read_bytes()).hexdigest()!=audit['archive_sha256']:
        raise ValueError('Expected the unchanged audited retention download')
    files={}
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if ((name.startswith(('src/','followup/','inspection/','configs/','scripts/')) and name.endswith(('.py','.json')) and name!='scripts/operations/package_followup.py') or name in ('data/manifests/pilot_v1.json','data/manifests/pilot_v1.json.sha256')):
                files[name]=z.read(name).decode('utf-8')
    frozen={n:hashlib.sha256(t.encode()).hexdigest() for n,t in files.items()}
    extra={}
    for name in ('scripts/experiments/multiseed_pipeline.py','scripts/operations/followup_study_inputs.py','scripts/operations/package_followup.py','tests/test_multiseed_pipeline.py','tests/test_phase2.py','tests/test_followup.py'):
        text=(ROOT/name).read_bytes().decode('utf-8');files[name]=text;extra[name]=hashlib.sha256(text.encode()).hexdigest()
    plan={'protocol':'multiseed_delivery_v1','reference_retention_archive_sha256':audit['archive_sha256'],
          'training_seeds':[42,73,101],'source_models':['litecrack','litecrack_ba','stabilized_r2au'],
          'frozen_source':frozen,'orchestration_source':extra,
          'scope':'Nine source training runs, then validation-selected adaptation. Old query remains exploratory; no independent scene/human claims.'}
    files['MULTISEED_PLAN.json']=json.dumps(plan,indent=2)+'\n'
    raw=json.dumps(files).encode();payload=base64.b64encode(zlib.compress(raw,9)).decode()
    cells=[]
    def add(kind,text):
        c={'cell_type':kind,'id':f'multiseed-{len(cells)}','metadata':{},'source':text.splitlines(True)}
        if kind=='code':c.update(execution_count=None,outputs=[])
        cells.append(c)
    add('markdown','''# Three-seed GPU retraining and matched evaluation

This is the next stage after the verified 90-run retention experiment. Open in
VS Code using the existing Kaggle GPU Jupyter kernel. Enable GPU and Internet.
All three seed blocks below are **enabled**: 42, 73, 101, each training LiteCrack,
LiteCrack-BA and stabilized R2AU from scratch. No parent model attachment is needed
for source retraining; the exact audited source/configs are embedded.

New directory: `/kaggle/working/road_crack_multiseed_v1`. Previous experiments stay
unchanged. A separate recovery ZIP and SHA256 are written after every seed block
and evaluation/adaptation stage, including a safely stopped stage. Download them
after each block. GPU time and disk use are substantial; stop between blocks when
needed. Source checkpoints use the frozen 30-step/epoch cadence. On low storage,
restore a recovery ZIP in a fresh session; do not delete original snapshots blindly.

For a resumed session attach the latest **multiseed_*.zip**, or its complete
extracted contents. Restore supports both forms and verifies the full inventory.
Exact resume requires the original compatible runtime. This notebook does not
fabricate independent road scenes or timed human corrections.
''')
    add('code',f"payload={payload!r}\npayload_sha256={hashlib.sha256(raw).hexdigest()!r}\n"+'''import base64,zlib,json,hashlib,sys,os,subprocess,importlib.util
from pathlib import Path
raw=zlib.decompress(base64.b64decode(payload));assert hashlib.sha256(raw).hexdigest()==payload_sha256
files=json.loads(raw);expected_plan=json.loads(files['MULTISEED_PLAN.json'])
project=Path('/kaggle/working/road_crack_multiseed_v1');project.mkdir(exist_ok=True)
had_state=(project/'MULTISEED_PLAN.json').exists() or (project/'source_multiseed').exists()
for name,text in files.items():
    if name=='MULTISEED_PLAN.json':continue
    target=project/name;target.parent.mkdir(parents=True,exist_ok=True)
    content=text.encode('utf-8')
    if target.exists():assert target.read_bytes()==content,'Existing code differs; preserve the old project'
    else:target.write_bytes(content)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
spec=importlib.util.spec_from_file_location('multiseed_bootstrap',project/'scripts/experiments/multiseed_pipeline.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
if not had_state:
    candidates=list(Path('/kaggle/input').rglob('multiseed_*.zip'))
    candidates+=[p.parent for p in Path('/kaggle/input').rglob('MULTISEED_PLAN.json') if (p.parent/'FOLLOWUP_BACKUP_MANIFEST.json').is_file()]
    backup=helper.latest_backup(candidates,expected_plan)
    if backup:print(helper.restore(backup,project,expected_plan))
    else:print('Starting a new three-seed study; no matching recovery input found.')
plan=project/'MULTISEED_PLAN.json'
if plan.exists():assert json.loads(plan.read_text())==expected_plan,'Protocol changed'
else:plan.write_bytes(files['MULTISEED_PLAN.json'].encode())
helper.verify_project(project)
import torch
assert torch.cuda.is_available(),'GPU required: enable Kaggle GPU and reconnect the existing Jupyter kernel'
print('GPU:',torch.cuda.get_device_name(0),'PyTorch:',torch.__version__)
os.chdir(project)
subprocess.run([sys.executable,'scripts/data/restore_datasets.py','--manifest','data/manifests/pilot_v1.json',
    '--destination','/kaggle/working/road_crack_followup_data','--sources','configs/data_sources.json',
    '--report','dataset_restore.json'],check=True)
data_root=Path(json.loads((project/'dataset_restore.json').read_text())['data_root'])
with (project/'gpu-environment.lock.txt').open('w') as stream:
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=stream,check=True)
backup_dir=Path('/kaggle/working')
''')
    add('code','''import xml.etree.ElementTree as ET
from uuid import uuid4
evidence=project/'test_evidence';evidence.mkdir(exist_ok=True)
result=subprocess.run([sys.executable,'-m','pytest','tests/test_followup.py','tests/test_phase2.py','tests/test_multiseed_pipeline.py',
    '-q','-rA','-p','no:cacheprovider','--basetemp='+str(evidence/('tmp-'+uuid4().hex)),
    '--junitxml='+str(evidence/'tests.xml')],capture_output=True,text=True)
(evidence/'tests.log').write_text(result.stdout+result.stderr)
print(result.stdout,result.stderr);assert result.returncode==0,'Fix checks before training'
cases=ET.parse(evidence/'tests.xml').findall('.//testcase')
recovery=[c for c in cases if any(n in c.attrib.get('name','') for n in ('test_actual_trainer_mid_epoch_resume','test_followup_exact_resume'))]
assert len(recovery)==12 and all(c.find('skipped') is None for c in recovery),'CPU and CUDA recovery must pass for source training and adaptation'
''')
    add('code','''def run_stage(stage,seed=None):
    command=[sys.executable,'scripts/experiments/multiseed_pipeline.py','--project',str(project),'--data-root',str(data_root),
        '--stage',stage,'--backup-dir',str(backup_dir)]
    if seed is not None:command+=['--seed',str(seed)]
    import signal
    worker=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        for line in worker.stdout:print(line,end='',flush=True)
        status=worker.wait()
    except KeyboardInterrupt:
        # SIGINT is handled by the orchestrator, which waits for its training worker.
        worker.send_signal(signal.SIGINT);worker.wait(timeout=180);raise
    assert status==0,'Stage stopped or failed; download recovery, then rerun the same cell to resume'
print('Three seeds are enabled. Run the next cells in order; download each printed recovery ZIP.')
''')
    for seed in (42,73,101):add('code',f"# Enabled source-training block: three models, seed {seed}.\nrun_stage('train',{seed})\n")
    add('code',"# Requires all nine source runs; no best-seed selection.\nrun_stage('evaluate')\n")
    add('code',"# Six LiteCrack/BA parents, validation-only selection then locked adaptation.\n# Download recovery and restore in fresh storage if disk headroom is exhausted.\nrun_stage('adapt')\n")
    add('markdown','''The final `multiseed_matrix/summary.json` separates training-seed SD from
support-draw SD. Completed parents with no admissible candidate remain recorded
as failures; no gate is relaxed. Historical query reuse remains exploratory.
New site-separated scene annotations and genuine human timing sessions are the
remaining external inputs. See the local `docs/NEXT_GPU_RUN.md` guide.
''')
    for cell in cells:
        if cell['cell_type']=='code':compile(''.join(cell['source']),cell['id'],'exec')
    path=ROOT/'output/kaggle/Road_Crack_Followup_Multiseed.ipynb'
    path.write_text(json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells},indent=2),encoding='utf-8')
    print(path);print(f'Compiled {len(cells)} cells; embedded {len(files)} files from audited source and new orchestration.')


if __name__=='__main__':build()
