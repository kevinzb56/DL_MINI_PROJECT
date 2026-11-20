"""Build the no-training Phase 3 evaluation notebook."""
import json
import hashlib
import zipfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.project_layout import archive_path

root=Path(__file__).resolve().parents[2]
source=(root/'scripts/evaluation/evaluate_litecrack.py').read_text()
with zipfile.ZipFile(archive_path(root, 'phase3_results.zip')) as backup:
    expected_files={n:hashlib.sha256(backup.read(n)).hexdigest() for n in backup.namelist() if not n.endswith('/')}
cells=[]
def add(kind,text):
    cell={'cell_type':kind,'id':f'phase3-eval-{len(cells)}','metadata':{},'source':text.splitlines(True)}
    if kind=='code': cell.update(execution_count=None,outputs=[])
    cells.append(cell)
add('markdown','''# Phase 3 — Fixed LiteCrack evaluation

No retraining or threshold tuning. Replays recorded validation scores before
evaluating 237 source holdout and 95 target-query images at threshold 0.5.
Adds deterministic prediction previews for visual inspection. Keep the saved
Phase 3 checkpoints unchanged. Use the same Kaggle session, or attach the
verified phase3_results.zip as Kaggle input if the session has reset.
''')
add('code',f'expected_files={expected_files!r}\n'+'''import os,sys,json,hashlib,zipfile,subprocess,shutil
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
project=Path('/kaggle/working/road_crack_phase3_v1')
if not (project/'artifacts/phase3/litecrack/checkpoints/best.json').exists():
    candidates=list(Path('/kaggle/input').rglob('phase3_results.zip'))
    local=Path('/kaggle/working/phase3_results.zip')
    if local.exists(): candidates.insert(0,local)
    project.mkdir(exist_ok=True)
    if candidates:
        archive=candidates[0]
        assert hashlib.sha256(archive.read_bytes()).hexdigest()=='d1d9c4fa7c42efb64e2f30f775fbbf8f5973f22676f591ba1537523d63119470', 'Unexpected backup checksum'
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                path=(project/name).resolve(); assert path.is_relative_to(project.resolve())
                if path.exists(): assert path.read_bytes()==z.read(name), 'Existing file differs: preserve and inspect'
            z.extractall(project)
    else:
        extracted=[p.parents[4] for p in Path('/kaggle/input').rglob('best.json')
                   if p.as_posix().endswith('/artifacts/phase3/litecrack/checkpoints/best.json')]
        valid=[]
        for candidate in extracted:
            if all((candidate/name).is_file() and hashlib.sha256((candidate/name).read_bytes()).hexdigest()==digest
                   for name,digest in expected_files.items()): valid.append(candidate)
        assert valid, 'Attach phase3_results.zip or its complete extracted contents as Kaggle input'
        for name,digest in expected_files.items():
            path=(project/name).resolve(); assert path.is_relative_to(project.resolve())
            if path.exists(): assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, 'Existing file differs; preserve and inspect'
        for name in expected_files:
            path=project/name; path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists(): shutil.copyfile(valid[0]/name,path)
        print('Restored extracted Kaggle input; all backup file hashes verified.')
os.chdir(project)
'''+f'''script=project/'scripts/evaluation/evaluate_litecrack.py'
source={source!r}
if script.exists(): assert script.read_text()==source, 'Existing evaluation code differs; preserve and inspect'
else: script.write_text(source)
'''+'''
restore_report=project/'artifacts/final_eval_dataset_restore.json'
subprocess.run([sys.executable,'scripts/data/restore_datasets.py','--manifest','data/manifests/pilot_v1.json',
    '--sources','configs/data_sources.json','--destination','/kaggle/working/road_crack_phase1/data/raw',
    '--report',str(restore_report)],check=True)
data_root=Path(json.loads(restore_report.read_text())['data_root'])
''')
add('code','''from datetime import datetime,timezone
output=project/'artifacts'/('phase3_final_evaluation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
result=subprocess.run([sys.executable,'scripts/evaluation/evaluate_litecrack.py','--project',str(project),
    '--data-root',str(data_root),'--output',str(output)],text=True,capture_output=True)
execution_log=output.with_suffix('.log')
execution_log.write_text(result.stdout+result.stderr)
print(result.stdout,result.stderr)
assert result.returncode==0, 'Evaluation failed; retain execution log for diagnosis'
''')
add('code','''from IPython.display import display,Image
for split in ['source_holdout','target_query']:
    display(Image(filename=str(output/(split+'_predictions.jpg'))))
''')
add('code','''backup=Path('/kaggle/working/phase3_final_evaluation.zip')
with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(output.rglob('*')):
        if p.is_file(): z.write(p,p.relative_to(output).as_posix())
    z.write(script,'evaluate_litecrack.py')
    z.write(execution_log,'execution.log')
digest=hashlib.sha256(backup.read_bytes()).hexdigest()
backup.with_suffix('.zip.sha256').write_text(digest+'  '+backup.name+'\\n')
print('Download from Kaggle Files:',backup)
print('SHA256:',digest)
''')
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
for c in cells:
    if c['cell_type']=='code': compile(''.join(c['source']),c['id'],'exec')
target=root/'output/kaggle/Road_Crack_Phase3_Final_Evaluation.ipynb'
target.write_text(json.dumps(notebook,indent=2),encoding='utf-8')
print(target)
print('Notebook code-cell syntax checks passed')
