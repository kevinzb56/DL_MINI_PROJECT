"""Create the small no-training follow-up notebook for Phase 2."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[2]
script=(root/'scripts/evaluation/evaluate_frozen_baselines.py').read_text()
cells=[]
def add(kind,source):
    cell={'cell_type':kind,'id':f'final-eval-{len(cells)}','metadata':{},'source':source.splitlines(True)}
    if kind=='code': cell.update(execution_count=None,outputs=[])
    cells.append(cell)
add('markdown','''# Phase 2 — Final fixed-checkpoint evaluation

Use the same Kaggle session that completed stabilization. This notebook performs
no training. It rechecks validation against recorded scores, then evaluates source
holdout and target query at threshold 0.5. Existing results/checkpoints are retained.
If the session reset, attach the saved phase2_stabilization_results.zip as Kaggle
input; the next cell can restore it. Enable Internet if raw data needs restoration.
''')
add('code','''import os,sys,json,zipfile,subprocess,hashlib
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
project=Path('/kaggle/working/road_crack_phase2_v2')
if not (project/'artifacts/phase2/unet/checkpoints/best.json').exists():
    candidates=list(Path('/kaggle/input').rglob('phase2_stabilization_results.zip'))
    local=Path('/kaggle/working/phase2_stabilization_results.zip')
    if local.exists(): candidates.insert(0,local)
    assert candidates, 'Attach your phase2_stabilization_results.zip backup as Kaggle input first'
    archive=candidates[0]
    assert hashlib.sha256(archive.read_bytes()).hexdigest()=='578ea01fb8389abd0904f63ed621192bc6708b6d76cac8ca9c2840e01d1a5040', 'Unexpected backup: verify before restoring'
    project.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            path=(project/name).resolve()
            assert path.is_relative_to(project.resolve())
            if path.exists(): assert path.read_bytes()==z.read(name), 'Existing file differs; preserve it'
        z.extractall(project)
os.chdir(project)
'''+f"(project/'scripts/evaluation/evaluate_frozen_baselines.py').write_text({script!r})\n"+'''
restore_report=project/'artifacts/final_eval_dataset_restore.json'
subprocess.run([sys.executable,'scripts/data/restore_datasets.py','--manifest','data/manifests/pilot_v1.json',
    '--sources','configs/data_sources.json','--destination','/kaggle/working/road_crack_phase1/data/raw',
    '--report',str(restore_report)],check=True)
data_root=Path(json.loads(restore_report.read_text())['data_root'])
''')
add('code','''from datetime import datetime,timezone
output=project/'artifacts'/('final_evaluation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
subprocess.run([sys.executable,'scripts/evaluation/evaluate_frozen_baselines.py','--project',str(project),
    '--data-root',str(data_root),'--output',str(output)],check=True)
print((output/'evaluation.json').read_text())
''')
add('code','''backup=Path('/kaggle/working/phase2_final_evaluation.zip')
with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(output.rglob('*')):
        if p.is_file(): z.write(p,p.relative_to(output).as_posix())
    z.write(project/'scripts/evaluation/evaluate_frozen_baselines.py','evaluate_frozen_baselines.py')
digest=hashlib.sha256(backup.read_bytes()).hexdigest()
backup.with_suffix('.zip.sha256').write_text(digest+'  '+backup.name+'\\n')
print('Download through Kaggle Files:',backup)
print('SHA256:',digest)
''')
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
for c in cells:
    if c['cell_type']=='code': compile(''.join(c['source']),c['id'],'exec')
target=root/'output/kaggle/Road_Crack_Phase2_Final_Evaluation.ipynb'
target.write_text(json.dumps(notebook,indent=2),encoding='utf-8')
print(target)
print('Notebook code-cell syntax checks passed')
