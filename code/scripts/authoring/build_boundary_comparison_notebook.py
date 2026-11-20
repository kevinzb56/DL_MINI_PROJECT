import json,hashlib,zipfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.project_layout import archive_path
root=Path(__file__).resolve().parents[2]
files={}
for folder in ('src','scripts'):
    for path in (root/folder).glob('*.py'):
        if not path.name.startswith('build_'): files[path.relative_to(root).as_posix()]=path.read_text()
files['configs/data_sources.json']=(root/'configs/data_sources.json').read_text()
specs=[]
for phase in (3,4):
    archive=archive_path(root,f'phase{phase}_results.zip')
    with zipfile.ZipFile(archive) as z:
        hashes={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if not n.endswith('/')}
    specs.append({'phase':phase,'archive':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'hashes':hashes})
cells=[]
def add(kind,text):
    c={'cell_type':kind,'id':f'boundary-comparison-{len(cells)}','metadata':{},'source':text.splitlines(True)}
    if kind=='code': c.update(execution_count=None,outputs=[])
    cells.append(c)
add('markdown','''# Phase 4 — Paired boundary evaluation

Use the existing Kaggle session. If it reset, attach BOTH phase3_results.zip and
phase4_results.zip as private inputs. ZIP and automatically extracted inputs are
supported and verified. Enable Internet for raw dataset restoration if needed.
No training occurs. Both validation replays must pass before held-out evaluation.
''')
add('code',f'files={files!r}\nspecs={specs!r}\n'+'''import os,sys,json,hashlib,zipfile,shutil,subprocess
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
project=Path('/kaggle/working/road_crack_boundary_comparison_v1')
project.mkdir(exist_ok=True)
for name,text in files.items():
    path=project/name; path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): assert path.read_text()==text, 'Existing comparison code differs; preserve it'
    else: path.write_text(text)
projects={}
for spec in specs:
    phase=spec['phase']; name='litecrack' if phase==3 else 'litecrack_ba'
    destination=Path(f'/kaggle/working/road_crack_phase{phase}_v1')
    marker=f'artifacts/phase{phase}/{name}/checkpoints/best.json'
    if not (destination/marker).exists():
        archives=list(Path('/kaggle/input').rglob(spec['archive']))
        local=Path('/kaggle/working')/spec['archive']
        if local.exists(): archives.insert(0,local)
        destination.mkdir(exist_ok=True)
        if archives:
            assert hashlib.sha256(archives[0].read_bytes()).hexdigest()==spec['sha256'], 'Unexpected backup'
            with zipfile.ZipFile(archives[0]) as z:
                for member in z.namelist():
                    target=(destination/member).resolve(); assert target.is_relative_to(destination.resolve())
                    if target.exists(): assert target.read_bytes()==z.read(member), 'Existing file differs'
                z.extractall(destination)
        else:
            roots=[p.parents[4] for p in Path('/kaggle/input').rglob('best.json') if p.as_posix().endswith('/'+marker)]
            valid=[r for r in roots if all((r/n).is_file() and hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in spec['hashes'].items())]
            assert valid, 'Attach '+spec['archive']+' or its complete extracted contents'
            for n,h in spec['hashes'].items():
                target=destination/n
                if target.exists(): assert hashlib.sha256(target.read_bytes()).hexdigest()==h, 'Existing file differs'
            for n in spec['hashes']:
                target=destination/n; target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists(): shutil.copyfile(valid[0]/n,target)
    projects[phase]=destination
os.chdir(project)
restore_report=project/'dataset_restore.json'
subprocess.run([sys.executable,'scripts/data/restore_datasets.py','--manifest',str(projects[4]/'data/manifests/pilot_v1.json'),
    '--sources','configs/data_sources.json','--destination','/kaggle/working/road_crack_phase1/data/raw','--report',str(restore_report)],check=True)
data_root=Path(json.loads(restore_report.read_text())['data_root'])
''')
add('code','''from datetime import datetime,timezone
output=project/('results_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
result=subprocess.run([sys.executable,'scripts/evaluation/compare_boundaries.py','--control',str(projects[3]),
    '--treatment',str(projects[4]),'--data-root',str(data_root),'--output',str(output)],capture_output=True,text=True)
log=output.with_suffix('.log'); log.write_text(result.stdout+result.stderr)
print(result.stdout,result.stderr)
assert result.returncode==0, 'Comparison failed; preserve the log for diagnosis'
''')
add('code','''backup=Path('/kaggle/working/phase4_comparison_results.zip')
with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(output.rglob('*')):
        if path.is_file(): z.write(path,path.relative_to(output).as_posix())
    for name in ['scripts/evaluation/compare_boundaries.py','src/boundaries.py']:
        z.write(project/name,name)
    z.write(log,'execution.log')
digest=hashlib.sha256(backup.read_bytes()).hexdigest()
backup.with_suffix('.zip.sha256').write_text(digest+'  '+backup.name+'\\n')
print('Download from Kaggle Files:',backup)
print('SHA256:',digest)
''')
for c in cells:
    if c['cell_type']=='code':compile(''.join(c['source']),c['id'],'exec')
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
target=root/'output/kaggle/Road_Crack_Phase4_Comparison.ipynb'
target.write_text(json.dumps(notebook,indent=2),encoding='utf-8')
print(target)
print('Notebook syntax verified')
