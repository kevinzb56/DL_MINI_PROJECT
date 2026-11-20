"""Reuse verified dual-backup bootstrap, then add gated few-shot execution."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
template=(root/'scripts/authoring/build_boundary_comparison_notebook.py').read_text()
template=template.replace("('src','scripts')","('src','scripts','tests')")
template=template.replace('specs=[]',"files['configs/fewshot.json']=(root/'configs/fewshot.json').read_text()\nspecs=[]")
template=template.replace('road_crack_boundary_comparison_v1','road_crack_phase5_v1')
template=template.replace('Road_Crack_Phase4_Comparison.ipynb','Road_Crack_Phase5_FewShot.ipynb')
namespace={'__file__':str(root/'scripts/authoring/build_boundary_comparison_notebook.py')}
exec(compile(template,'phase5_bootstrap_builder','exec'),namespace)
path=root/'output/kaggle/Road_Crack_Phase5_FewShot.ipynb'
notebook=json.loads(path.read_text())
def cell(kind,text,index):
    value={'cell_type':kind,'id':f'fewshot-{index}','metadata':{},'source':text.splitlines(True)}
    if kind=='code':value.update(execution_count=None,outputs=[])
    return value
notebook['cells'][0]=cell('markdown','''# Phase 5 — Fixed-budget few-shot adaptation

Attach phase3_results.zip and phase4_results.zip if the session reset. Existing
and extracted backups are supported. Enable GPU and Internet for data restoration.
This experiment uses existing labeled acquisition images, not newly timed human
corrections. Frozen supports: five draws at K=1/5/10 for each of two parent models.
Five support passes, decoder-only Adam 0.0001, batch two, no replay/augmentation.
No model/epoch/threshold is selected using query scores. Every optimizer step is
checkpointed; rerun resumes. Raw query metrics appear only after all adaptation
runs finish. Download the result ZIP before ending the session.
''',0)
notebook['cells']=notebook['cells'][:2]
notebook['cells'].append(cell('code','''import xml.etree.ElementTree as ET
test_dir=project/'test_evidence';test_dir.mkdir(exist_ok=True)
test=subprocess.run([sys.executable,'-m','pytest','tests/test_fewshot.py','-q','-rA',
    '--junitxml='+str(test_dir/'tests.xml')],text=True,capture_output=True)
(test_dir/'tests.txt').write_text(test.stdout+test.stderr)
print(test.stdout,test.stderr)
assert test.returncode==0, 'Few-shot tests failed'
cases=ET.parse(test_dir/'tests.xml').findall('.//testcase')
recovery=[c for c in cases if 'test_adaptation_resume' in c.attrib.get('name','')]
assert len(recovery)==4 and all(c.find('skipped') is None for c in recovery), 'CPU and CUDA recovery must pass for both parents'
''',2))
notebook['cells'].append(cell('code','''import signal,shutil
assert shutil.disk_usage('/kaggle/working').free>3*1024**3, 'Need 3 GiB free for immutable adaptation snapshots'
output=project/'results'
command=[sys.executable,'scripts/experiments/run_fewshot.py','--control',str(projects[3]),'--treatment',str(projects[4]),
    '--data-root',str(data_root),'--settings','configs/fewshot.json','--output',str(output)]
with open(project/'execution.log','a',buffering=1) as log:
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    print('Worker PID:',process.pid,flush=True)
    try:
        for line in process.stdout:print(line,end='',flush=True);log.write(line)
        code=process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGTERM);process.wait(timeout=120);raise
assert code==0, 'Worker failed; preserve logs and checkpoints'
if (output/'status.json').exists():print((output/'status.json').read_text())
else:print('Stopped safely. Export recovery state before ending the session.')
''',3))
notebook['cells'].append(cell('code','''backup=Path('/kaggle/working/phase5_results.zip')
selected={p for p in project.rglob('*') if p.is_file() and p.suffix!='.pt' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
for pointer in (project/'results').rglob('checkpoints/latest.json'):
    entry=json.loads(pointer.read_text());snapshot=(pointer.parent/entry['file']).resolve()
    assert snapshot.parent==pointer.parent.resolve()
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==entry['sha256']
    selected.add(snapshot)
with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(selected):z.write(p,p.relative_to(project).as_posix())
digest=hashlib.sha256(backup.read_bytes()).hexdigest()
backup.with_suffix('.zip.sha256').write_text(digest+'  '+backup.name+'\\n')
print('Download through Kaggle Files:',backup)
print('SHA256:',digest)
print('Latest checkpoint for each run included. Older step snapshots remain on Kaggle without deletion.')
''',4))
for c in notebook['cells']:
    if c['cell_type']=='code':compile(''.join(c['source']),c['id'],'exec')
path.write_text(json.dumps(notebook,indent=2),encoding='utf-8')
print('Phase 5 notebook syntax verified')
