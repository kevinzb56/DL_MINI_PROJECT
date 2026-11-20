"""Package the next audit step for the already-connected Kaggle kernel."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[2]
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
    for name in ('src/__init__.py','src/atomic.py','scripts/data/audit_kaggle_data.py'):
        archive.write(root/name,name)
payload=buffer.getvalue()
cells=[]
def add(kind,source):
    cell={'cell_type':kind,'id':f'audit-{len(cells):02d}','metadata':{},'source':source.splitlines(True)}
    if kind=='code': cell.update(execution_count=None,outputs=[])
    cells.append(cell)

add('markdown','''# Phase 1 — Real dataset audit

Use the same Kaggle session as the previous notebook. This notebook includes its
own audit code and reads the downloaded repositories without modifying them.
It produces a review bundle and candidate splits. It never silently converts
unknown label values, invents capture metadata, starts training, or freezes splits.
''')
add('code',f'''import base64, hashlib, io, os, sys, zipfile
from pathlib import Path
payload=base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest()=={hashlib.sha256(payload).hexdigest()!r}
project=Path('/kaggle/working/road_crack_data_audit_v2')
project.mkdir(exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name in archive.namelist():
        path=(project/name).resolve()
        assert path.is_relative_to(project.resolve())
        if path.exists(): assert path.read_bytes()==archive.read(name), 'Existing code differs; choose a new project directory'
    archive.extractall(project)
os.chdir(project)
data_root=Path('/kaggle/working/road_crack_phase1/data/raw')
if not data_root.exists():
    raise RuntimeError('Previous data directory missing: restore/clone the datasets first, or update data_root')
print('Using data:',data_root)
''')
add('code','''import subprocess
from datetime import datetime, timezone
# New output directory on every run preserves earlier audit evidence.
run=project/'artifacts'/('audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
result=subprocess.run([sys.executable,'scripts/data/audit_kaggle_data.py',
    '--data-root',str(data_root),'--output',str(run)],text=True,capture_output=True)
print(result.stdout)
print(result.stderr)
run.mkdir(parents=True,exist_ok=True)
(run/'execution.txt').write_text(result.stdout+result.stderr)
assert result.returncode==0, 'Audit failed; export the evidence below for diagnosis'
''')
add('code','''from IPython.display import display, Image
sheet=run/'contact_sheet.jpg'
if sheet.exists():
    display(Image(filename=str(sheet)))
else:
    print('No contact sheet: inspect audit_report.json and inventory.json for pairing errors.')
print('Review whether WHITE mask pixels follow actual cracks. Similarity groups do not verify capture independence.')
''')
add('code','''import zipfile, hashlib
backup=Path('/kaggle/working/phase1_data_audit_v2.zip')
with zipfile.ZipFile(backup,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(run.rglob('*')):
        if path.is_file(): archive.write(path,path.relative_to(run).as_posix())
digest=hashlib.sha256(backup.read_bytes()).hexdigest()
backup.with_suffix('.zip.sha256').write_text(digest+'  '+backup.name+'\\n')
print('Download from Kaggle Files/Output panel:',backup)
print('SHA256:',digest)
print('Copy the ZIP into your local DL_MINI_PROJECT folder for inspection. Raw datasets are excluded.')
''')
add('markdown','''## What remains after this runs

Inspect every audit error, contact-sheet polarity and any suspicious similarity
pairs. If capture/sequence groups are unavailable, report the sample-level pilot
limitation explicitly rather than claiming unseen-road independence. Only after
that review should a new manifest be frozen. Keep query images outside support
selection and query labels outside training. Data counts above are measured at
runtime; this notebook contains no assumed counts or fabricated audit results.
''')
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'}},'cells':cells}
out=root/'output/kaggle/Road_Crack_Phase1_Data_Audit.ipynb'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(notebook,indent=2),encoding='utf-8')
for cell in cells:
    if cell['cell_type']=='code': compile(''.join(cell['source']),cell['id'],'exec')
print(out)
print('All notebook code cells compile')
