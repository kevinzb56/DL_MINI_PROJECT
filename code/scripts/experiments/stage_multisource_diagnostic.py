"""Verify and stage the generated diagnostic bundle and credential-free launcher."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[2]
bundle=ROOT/'output/multisource_diagnostic/Multisource_Diagnostic_v1.zip'
with zipfile.ZipFile(bundle) as archive:
    # Normalize the initial builder's duplicated parent manifest, retaining the
    # first, original manifest bytes whose hash was stored in the checkpoints.
    if len(archive.namelist())!=len(set(archive.namelist())):
        temporary=bundle.with_suffix('.zip.normalized')
        seen=set()
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as target:
            for item in archive.infolist():
                if item.filename in seen:
                    assert item.filename=='parent/manifest.json'
                    continue
                seen.add(item.filename)
                target.writestr(item.filename,archive.read(item))
    else: temporary=None
if temporary: temporary.replace(bundle)
with zipfile.ZipFile(bundle) as archive:
    assert archive.testzip() is None
    index=json.loads(archive.read('cache/index.json'))
    assert hashlib.sha256(archive.read('parent/manifest.json')).hexdigest()==index['parent_manifest_sha256']
    assert hashlib.sha256(archive.read('cache/samples.npz')).hexdigest()==index['samples_sha256']
    for name in ('best','latest'):
        pointer=json.loads(archive.read(f'parent/checkpoints/{name}.json'))
        assert hashlib.sha256(archive.read('parent/checkpoints/'+pointer['file'])).hexdigest()==pointer['sha256']
    assert archive.read('scripts/experiments/run_multisource_diagnostic.py')==(ROOT/'scripts/experiments/run_multisource_diagnostic.py').read_bytes()
digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
bundle.with_suffix('.zip.sha256').write_text(digest+'  '+bundle.name+'\n')
launcher=f'''from pathlib import Path
import hashlib, json, subprocess, sys, zipfile
root=Path('/kaggle/working/cracklab_multisource_diagnostic_v1')
bundle=Path('/kaggle/working/Multisource_Diagnostic_v1.zip')
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=={digest!r}
assert not root.exists(), 'Inspect existing diagnostic before launching another worker'
root.mkdir()
with zipfile.ZipFile(bundle) as archive:
    assert len(archive.namelist())==len(set(archive.namelist()))
    for name in archive.namelist():
        target=(root/name).resolve()
        assert target.is_relative_to(root.resolve())
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(archive.read(name))
with open(root.parent/'multisource_diagnostic.log','w') as log:
    worker=subprocess.Popen([sys.executable,'-u','scripts/experiments/run_multisource_diagnostic.py'],cwd=root,
                            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'worker.json').write_text(json.dumps(dict(pid=worker.pid)))
print('Diagnostic worker started:',worker.pid)
'''
(ROOT/'output/kaggle/multisource_diagnostic_launch.py').write_text(launcher,encoding='utf-8')
print('Verified diagnostic bundle:',bundle.stat().st_size,'bytes',digest)
