"""Package exact prepared masks and verified training recovery for a fresh session."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def build(output, backup=None):
    root=Path(__file__).resolve().parents[2]
    prepared=root/'data/external/milano_prepared_v1'
    backup=Path(backup) if backup else max((root/'output/milano_v1/recovery').glob('*.zip'),key=lambda p:p.stat().st_mtime)
    digest=hashlib.sha256(backup.read_bytes()).hexdigest()
    if digest!=backup.with_suffix('.zip.sha256').read_text().split()[0]:
        raise ValueError('Backup archive checksum mismatch')
    output=Path(output)
    if output.exists(): raise ValueError('Preserve existing package; choose a new name')
    with zipfile.ZipFile(backup) as source:
        if source.testzip() is not None: raise ValueError('Backup ZIP corrupt')
        manifest=(prepared/'manifest.json').read_bytes()
        resume_steps={}
        for name in source.namelist():
            if name.endswith(('/latest.json','/best.json')):
                pointer=json.loads(source.read(name))
                parent=name.rsplit('/',1)[0]
                if Path(pointer['file']).name!=pointer['file']: raise ValueError('Unsafe checkpoint pointer')
                if hashlib.sha256(source.read(parent+'/'+pointer['file'])).hexdigest()!=pointer['sha256']:
                    raise ValueError('Backup checkpoint checksum mismatch')
                if name.endswith('/latest.json'):
                    resume_steps[name.split('/')[1]]=pointer['global_step']
            if name.endswith('/manifest.json') and json.loads(source.read(name))!=json.loads(manifest):
                raise ValueError('Prepared dataset differs from training manifest')
        output.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as target:
            for p in sorted(prepared.rglob('*')):
                relative=p.relative_to(prepared)
                if p.is_file() and 'images' not in relative.parts:
                    target.write(p,'data/'+relative.as_posix())
            for name in source.namelist():
                if name.startswith('/') or '..' in Path(name).parts: raise ValueError('Unsafe backup path')
                target.writestr(name,source.read(name))
            target.writestr('restoration_identity.json',json.dumps({
                'backup':backup.name,'backup_sha256':digest,'resume_steps':resume_steps,
                'manifest_sha256':hashlib.sha256(manifest).hexdigest()},indent=2))
    output.with_suffix('.zip.sha256').write_text(hashlib.sha256(output.read_bytes()).hexdigest()+'  '+output.name+'\n')
    print(json.dumps({'package':str(output),'bytes':output.stat().st_size,'resume_steps':resume_steps},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backup',type=Path)
    a=p.parse_args(); build(a.output,a.backup)
