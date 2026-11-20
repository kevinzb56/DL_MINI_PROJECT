"""Standard-library integrity checks for a trusted CrackLab release bundle."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

MANIFEST='RELEASE_MANIFEST.json'
MAX_TOTAL=2*1024**3
MODEL_ARCHIVES={
    'phase3_results.zip':'d1d9c4fa7c42efb64e2f30f775fbbf8f5973f22676f591ba1537523d63119470',
    'phase4_results.zip':'0851dc03b170e41df2639432a2373579e594a7398f8e0c78183eb5f2a86d652c',
    'phase5_results.zip':'849b8a1c80a406923d4f2cb49b24dd695745bc1cedc18f211e286e3507b8a67a'}


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def safe_member(name):
    parts=PurePosixPath(name).parts
    if not parts or PurePosixPath(name).as_posix()!=name or name.startswith('/') or '\\' in name or ':' in name or any(p in ('.','..') or p.rstrip(' .')!=p for p in parts):
        raise ValueError('Unsafe release path: '+name)
    return name


def verify_archive(path):
    with zipfile.ZipFile(path) as z:
        infos=z.infolist();names=[safe_member(i.filename) for i in infos]
        if len(names)!=len(set(n.casefold() for n in names)):raise ValueError('Duplicate release paths')
        if sum(i.file_size for i in infos)>MAX_TOTAL:raise ValueError('Release exceeds size limit')
        if any(stat.S_ISLNK(i.external_attr>>16) for i in infos):raise ValueError('Symlinks are not allowed')
        if MANIFEST not in names:raise ValueError('Missing release manifest')
        if z.getinfo(MANIFEST).file_size>8*1024*1024:raise ValueError('Oversized release manifest')
        manifest=json.loads(z.read(MANIFEST));entries=manifest['files']
        if set(names)!=(set(entries)|{MANIFEST}):raise ValueError('Archive and inventory disagree')
        for name,entry in entries.items():
            safe_member(name)
            if z.getinfo(name).file_size!=entry['bytes']:raise ValueError('Size mismatch: '+name)
            h=hashlib.sha256()
            with z.open(name) as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            if h.hexdigest()!=entry['sha256']:raise ValueError('Checksum mismatch: '+name)
        for name,expected in MODEL_ARCHIVES.items():
            if name not in entries or entries[name]['sha256']!=expected:raise ValueError('Unrecognized trained-model archive: '+name)
        return {'status':'verified','version':manifest['version'],'files':len(entries),'bytes':sum(i.file_size for i in infos)}


def verify_directory(root):
    root=Path(root).resolve();manifest=json.loads((root/MANIFEST).read_text())
    for name,entry in manifest['files'].items():
        safe_member(name);path=(root/name).resolve()
        if not path.is_relative_to(root) or path.is_symlink():raise ValueError('Unsafe extracted path')
        if not path.is_file() or path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:
            raise ValueError('Missing/changed release file: '+name)
    for name,expected in MODEL_ARCHIVES.items():
        if manifest['files'].get(name,{}).get('sha256')!=expected:raise ValueError('Unrecognized model archive')
    return {'status':'verified','version':manifest['version'],'files':len(manifest['files'])}
