"""Content-based frontend freshness: ZIP extraction timestamps are not build inputs."""
import hashlib
from pathlib import Path


def fingerprint(web):
    web=Path(web)
    files=[web/name for name in ('index.html','package.json','package-lock.json','vite.config.js')]
    for name in ('src','public'):files.extend(path for path in (web/name).rglob('*') if path.is_file())
    return {'schema_version':1,'files':{p.relative_to(web).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(files) if p.is_file()}}
