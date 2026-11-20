"""Read checked archive members without extraction or persistent ZIP handles."""
import hashlib
from pathlib import Path, PurePosixPath
import zipfile


def safe_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Dataset path escapes root')
    return path


def check_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe ZIP member')


def read_sample(root, row, field):
    if 'archive' not in row:
        return safe_path(root, row[field]).read_bytes()
    check_member(row[field])
    with zipfile.ZipFile(safe_path(root, row['archive'])) as archive:
        return archive.read(row[field])


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()
