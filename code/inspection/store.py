"""Small durable SQLite catalog; binary artifacts stay in a private workspace."""
import json
from pathlib import Path
import re
import sqlite3
import uuid
import os
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def identity():
    return uuid.uuid4().hex


def checked_id(value):
    if not re.fullmatch(r'[a-f0-9]{16,64}', value):
        raise ValueError('Invalid record ID')
    return value


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / 'catalog.sqlite3'
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, project TEXT, payload TEXT, PRIMARY KEY(kind,id))')
            db.execute('CREATE INDEX IF NOT EXISTS project_records ON records(kind,project)')

    def connect(self):
        return sqlite3.connect(self.database, timeout=30)

    def put(self, kind, value, project=''):
        checked_id(value['id'])
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?,?)',
                       (kind, value['id'], project, json.dumps(value, allow_nan=False)))
        return value

    def get(self, kind, record_id, project=None):
        checked_id(record_id)
        with self.connect() as db:
            row = db.execute('SELECT project,payload FROM records WHERE kind=? AND id=?', (kind, record_id)).fetchone()
        if row is None or (project is not None and row[0] != project):
            raise KeyError('Record not found')
        return json.loads(row[1])

    def list(self, kind, project=None):
        with self.connect() as db:
            if project is None:
                rows = db.execute('SELECT payload FROM records WHERE kind=? ORDER BY rowid', (kind,)).fetchall()
            else:
                rows = db.execute('SELECT payload FROM records WHERE kind=? AND project=? ORDER BY rowid', (kind, project)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def folder(self, *parts):
        target = self.root.joinpath(*parts).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError('Path escapes workspace')
        target.mkdir(parents=True, exist_ok=True)
        return target

    def version(self, version_id, project_id):
        version = self.get('version', version_id)
        if version.get('project') not in (None, project_id):
            raise KeyError('Model belongs to another project')
        return version
