"""Durable replacement: failed writes never replace an existing result."""
import json
import os
from pathlib import Path
import tempfile


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_write(path, lambda stream: stream.write(
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")))
