import hashlib
import json
from pathlib import Path
import zipfile
import pytest
from scripts.release_integrity import MODEL_ARCHIVES,MANIFEST,safe_member,verify_archive,verify_directory


@pytest.mark.parametrize('name',['../escape','/root','a\\b','C:escape','dir/../a','name.','name '])
def test_unsafe_release_paths_rejected(name):
    with pytest.raises(ValueError):safe_member(name)


def test_archive_rejects_uninventoried_file(tmp_path):
    path=tmp_path/'extra.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr(MANIFEST,json.dumps({'version':'fixture','files':{}}));z.writestr('extra.py','unexpected')
    with pytest.raises(ValueError,match='inventory'):verify_archive(path)


def test_duplicate_windows_paths_rejected(tmp_path):
    path=tmp_path/'duplicate.zip'
    with zipfile.ZipFile(path,'w') as z:z.writestr('A.txt','a');z.writestr('a.txt','b')
    with pytest.raises(ValueError,match='Duplicate'):verify_archive(path)


def test_extracted_tamper_rejected(tmp_path):
    content=b'fixture'
    (tmp_path/'file.txt').write_bytes(content)
    manifest={'version':'fixture','files':{'file.txt':{'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()},
                                         **{name:{'sha256':sha} for name,sha in MODEL_ARCHIVES.items()}}}
    # A real release supplies all model archives; this focused case must fail on changed content first.
    (tmp_path/MANIFEST).write_text(json.dumps(manifest))
    (tmp_path/'file.txt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):verify_directory(tmp_path)


def test_frontend_freshness_uses_contents_not_extraction_times(tmp_path):
    import os
    from scripts.frontend_build import fingerprint
    (tmp_path/'src').mkdir();(tmp_path/'public').mkdir()
    source=tmp_path/'src/main.jsx';source.write_text('fixture')
    before=fingerprint(tmp_path)
    os.utime(source,(1000000000,1000000000))
    assert fingerprint(tmp_path)==before
    source.write_text('changed')
    assert fingerprint(tmp_path)!=before
