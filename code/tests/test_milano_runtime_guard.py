"""Recovery may tolerate build provenance only; numeric versions remain frozen."""
import ast
import json
from pathlib import Path

import pytest


def guard_source():
    tree=ast.parse((Path(__file__).parents[1]/'scripts/operations/build_milano_session_launch.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='replacement' for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError('Missing embedded runtime guard')


def original_runtime():
    return {'python':'3.13.15 (original build)','torch':'2.11.0+cu128',
            'torchvision':'0.26.0+cu128','cuda':'12.8','gpu':'Tesla T4',
            'numpy':'2.1.3','pillow':'13.0.0'}


def execute_guard(lock,runtime):
    exec(compile(guard_source(),'<runtime_guard>','exec'),{'json':json,'lock':lock,'runtime':runtime})


def test_rebuilt_python_preserves_original_lock(tmp_path):
    lock=tmp_path/'runtime.json'
    raw=json.dumps(original_runtime())
    lock.write_text(raw)
    runtime={**original_runtime(),'python':'3.13.15 (rebuilt interpreter)'}
    execute_guard(lock,runtime)
    assert lock.read_text()==raw
    proof=json.loads((tmp_path/'runtime_rebuild.json').read_text())
    assert proof['original_runtime']==original_runtime()
    assert proof['rebuilt_runtime']==runtime
    execute_guard(lock,runtime)
    with pytest.raises(AssertionError,match='changed again'):
        execute_guard(lock,{**runtime,'python':'3.13.15 (another build)'})


@pytest.mark.parametrize(('field','value'),[
    ('python','3.13.16 (new patch)'),('torch','2.11.0+cu130'),
    ('torchvision','0.26.0+cu130'),('cuda','13.0'),
    ('numpy','2.2.0'),('pillow','13.1.0'),('gpu','NVIDIA L4')])
def test_changed_versions_or_gpu_are_rejected(tmp_path,field,value):
    lock=tmp_path/'runtime.json'
    lock.write_text(json.dumps(original_runtime()))
    with pytest.raises(AssertionError): execute_guard(lock,{**original_runtime(),field:value})
    assert not (tmp_path/'runtime_rebuild.json').exists()
