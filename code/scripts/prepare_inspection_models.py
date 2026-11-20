"""Import only verified project archives into a separate Phase 6 model registry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspection.store import Store, now
from src.atomic import atomic_write, write_json
from scripts.project_layout import archive_path

EXPECTED = {3: 'd1d9c4fa7c42efb64e2f30f775fbbf8f5973f22676f591ba1537523d63119470',
            4: '0851dc03b170e41df2639432a2373579e594a7398f8e0c78183eb5f2a86d652c',
            5: '849b8a1c80a406923d4f2cb49b24dd695745bc1cedc18f211e286e3507b8a67a'}
MULTISOURCE_SHA256 = '7f2f0bb391f1eea70c33256e0f17aead23c6d8e208a85adf9168a9a1745ff376'


def prepare_multisource(root, store):
    """Register the validation-frozen model without changing any project default."""
    selection_path = root / 'output/multisource_test_v1/FROZEN_SELECTION.json'
    checkpoint = root / 'output/multisource_512_continuation/selected_model/best.pt'
    if not selection_path.exists() or not checkpoint.exists():
        return
    selection = json.loads(selection_path.read_text())
    if (selection['status'] != 'frozen_before_test' or
        selection['checkpoint']['sha256'] != MULTISOURCE_SHA256 or
        selection['architecture'] != 'litecrack' or
        selection['image_size'] != 512 or selection['threshold'] != .5):
        raise ValueError('Multi-source freeze record differs from the selected model')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != MULTISOURCE_SHA256:
        raise ValueError('Multi-source checkpoint hash mismatch')
    destination = store.folder('models') / (MULTISOURCE_SHA256 + '.pt')
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != MULTISOURCE_SHA256:
            raise ValueError('Registered multi-source model differs')
    else:
        atomic_write(destination, lambda stream: stream.write(checkpoint.read_bytes()))
    value = {'id': MULTISOURCE_SHA256[:32], 'sha256': MULTISOURCE_SHA256,
             'label': 'LiteCrack · multi-source 512 px', 'phase': 9, 'project': None,
             'created': now(), 'state': 'reference', 'threshold': .5,
             'validation_dice': selection['validation']['dice'],
             'test_dice': .2873248092622699,
             'description': 'Six-archive research reference; ground camera results are stronger than UAV results.'}
    existing = next((v for v in store.list('version') if v['id'] == value['id']), None)
    if existing and (existing['sha256'] != value['sha256'] or existing.get('phase') != 9):
        raise ValueError('Multi-source registry identity conflict')
    if not existing:
        store.put('version', value)


def prepare(root, workspace):
    store = Store(workspace)
    models = store.folder('models')
    for phase, digest in EXPECTED.items():
        archive = archive_path(root, f'phase{phase}_results.zip')
        if not archive.exists():
            if phase == 5:
                continue
            raise ValueError(f'Missing {archive.name}')
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise ValueError(f'{archive.name} differs from the verified backup; preserve it for review')
        with zipfile.ZipFile(archive) as z:
            if phase in (3, 4):
                model = 'litecrack' if phase == 3 else 'litecrack_ba'
                pointers = [(f'artifacts/phase{phase}/{model}/checkpoints/best.json',
                             'LiteCrack · base' if phase == 3 else 'LiteCrack-BA · base')]
            else:
                pointers = [(n, n.removeprefix('results/').removesuffix('/checkpoints/latest.json').replace('/', ' · '))
                            for n in z.namelist() if n.startswith('results/') and n.endswith('/checkpoints/latest.json')]
            for name, label in pointers:
                pointer = json.loads(z.read(name))
                if Path(pointer['file']).name != pointer['file']:
                    raise ValueError('Invalid checkpoint filename')
                data = z.read(name.rsplit('/', 1)[0] + '/' + pointer['file'])
                sha = hashlib.sha256(data).hexdigest()
                if sha != pointer['sha256']:
                    raise ValueError('Checkpoint hash mismatch')
                path = models / (sha + '.pt')
                if path.exists():
                    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                        raise ValueError('Existing model differs')
                else:
                    atomic_write(path, lambda stream: stream.write(data))
                value = {'id': sha[:32], 'sha256': sha, 'label': label, 'phase': phase,
                         'project': None, 'created': now(), 'state': 'reference',
                         'source_archive_sha256': digest,
                         'description': 'Frozen experiment reference; explicitly select a support draw when comparing.' if phase == 5 else 'Frozen source-trained reference.'}
                if not any(v['id'] == value['id'] for v in store.list('version')):
                    store.put('version', value)
            if phase == 4:
                write_json(store.root / 'frozen_manifest.json', json.loads(z.read('data/manifests/pilot_v1.json')))
    prepare_multisource(root, store)
    print(f'Ready: {len(store.list("version"))} models in {store.root}')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=root / 'artifacts/phase6/workspace')
    args = parser.parse_args()
    prepare(root, args.workspace)
