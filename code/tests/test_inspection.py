"""Phase 6 API isolation, exact mask round trip, and actual worker recovery."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch
import httpx
from fastapi.testclient import TestClient
from inspection.api import create_app
from inspection.engine import decode_image, png_bytes, probability
from inspection.store import now
from inspection.vision_agent import suggest_cracks
from scripts.inspection_worker import run
from src.model_factory import build_model
from src.atomic import write_json


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(42)
    app = create_app(tmp_path, start_workers=False)
    store = app.state.store
    config = {'architecture': 'litecrack_ba', 'image_size': 32, 'encoder_width_mult': .5, 'boundary_weight': .2}
    model = build_model(config)
    # Stable all-foreground fixture gives a known passing activation case.
    # Real-checkpoint, nonuniform predictions are exercised by browser QA.
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(5.)
    data = io.BytesIO(); torch.save({'config': config, 'model': model.state_dict()}, data)
    sha = hashlib.sha256(data.getvalue()).hexdigest()
    (store.folder('models') / (sha + '.pt')).write_bytes(data.getvalue())
    version = {'id': sha[:32], 'sha256': sha, 'label': 'Fixture parent', 'state': 'reference', 'phase': 4,
               'project': None, 'created': now()}
    store.put('version', version)
    client = TestClient(app)
    project = client.post('/api/projects', json={'name': 'Fixture road'}).json()
    base = f'/api/projects/{project["id"]}'
    return client, store, project, base, version, model


def upload(client, base, seed=1):
    rgb = np.random.default_rng(seed).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    return client.post(base + '/images', files={'file': ('road.png', png_bytes(Image.fromarray(rgb)), 'image/png')}).json()


def correction(client, base, iid, complete=True, array=None):
    if array is None:
        array = np.zeros((32, 48), dtype=np.uint8); array[:, 10:13] = 255
    return client.post(base + f'/images/{iid}/corrections?complete={str(complete).lower()}',
                       files={'file': ('mask.png', png_bytes(Image.fromarray(array)), 'image/png')})


def test_vision_suggestions_use_supported_default(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'test-key')
    monkeypatch.delenv('CRACKLAB_VISION_MODEL', raising=False)
    requested = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {'choices': [{'message': {'content': '{"paths":[],"note":"No missed cracks"}'}}]}

    def post(url, json, headers, timeout):
        requested.update(json)
        return Response()

    monkeypatch.setattr('inspection.vision_agent.httpx.post', post)
    result = suggest_cracks(Image.new('RGB', (32, 48)), Image.new('RGB', (32, 48), 'black'))
    assert requested['model'] == 'qwen/qwen3.8-27b'
    assert result['model'] == requested['model']


def test_vision_suggestions_retry_without_json_mode(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'test-key')
    monkeypatch.delenv('CRACKLAB_VISION_MODEL', raising=False)
    requests = []

    class Rejected:
        status_code = 400
        text = '{"error":{"message":"json mode unsupported"}}'

        def json(self):
            return {'error': {'message': 'json mode unsupported'}}

        def raise_for_status(self):
            request = httpx.Request('POST', 'https://api.groq.com/openai/v1/chat/completions')
            raise httpx.HTTPStatusError('400', request=request, response=httpx.Response(400, json=self.json(), request=request))

    class Accepted:
        def raise_for_status(self):
            pass

        def json(self):
            return {'choices': [{'message': {'content': '{"paths":[],"note":"No missed cracks"}'}}]}

    def post(url, json, headers, timeout):
        requests.append(json)
        return Rejected() if len(requests) == 1 else Accepted()

    monkeypatch.setattr('inspection.vision_agent.httpx.post', post)
    result = suggest_cracks(Image.new('RGB', (32, 48)), Image.new('RGB', (32, 48), 'black'))
    assert 'response_format' in requests[0]
    assert 'response_format' not in requests[1]
    assert result['paths'] == []


def test_health_reports_ai_configuration(setup, monkeypatch):
    client, _, _, _, _, _ = setup
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    health = client.get('/api/health').json()
    assert health['ai_ready'] is False
    assert 'Groq API key' in health['ai_message']

    monkeypatch.setenv('GROQ_API_KEY', 'test-key')
    health = client.get('/api/health').json()
    assert health['ai_ready'] is True


def test_image_validation_roundtrip_and_isolation(setup):
    client, store, project, base, version, _ = setup
    assert client.post(base + '/images', files={'file': ('bad.jpg', b'invalid')}).status_code == 400
    assert client.post(base + '/images', files={'file': ('bad.gif', png_bytes(Image.new('RGB', (4, 4))))}).status_code == 400
    image = upload(client, base)
    assert upload(client, base)['id'] == image['id']
    expected = np.zeros((32, 48, 4), dtype=np.uint8); expected[..., 3] = 255; expected[3:17, 8:13, :3] = 255
    saved = correction(client, base, image['id'], array=expected)
    assert saved.status_code == 201
    result = client.get(base + f'/corrections/{saved.json()["id"]}/mask')
    actual = np.asarray(Image.open(io.BytesIO(result.content)))
    assert np.array_equal(actual, expected[..., 0])
    assert correction(client, base, image['id'], array=np.zeros((16, 16), dtype=np.uint8)).status_code == 400
    other = client.post('/api/projects', json={'name': 'Other'}).json()
    assert client.get(f'/api/projects/{other["id"]}/images/{image["id"]}/original').status_code == 404
    assert client.post(base + '/images', files={'file': ('cross.png', b'invalid')}, headers={'Origin': 'https://external.example'}).status_code == 403
    assert client.get('/api/health', headers={'Host': 'external.example'}).status_code == 400
    assert client.get(base + '/images/' + 'a'*32 + '/original').status_code == 404


def test_inference_export_and_activation_gate(setup):
    client, store, project, base, version, _ = setup
    image = upload(client, base)
    first = client.post(base + '/predict', json={'image_id': image['id'], 'version_id': version['id']})
    assert first.status_code == 201, first.text
    prediction = first.json(); assert prediction['metrics'] is None
    assert prediction['width'] == 48 and prediction['height'] == 32
    predicted = client.get(base + f'/predictions/{prediction["id"]}/mask').content
    saved = client.post(base + f'/images/{image["id"]}/corrections?complete=true', files={'file': ('mask.png', predicted)})
    assert saved.status_code == 201
    second = client.post(base + '/predict', json={'image_id': image['id'], 'version_id': version['id']}).json()
    assert second['metrics'] == {'dice': 1., 'iou': 1.}
    exported = client.get(base + '/export')
    with zipfile.ZipFile(io.BytesIO(exported.content)) as z:
        metadata = json.loads(z.read('metadata.json'))
        assert len(metadata['predictions']) == 2
        assert metadata['predictions'][-1]['checkpoint_sha256'] == version['sha256']
        assert Image.open(io.BytesIO(z.read('predictions/' + prediction['id'] + '/mask.png'))).size == (48, 32)
        assert 'measurements.csv' in z.namelist()
    rejected = dict(version, id='f'*32, project=project['id'], state='rejected')
    store.put('version', rejected)
    assert client.post(base + '/activate', json={'version_id': rejected['id']}).status_code == 400
    assert client.post(base + '/activate', json={'version_id': version['id']}).status_code == 200


def test_incomplete_overlap_and_frozen_data_rejected(setup):
    client, store, project, base, version, _ = setup
    a, b = upload(client, base, 2), upload(client, base, 3)
    correction(client, base, a['id'], complete=False); correction(client, base, b['id'])
    body = {'parent_id': version['id'], 'support_ids': [a['id']], 'validation_ids': [b['id']]}
    assert client.post(base + '/jobs', json=body).status_code == 400
    correction(client, base, a['id'])
    assert client.post(base + '/jobs', json=dict(body, validation_ids=[a['id']])).status_code == 400
    protected = dict(a, training_eligible=False, frozen_splits=['target_query'])
    store.put('image', protected, project['id'])
    assert client.post(base + '/jobs', json=body).status_code == 400
    write_json(store.root / 'frozen_manifest.json', {'records': [{'split': 'target_query', 'image_sha256': 'unused', 'pixel_sha256': b['pixel_sha256']}]})
    other = client.post('/api/projects', json={'name': 'Protected upload'}).json()
    c = upload(client, f'/api/projects/{other["id"]}', 3)
    assert c['training_eligible'] is False


def test_worker_exact_resume_and_immutable_parent(setup):
    client, store, project, base, version, baseline = setup
    a, b = upload(client, base, 4), upload(client, base, 5)
    for image in (a, b):
        with Image.open(store.root / 'images' / image['id'] / 'original.png') as rgb:
            binary = (probability(baseline, rgb.convert('RGB'), 32) >= .5).astype(np.uint8)*255
        assert correction(client, base, image['id'], array=binary).status_code == 201
    body = {'parent_id': version['id'], 'support_ids': [a['id']], 'validation_ids': [b['id']], 'epochs': 2}
    full = client.post(base + '/jobs', json=body).json()
    run(store.root, full['id'])
    full_job = store.get('job', full['id'])
    assert full_job['state'] == 'completed'
    resumed = client.post(base + '/jobs', json=body).json()
    run(store.root, resumed['id'], stop_after_steps=1)
    assert store.get('job', resumed['id'])['state'] == 'cancelled'
    pointer = json.loads((store.root / 'jobs' / resumed['id'] / 'checkpoints/latest.json').read_text())
    assert pointer['global_step'] == 1
    run(store.root, resumed['id'])
    resumed_job = store.get('job', resumed['id'])
    assert resumed_job['state'] == 'completed'
    def final_state(job):
        folder = store.root / 'jobs' / job['id'] / 'checkpoints'
        return torch.load(folder / json.loads((folder / 'latest.json').read_text())['file'], weights_only=False, map_location='cpu')
    f, r = final_state(full), final_state(resumed)
    assert f['sampler_state']['history'] == r['sampler_state']['history']
    for key in f['model']:
        assert torch.equal(f['model'][key], r['model'][key])
        if key.startswith('encoder.'):
            assert torch.equal(f['model'][key], baseline.state_dict()[key])
    path = store.root / 'models' / (version['sha256'] + '.pt')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == version['sha256']
    # A checkpoint never becomes the active default merely because its job finished.
    assert store.get('project', project['id'])['active_version'] == version['id']
    candidate = store.get('version', resumed_job['candidate_id'])
    assert resumed_job['validation']['passed'] is True
    activation = client.post(base + '/activate', json={'version_id': candidate['id']})
    assert activation.status_code == 200
    assert client.post(base + '/activate', json={'version_id': version['id']}).status_code == 200
    assert store.get('project', project['id'])['active_version'] == version['id']
    assert len(store.get('project', project['id'])['activation_history']) == 2
    recovery = client.get(base + f'/jobs/{resumed["id"]}/export')
    with zipfile.ZipFile(io.BytesIO(recovery.content)) as z:
        pointer = json.loads(z.read('checkpoints/latest.json'))
        assert hashlib.sha256(z.read('checkpoints/' + pointer['file'])).hexdigest() == pointer['sha256']
        assert 'manifest.json' in z.namelist()


def test_cancel_checkpoint_and_dead_worker_recovery(setup):
    client, store, project, base, version, _ = setup
    a, b = upload(client, base, 6), upload(client, base, 7)
    correction(client, base, a['id']); correction(client, base, b['id'])
    body = {'parent_id': version['id'], 'support_ids': [a['id']], 'validation_ids': [b['id']], 'epochs': 1}
    job = client.post(base + '/jobs', json=body).json()
    assert client.post(base + f'/jobs/{job["id"]}/cancel').status_code == 200
    run(store.root, job['id'])
    assert store.get('job', job['id'])['state'] == 'cancelled'
    assert (store.root / 'jobs' / job['id'] / 'checkpoints/latest.json').exists()
    job = store.get('job', job['id'])
    job.update(state='running', pid=99999999, updated='2000-01-01T00:00:00+00:00')
    store.put('job', job, project['id'])
    recovered = client.get(base + '/jobs').json()[0]
    assert recovered['state'] == 'failed' and 'interrupted' in recovered['error']
    assert client.post(base + f'/jobs/{job["id"]}/resume').status_code == 200
    assert store.get('job', job['id'])['state'] == 'queued'

