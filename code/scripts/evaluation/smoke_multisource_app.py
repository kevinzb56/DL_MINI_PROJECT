"""Check the frozen multi-source checkpoint through CrackLab's inference path."""
from pathlib import Path
import io
import json
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from inspection.api import create_app
from inspection.engine import infer
from inspection.store import Store
from scripts.prepare_inspection_models import prepare_multisource, MULTISOURCE_SHA256


def main():
    root = Path(__file__).resolve().parents[2]
    store = Store(root / 'artifacts/phase6/workspace')
    prepare_multisource(root, store)
    version = store.get('version', MULTISOURCE_SHA256[:32])
    image = Image.fromarray(np.full((512, 512, 3), 128, dtype=np.uint8))
    values, config, elapsed = infer(store, version, image)
    assert config['architecture'] == 'litecrack' and config['image_size'] == 512
    assert values.shape == (512, 512) and np.isfinite(values).all()
    assert 0 <= values.min() <= values.max() <= 1
    with tempfile.TemporaryDirectory() as folder:
        app_store = Store(folder)
        prepare_multisource(root, app_store)
        with TestClient(create_app(folder, start_workers=False)) as client:
            project = client.post('/api/projects', json={'name': 'Multi-source smoke'}).json()
            encoded = io.BytesIO()
            image.save(encoded, format='PNG')
            uploaded = client.post(f"/api/projects/{project['id']}/images",
                                   files={'file': ('smoke.png', encoded.getvalue(), 'image/png')})
            assert uploaded.status_code == 201, uploaded.text
            response = client.post(f"/api/projects/{project['id']}/predict",
                                   json={'image_id': uploaded.json()['id'],
                                         'version_id': version['id'], 'threshold': .5})
            assert response.status_code == 201, response.text
            prediction = response.json()
            assert prediction['checkpoint_sha256'] == MULTISOURCE_SHA256
            assert prediction['preprocessing']['size'] == 512
            assert prediction['inference']['mode'] == 'resize'
            assert client.get(f"/api/projects/{project['id']}/predictions/{prediction['id']}/mask").status_code == 200
    print(json.dumps({'checkpoint_sha256': version['sha256'],
                      'image_size': config['image_size'], 'output_shape': values.shape,
                      'probability_range': [float(values.min()), float(values.max())],
                      'latency_ms': elapsed, 'api_prediction': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
