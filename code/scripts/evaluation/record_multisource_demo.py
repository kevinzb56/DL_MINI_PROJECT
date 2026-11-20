"""Record an isolated no-training CrackLab walkthrough with the frozen model."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
URL = 'http://127.0.0.1:8771'
OUT = ROOT / 'output/multisource_demo'
ASSETS = ROOT / 'release_assets/demo'
NEW_SHA = '7f2f0bb391f1eea70c33256e0f17aead23c6d8e208a85adf9168a9a1745ff376'


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ASSETS / 'manifest.json').read_text())
    for row in manifest['records']:
        for kind, field in (('image', 'image_sha256'), ('mask', 'mask_sha256')):
            if hashlib.sha256((ASSETS / row[kind]).read_bytes()).hexdigest() != row[field]:
                raise ValueError('Demo asset checksum mismatch')
    errors, screenshots = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        context = browser.new_context(viewport={'width': 1440, 'height': 1000},
                                      record_video_dir=str(OUT / 'raw_video'))
        page = context.new_page()
        video = page.video
        page.on('pageerror', lambda error: errors.append(str(error)))

        def settle():
            page.wait_for_function("document.querySelector('#busy').hidden", timeout=60000)

        def shot(name):
            page.screenshot(path=str(OUT / name), full_page=True, animations='disabled')
            screenshots.append(name)

        page.goto(URL + '/#studio')
        page.wait_for_function("document.querySelector('#connection').textContent.includes('models ready') && document.querySelector('#busy').hidden")
        pid = page.locator('#projectSelect').input_value()
        base = URL + f'/api/projects/{pid}'
        versions = httpx.get(base + '/versions').json()
        frozen = next(v for v in versions if v['sha256'] == NEW_SHA)
        legacy = next(v for v in versions if v['phase'] == 4)
        page.locator('#upload').set_input_files([str(ASSETS / row['image']) for row in manifest['records']])
        settle()
        page.wait_for_function("document.querySelectorAll('.image-item').length===2")
        shot('01-uploaded.png')

        page.locator('.image-item').first.click()
        settle()
        page.locator('#versionSelect').select_option(frozen['id'])
        assert page.locator('#threshold').input_value() == '50'
        assert page.locator('#inferenceMode').input_value() == 'resize'
        page.locator('#predict').click()
        settle()
        shot('02-frozen-model-prediction.png')
        first = httpx.get(base + '/images').json()[0]

        page.locator('#usePrediction').click()
        settle()
        rect = page.locator('#editorCanvas').bounding_box()
        page.mouse.move(rect['x'] + rect['width'] * .35, rect['y'] + rect['height'] * .35)
        page.mouse.down()
        page.mouse.move(rect['x'] + rect['width'] * .55, rect['y'] + rect['height'] * .55, steps=8)
        page.mouse.up()
        page.locator('#undo').click()
        page.locator('#redo').click()
        page.locator('#saveCorrection').click()
        settle()
        draft = next(i for i in httpx.get(base + '/images').json() if i['id'] == first['id'])['correction']
        assert draft and not draft['complete']
        shot('03-draft-correction.png')

        # Import the existing dataset annotation explicitly. The brush draft is not
        # represented as a complete human reference or a timed annotation.
        record = next(r for r in manifest['records'] if r['image'] == first['name'])
        imported = httpx.post(base + f"/images/{first['id']}/corrections?complete=true",
                              files={'file': ('existing-reference.png', (ASSETS / record['mask']).read_bytes(), 'image/png')})
        imported.raise_for_status()
        page.reload()
        settle()
        page.locator('.image-item').first.click()
        settle()
        page.locator('#versionSelect').select_option(frozen['id'])
        page.locator('#predict').click()
        settle()
        shot('04-reference-agreement.png')

        page.locator('[data-tab="compare"]').click()
        page.locator('#versionSelect').select_option(frozen['id'])
        page.locator('#compareVersion').select_option(legacy['id'])
        page.locator('#compareRun').click()
        settle()
        shot('05-side-by-side-comparison.png')
        with page.expect_download() as download:
            page.locator('#export').click()
        export = OUT / 'project-export.zip'
        download.value.save_as(str(export))
        with zipfile.ZipFile(export) as archive:
            assert archive.testzip() is None
            meta = json.loads(archive.read('metadata.json'))
            assert len(meta['images']) == 2
            assert any(v['sha256'] == NEW_SHA for v in meta['versions'])
            assert any(c['complete'] for c in meta['corrections'])
            assert any(p['checkpoint_sha256'] == NEW_SHA for p in meta['predictions'])
        page.reload()
        settle()
        assert page.locator('.image-item').count() == 2
        assert not errors, errors
        context.close()
        video.save_as(str(OUT / 'walkthrough.webm'))
        browser.close()

    result = {'status': 'passed', 'scope': 'Local app workflow on existing target-acquisition images; no retraining or new accuracy benchmark',
              'model_sha256': NEW_SHA, 'project_id': pid, 'images': 2,
              'annotation_provenance': 'Brush/undo/redo draft, then imported existing dataset mask as complete reference; no expert time measured',
              'screenshots': screenshots, 'export_sha256': hashlib.sha256(export.read_bytes()).hexdigest(),
              'export_zip_verified': True, 'reload_persistence': True, 'javascript_errors': errors,
              'video': 'walkthrough.webm'}
    (OUT / 'demo.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
