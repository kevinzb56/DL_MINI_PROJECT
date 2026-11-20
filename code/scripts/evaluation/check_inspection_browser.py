"""Browser evidence on synthetic fixtures using real imported checkpoints.

Run against a dedicated QA workspace, never a user's inspection project.
Requires the optional playwright package and its Chromium browser.
"""
import argparse
import io
import json
import os
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright


def check(url, output, phase7=False):
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(2):
        rng = np.random.default_rng(100 + index)
        pixels = np.clip(rng.normal(100 + index * 5, 15, (320, 480, 1)), 0, 255).astype(np.uint8)
        image = Image.fromarray(np.repeat(pixels, 3, axis=2))
        draw = ImageDraw.Draw(image)
        draw.line([(200 + index*20, 0), (230, 70), (205, 135), (270, 185), (235, 245), (290, 320)], fill=(20, 23, 20), width=4)
        path = output / f'synthetic-road-{index}.png'
        image.save(path); paths.append(path)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1536, 'height': 1100}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function("document.querySelector('#connection').textContent.includes('models ready') && document.querySelector('#busy').hidden")
        page.get_by_role('heading', name='Small model. Sharper perspective.').wait_for()
        page.screenshot(path=str(output / '00-react-overview.png'), full_page=True, animations="disabled")
        page.get_by_role('button', name='1 image', exact=True).click()
        assert '48.00' in page.locator('.experiment-card').inner_text()
        page.get_by_role('button', name='10 images', exact=True).click()
        page.get_by_role('button', name='Research & results', exact=True).click()
        page.get_by_role('heading', name='Measured progress. Honest tradeoffs.').wait_for()
        assert page.locator('table tbody tr').count() == 4
        page.screenshot(path=str(output / '00-react-research.png'), full_page=True, animations="disabled")
        page.locator('.nav-item').filter(has_text='Inspection studio').click()
        page.screenshot(path=str(output / '01-empty-workspace.png'), full_page=True, animations="disabled")
        page.locator('#upload').set_input_files([str(path) for path in paths])
        page.wait_for_function("document.querySelectorAll('.image-item').length === 2 && document.querySelector('#busy').hidden")
        page.locator('.image-item').first.click()
        page.wait_for_function("document.querySelector('#busy').hidden")
        portable_report=None
        if phase7:
            page.locator('[data-tab="adapt"]').click()
            page.locator('#rankPool').click()
            page.wait_for_function("document.querySelectorAll('.ranking-row').length === 2 && document.querySelector('#busy').hidden")
            page.screenshot(path=str(output/'phase7-ranking.png'),full_page=True,animations='disabled')
            page.locator('#rankingPolicy').select_option('random')
            page.locator('#rankPool').click()
            page.wait_for_function("document.querySelector('.ranking-caption').textContent.includes('random') && document.querySelector('#busy').hidden")
            page.locator('[data-tab="studio"]').click()
            page.locator('#inferenceMode').select_option('tiled')
            page.locator('#predict').click()
            page.wait_for_function("document.querySelector('#predictionViewport').classList.contains('loaded') && document.querySelector('#busy').hidden",timeout=120000)
            page.screenshot(path=str(output/'phase7-tiled.png'),full_page=True,animations='disabled')
            page.locator('#exportOnnx').click()
            page.locator('#downloadOnnx').wait_for(timeout=180000)
            with page.expect_download() as downloaded:
                page.locator('#downloadOnnx').click()
            portable=output/'browser-onnx.zip';downloaded.value.save_as(str(portable))
            with zipfile.ZipFile(portable) as z:
                import hashlib
                portable_report=json.loads(z.read('parity.json'))
                assert hashlib.sha256(z.read('model.onnx')).hexdigest()==portable_report['onnx_sha256']
                assert portable_report['max_probability_difference']<=.001
                assert portable_report['audited_source_validation']
            page.locator('#runtimeSelect').select_option('onnx')
            page.locator('#predict').click()
            page.wait_for_function("document.querySelector('#predictionViewport').classList.contains('loaded') && document.querySelector('#busy').hidden",timeout=120000)
            page.screenshot(path=str(output/'phase7-portable.png'),full_page=True,animations='disabled')
            page.locator('#runtimeSelect').select_option('pytorch')
            page.locator('#inferenceMode').select_option('resize')
        page.locator('#predict').click()
        page.wait_for_function("document.querySelector('#predictionViewport').classList.contains('loaded') && document.querySelector('#busy').hidden", timeout=120000)
        page.locator('#usePrediction').click()
        page.wait_for_function("document.querySelector('#busy').hidden")
        # Real pointer coordinates, followed by undo and redo, exercise the canvas editor.
        canvas = page.locator('#editorCanvas')
        rect = canvas.bounding_box()
        page.mouse.move(rect['x'] + rect['width'] * .4, rect['y'] + rect['height'] * .4)
        page.mouse.down()
        page.mouse.move(rect['x'] + rect['width'] * .5, rect['y'] + rect['height'] * .6, steps=8)
        page.mouse.up()
        assert page.locator('#undo').is_enabled()
        page.locator('#undo').click()
        assert page.locator('#redo').is_enabled()
        page.locator('#redo').click()
        page.locator('#complete').check()
        page.locator('#saveCorrection').click()
        page.wait_for_function("document.querySelector('#saveBadge').textContent === 'SAVED' && document.querySelector('#busy').hidden")
        page.locator('#predict').click()
        page.wait_for_function("document.querySelector('#qualityMetrics').textContent !== '—' && document.querySelector('#busy').hidden", timeout=120000)
        page.screenshot(path=str(output / '02-inspection-editor.png'), full_page=True, animations="disabled")
        # Second distinct image becomes the validation correction.
        page.locator('.image-item').nth(1).click()
        page.wait_for_function("document.querySelector('#busy').hidden")
        page.locator('#predict').click()
        page.wait_for_function("document.querySelector('#predictionViewport').classList.contains('loaded') && document.querySelector('#busy').hidden", timeout=120000)
        page.locator('#usePrediction').click()
        page.wait_for_function("document.querySelector('#busy').hidden")
        page.locator('#complete').check()
        page.locator('#saveCorrection').click()
        page.wait_for_function("document.querySelector('#saveBadge').textContent === 'SAVED' && document.querySelector('#busy').hidden")
        page.locator('[data-tab="adapt"]').click()
        page.locator('#supportList select').nth(0).select_option('support')
        page.locator('#supportList select').nth(1).select_option('validation')
        page.locator('#epochs').select_option('1')
        if phase7:
            page.locator('#sourceReplay').check()
        page.locator('#startJob').click()
        page.wait_for_function("document.querySelector('#jobs').textContent.includes('completed')", timeout=180000)
        page.wait_for_function("document.querySelector('#versionList').textContent.includes('adaptation')", timeout=10000)
        page.screenshot(path=str(output / '03-adaptation-review.png'), full_page=True, animations="disabled")
        # Activate only if the actual validation gate passed, then roll back.
        candidate = page.locator('.version-row').filter(has_text='Replay adaptation' if phase7 else 'Local adaptation').last
        activated = False
        if candidate.locator('button').is_enabled():
            candidate.locator('button').click()
            page.wait_for_function("document.querySelector('#busy').hidden")
            activated = True
            page.locator('.version-row').filter(has_text='LiteCrack-BA · base').locator('button').click()
            page.wait_for_function("document.querySelector('#busy').hidden")
        page.locator('[data-tab="compare"]').click()
        page.locator('#compareVersion').select_option(label='LiteCrack · base')
        page.locator('#compareRun').click()
        page.wait_for_function("document.querySelector('#editorViewport').classList.contains('loaded') && document.querySelector('#busy').hidden", timeout=120000)
        page.screenshot(path=str(output / '04-model-comparison.png'), full_page=True, animations="disabled")
        with page.expect_download() as download:
            page.locator('#export').click()
        export = output / 'browser-project-export.zip'
        download.value.save_as(str(export))
        with zipfile.ZipFile(export) as z:
            metadata = json.loads(z.read('metadata.json'))
            assert len(metadata['images']) == 2 and len(metadata['corrections']) == 2
            assert metadata['jobs'][0]['state'] == 'completed'
            assert all(p['width'] == 480 and p['height'] == 320 for p in metadata['predictions'])
            assert any(p['metrics'] is not None for p in metadata['predictions'])
        page.reload()
        page.wait_for_function("document.querySelectorAll('.image-item').length === 2 && document.querySelector('#busy').hidden")
        # Responsive layout and a basic overflow check.
        page.set_viewport_size({'width': 390, 'height': 844})
        page.screenshot(path=str(output / '05-mobile-workspace.png'), full_page=True, animations="disabled")
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1')
        assert not errors, errors
        browser.close()
    report = {'status': 'passed', 'fixtures': 'Two procedural synthetic road images; functional evidence only.',
              'checkpoint_source': 'Real hash-verified Phase 4 base; paired comparison against Phase 3 base.',
              'browser': 'Playwright Chromium, desktop 1536x1100 and mobile 390x844',
              'workflow': ['upload', 'inference', 'brush', 'undo', 'redo', 'complete correction save',
                           'metrics against saved correction', 'subprocess adaptation', 'validation review',
                           'paired comparison', 'ZIP export', 'reload persistence', 'mobile overflow'],
              'candidate_activated_and_rolled_back': activated, 'javascript_errors': errors,
              'job': metadata['jobs'][0]}
    if phase7:
        assert metadata['jobs'][0]['settings']['source_replay'] is True
        assert 'source_validation' in metadata['jobs'][0]['validation']
        report['phase7']={'entropy_and_random_ranking':True,'tiled_native_and_onnx':True,
                          'onnx_parity':portable_report,'source_replay_job':True}
    (output / 'browser-verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', str(root / 'artifacts/phase6/browsers'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--output', type=Path, default=root / 'output/phase6/browser')
    parser.add_argument('--phase7', action='store_true')
    args = parser.parse_args()
    check(args.url, args.output, args.phase7)
