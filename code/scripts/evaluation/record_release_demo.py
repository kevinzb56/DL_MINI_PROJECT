"""Record a real-image workflow against a running isolated release server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile
import httpx
from playwright.sync_api import sync_playwright


def record(url,assets,output):
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((assets/'demo/manifest.json').read_text())
    originals={r['role']:r for r in manifest['records']}
    errors=[];steps=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        context=browser.new_context(viewport={'width':1440,'height':1000},record_video_dir=str(output.parent/'raw_video'),record_video_size={'width':1440,'height':1000})
        page=context.new_page();video=page.video
        page.on('pageerror',lambda error:errors.append(str(error)))
        def settle():page.wait_for_function("document.querySelector('#busy').hidden",timeout=180000)
        def shot(name):
            page.evaluate('window.scrollTo(0,0)')
            page.screenshot(path=str(output/name),full_page=True,animations='disabled')
            steps.append(name)
        def nav(name):page.locator(f'[data-tab="{name}"]').click()
        page.goto(url+'/#overview');page.wait_for_function("document.querySelector('#connection').textContent.includes('models ready') && document.querySelector('#busy').hidden")
        pid=page.locator('#projectSelect').input_value();base=url+f'/api/projects/{pid}'
        versions=httpx.get(base+'/versions').json();static=next(v for v in versions if v['phase']==4)
        shot('01-overview.png');nav('studio')
        page.locator('#upload').set_input_files([str(assets/'demo'/originals[role]['image']) for role in ('support','validation')]);settle()
        page.wait_for_function("document.querySelectorAll('.image-item').length===2")
        nav('adapt');page.locator('#rankPool').click();settle();shot('02-real-acquisition-ranking.png')
        nav('studio');page.locator('.image-item').first.click();settle();page.locator('#predict').click();settle()
        page.wait_for_function("document.querySelector('#predictionViewport').classList.contains('loaded')")
        shot('03-real-prediction.png')
        page.locator('#usePrediction').click();settle()
        rect=page.locator('#editorCanvas').bounding_box()
        page.mouse.move(rect['x']+.35*rect['width'],rect['y']+.35*rect['height']);page.mouse.down()
        page.mouse.move(rect['x']+.5*rect['width'],rect['y']+.6*rect['height'],steps=10);page.mouse.up()
        page.locator('#undo').click();page.locator('#redo').click();page.locator('#undo').click()
        # Save a draft from real predicted output to exercise UI persistence first.
        page.locator('#saveCorrection').click();settle()
        images=httpx.get(base+'/images').json()
        source_map={}
        for role in ('support','validation'):
            row=originals[role];uploaded=next(i for i in images if i['name']==row['image']);source_map[role]=uploaded['id']
            mask=(assets/'demo'/row['mask']).read_bytes()
            if hashlib.sha256(mask).hexdigest()!=row['mask_sha256']:raise ValueError('Demo mask source changed')
            result=httpx.post(base+f'/images/{uploaded["id"]}/corrections?complete=true',files={'file':('reference-mask.png',mask,'image/png')})
            result.raise_for_status()
        # Imported masks are explicitly reference annotations, not timed human corrections.
        page.reload();settle();page.locator('.image-item').first.click();settle()
        page.locator('#predict').click();settle();shot('04-reference-correction-and-metrics.png')
        page.locator('#inferenceMode').select_option('tiled');page.locator('#predict').click();settle();shot('05-tiled-inference.png')
        page.locator('#exportOnnx').click();page.locator('#downloadOnnx').wait_for(timeout=180000);settle()
        with page.expect_download() as portable:page.locator('#downloadOnnx').click()
        portable.value.save_as(str(output/'onnx-export.zip'))
        with zipfile.ZipFile(output/'onnx-export.zip') as z:
            parity=json.loads(z.read('parity.json'))
            assert hashlib.sha256(z.read('model.onnx')).hexdigest()==parity['onnx_sha256']
        page.locator('#runtimeSelect').select_option('onnx');page.locator('#predict').click();settle();shot('06-portable-inference.png')
        page.locator('#runtimeSelect').select_option('pytorch');page.locator('#inferenceMode').select_option('resize')
        nav('adapt')
        page.locator('#supportList select').nth(0).select_option('support');page.locator('#supportList select').nth(1).select_option('validation')
        page.locator('#epochs').select_option('1');page.locator('#sourceReplay').check();page.locator('#startJob').click()
        page.wait_for_function("document.querySelector('#jobs').textContent.includes('completed')",timeout=180000)
        settle();page.wait_for_function("document.querySelectorAll('.version-row').length>=3",timeout=15000)
        jobs=httpx.get(base+'/jobs').json();job=jobs[-1]
        assert job['state']=='completed' and job['settings']['source_replay']
        shot('07-replay-validation.png')
        activated=False
        if job['validation']['passed']:
            page.locator('.version-row').last.locator('button').click();settle();activated=True
            page.wait_for_function('(id)=>document.querySelector("#versionSelect").value===id',arg=job['candidate_id'])
            shot('08-candidate-activation.png')
            page.locator('.version-row').nth(1).locator('button').click();settle()
        nav('compare');page.locator('.image-item').nth(1).click();settle()
        page.locator('#versionSelect').select_option(static['id']);page.locator('#compareVersion').select_option(job['candidate_id'])
        page.locator('#compareRun').click();settle();shot('09-real-model-comparison.png')
        with page.expect_download() as exported:page.locator('#export').click()
        exported.value.save_as(str(output/'project-export.zip'))
        nav('adapt')
        with page.expect_download() as recovered:page.get_by_text('Download recovery ZIP',exact=True).click()
        recovered.value.save_as(str(output/'job-recovery.zip'))
        with zipfile.ZipFile(output/'job-recovery.zip') as z:
            pointer=json.loads(z.read('checkpoints/latest.json'))
            assert hashlib.sha256(z.read('checkpoints/'+pointer['file'])).hexdigest()==pointer['sha256']
        page.reload();settle();assert page.locator('.version-row').count()>=3
        nav('overview');page.wait_for_timeout(1000)
        assert not errors,errors
        context.close();video.save_as(str(output/'walkthrough.webm'));browser.close()
    report={'status':'passed','url':url,'real_images':True,'image_provenance':manifest,
            'mask_provenance':'Existing acquisition masks imported after a brush/undo/redo/draft-save demonstration; no annotation-time or expert-effort claim.',
            'scope':'Application demonstration on known acquisition images; not independent query/holdout evaluation.',
            'project_id':pid,'source_map':source_map,'screenshots':steps,'javascript_errors':errors,
            'candidate_activated_and_rolled_back':activated,'job':job,'onnx_parity':parity,
            'recovery_zip_verified':True,'reload_persistence':True,'video':'walkthrough.webm'}
    (output/'demo.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','real_images':True,'job_state':job['state'],'candidate_passed':job['validation']['passed'],
                      'activated_and_rolled_back':activated,'javascript_errors':errors,'video':str(output/'walkthrough.webm')},indent=2))


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH',str(root/'artifacts/phase6/browsers'))
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8769')
    parser.add_argument('--assets',type=Path,default=root/'release_assets');parser.add_argument('--output',type=Path,default=root/'output/phase8/demo')
    args=parser.parse_args();record(args.url,args.assets,args.output)
