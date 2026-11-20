"""Loopback-only local inspection API. Browser clients never supply filesystem paths."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import zipfile

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from inspection.engine import MAX_BYTES, binary_metrics, decode_image, infer, render, save_png
from inspection.store import Store, identity, now, process_alive
from src.atomic import atomic_write, write_json

ROOT = Path(__file__).resolve().parents[1]


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class PredictionIn(BaseModel):
    image_id: str
    version_id: str
    threshold: float = Field(default=.5, ge=.01, le=.99, allow_inf_nan=False)
    opacity: float = Field(default=.5, ge=0, le=1, allow_inf_nan=False)
    inference_mode: Literal['resize','tiled'] = 'resize'
    runtime: Literal['pytorch','onnx'] = 'pytorch'


class VersionIn(BaseModel):
    version_id: str


class JobIn(BaseModel):
    parent_id: str
    support_ids: list[str] = Field(min_length=1, max_length=10)
    validation_ids: list[str] = Field(min_length=1, max_length=10)
    epochs: int = Field(default=5, ge=1, le=5)
    seed: int = Field(default=42, ge=0, le=2**31-1)
    source_replay: bool = False


class RankingIn(BaseModel):
    version_id: str
    policy: Literal['entropy','random'] = 'entropy'
    seed: int = Field(default=7301, ge=0, le=2**31-1)


def create_app(workspace=None, start_workers=True):
    store = Store(workspace or os.environ.get('CRACK_WORKSPACE', ROOT / 'artifacts/phase6/workspace'))
    app = FastAPI(title='CrackLab · local inspection', docs_url='/api/docs')
    app.state.store = store
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])
    mutation_lock = threading.RLock()
    processes = {}

    def reconcile_jobs():
        for job in store.list('job'):
            if job['state'] in ('queued', 'running') and not process_alive(job.get('pid')):
                # New queued jobs get a short grace period while Python starts.
                from datetime import datetime, timezone
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(job['updated'])).total_seconds()
                if age > 30:
                    job.update(state='failed', error='Worker was interrupted. Resume from the latest saved optimizer step.', updated=now())
                    store.put('job', job, job['project'])

    @app.middleware('http')
    async def local_requests(request: Request, call_next):
        # Same-origin UI; block drive-by web pages from writing to the local service.
        origin = request.headers.get('origin')
        if origin and origin != str(request.base_url).rstrip('/'):
            return JSONResponse({'detail': 'Use the local application tab'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        return response

    @app.exception_handler(KeyError)
    async def missing(_, exc):
        return JSONResponse({'detail': 'Record not found in this project'}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(_, exc):
        return JSONResponse({'detail': str(exc)}, status_code=400)

    def project(pid):
        return store.get('project', pid)

    def image_record(pid, iid):
        project(pid)
        return store.get('image', iid, pid)

    def original(pid, iid):
        image_record(pid, iid)
        return Image.open(store.root / 'images' / iid / 'original.png').convert('RGB')

    def latest_correction(pid, iid):
        image_record(pid, iid)
        rows = [r for r in store.list('correction', pid) if r['image_id'] == iid]
        return rows[-1] if rows else None

    def launch(job):
        folder = store.folder('jobs', job['id'])
        cancel = folder / 'cancel'
        cancel.unlink(missing_ok=True)
        log = open(folder / 'execution.log', 'ab', buffering=0)
        try:
            worker='phase7_worker.py' if job.get('protocol')=='phase7_source_replay_v1' else 'inspection_worker.py'
            process = subprocess.Popen([sys.executable, str(ROOT / 'scripts' / worker),
                                        '--workspace', str(store.root), '--job', job['id']],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except OSError as exc:
            job.update(state='failed', error=str(exc))
            store.put('job', job, job['project'])
            raise ValueError('Could not start adaptation worker') from exc
        finally:
            log.close()
        processes[job['id']] = process

        def reap():
            code = process.wait()
            current = store.get('job', job['id'], job['project'])
            if current['state'] in ('queued', 'running'):
                current.update(state='failed', error=f'Worker exited with code {code}; resume from the latest checkpoint.', updated=now())
                store.put('job', current, current['project'])
            processes.pop(job['id'], None)
        threading.Thread(target=reap, daemon=True).start()

    @app.get('/api/health')
    def health():
        ai_ready = bool(os.environ.get('GROQ_API_KEY'))
        return {
            'status': 'ok',
            'models': len(store.list('version')),
            'mode': 'local CPU',
            'phase': 7,
            'ai_ready': ai_ready,
            'ai_message': 'Groq vision API is configured.' if ai_ready else 'AI find cracks needs a Groq API key. Restart CrackLab with Start_CrackLab_Groq.ps1.',
        }

    @app.get('/api/projects')
    def projects():
        return store.list('project')

    @app.post('/api/projects', status_code=201)
    def new_project(body: ProjectIn):
        name = body.name.strip()
        if not name:
            raise ValueError('Enter a project name')
        versions = store.list('version')
        default = next((v['id'] for v in versions if v.get('phase') == 4), None)
        return store.put('project', {'id': identity(), 'name': name, 'created': now(),
                                    'active_version': default, 'activation_history': []})

    @app.get('/api/projects/{pid}')
    def get_project(pid: str):
        return project(pid)

    @app.get('/api/projects/{pid}/phase7')
    def phase7_status(pid: str):
        project(pid)
        from inspection.replay import status
        return {'source_replay':status(store),'onnx_exports':store.list('onnx',pid),
                'ranking':'mean_binary_entropy','tile_size':256,'overlap':64}

    @app.post('/api/projects/{pid}/rankings', status_code=201)
    def ranking(pid: str, body: RankingIn):
        project(pid)
        from inspection.advanced import rank_images
        report=rank_images(store,pid,store.version(body.version_id,pid),body.policy,body.seed)
        report.update(id=identity(),created=now())
        return store.put('ranking',report,pid)

    @app.post('/api/projects/{pid}/versions/{vid}/onnx', status_code=201)
    def portable_export(pid: str, vid: str):
        project(pid)
        from inspection.portable import export_model
        return export_model(store,pid,store.version(vid,pid))

    @app.get('/api/projects/{pid}/onnx/{eid}/download')
    def portable_download(pid: str, eid: str):
        project(pid)
        from inspection.portable import export_zip
        return Response(export_zip(store,pid,eid),media_type='application/zip',
                        headers={'Content-Disposition':'attachment; filename="cracklab-onnx.zip"'})

    @app.get('/api/projects/{pid}/versions')
    def versions(pid: str):
        project(pid)
        return [v for v in store.list('version') if v.get('project') in (None, pid)]

    @app.post('/api/projects/{pid}/activate')
    def activate(pid: str, body: VersionIn):
        with mutation_lock:
            value = project(pid)
            version = store.version(body.version_id, pid)
            if version['state'] not in ('reference', 'validated'):
                raise ValueError('Candidate must pass validation before activation')
            # Recheck the stored model before changing the default.
            model_path = store.root / 'models' / (version['sha256'] + '.pt')
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != version['sha256']:
                raise ValueError('Checkpoint checksum mismatch')
            value['activation_history'].append({'previous': value['active_version'], 'next': version['id'], 'time': now()})
            value['active_version'] = version['id']
            return store.put('project', value)

    @app.get('/api/projects/{pid}/images')
    def images(pid: str):
        project(pid)
        return [dict(r, correction=latest_correction(pid, r['id'])) for r in store.list('image', pid)]

    @app.post('/api/projects/{pid}/images', status_code=201)
    async def upload(pid: str, file: UploadFile = File(...)):
        project(pid)
        data = await file.read(MAX_BYTES + 1)
        image = decode_image(data)
        digest = hashlib.sha256(data).hexdigest()
        pixel = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        for r in store.list('image', pid):
            if r['pixel_sha256'] == pixel:
                return dict(r, duplicate=True)
        frozen = store.root / 'frozen_manifest.json'
        matches = []
        if frozen.exists():
            matches = [r['split'] for r in json.loads(frozen.read_text())['records']
                       if r['image_sha256'] == digest or r.get('pixel_sha256') == pixel]
        protected = any(split != 'target_acquisition' for split in matches)
        iid = identity()
        save_png(store.folder('images', iid) / 'original.png', image)
        return store.put('image', {'id': iid, 'project': pid, 'name': Path(file.filename or 'image').name[:150],
                                  'width': image.width, 'height': image.height, 'sha256': digest,
                                  'pixel_sha256': pixel, 'created': now(), 'frozen_splits': matches,
                                  'training_eligible': not protected}, pid)

    @app.get('/api/projects/{pid}/images/{iid}/original')
    def get_original(pid: str, iid: str):
        image_record(pid, iid)
        return FileResponse(store.root / 'images' / iid / 'original.png', media_type='image/png')

    @app.post('/api/projects/{pid}/images/{iid}/corrections', status_code=201)
    async def correction(pid: str, iid: str, complete: bool = False, file: UploadFile = File(...)):
        row = image_record(pid, iid)
        mask = decode_image(await file.read(MAX_BYTES + 1), mask=True)
        if mask.size != (row['width'], row['height']):
            raise ValueError('Correction dimensions must match the original image')
        cid = identity()
        path = store.folder('corrections', cid) / 'mask.png'
        save_png(path, mask)
        return store.put('correction', {'id': cid, 'image_id': iid, 'complete': complete,
                                       'created': now(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, pid)

    @app.get('/api/projects/{pid}/corrections/{cid}/mask')
    def correction_mask(pid: str, cid: str):
        project(pid)
        store.get('correction', cid, pid)
        return FileResponse(store.root / 'corrections' / cid / 'mask.png', media_type='image/png')

    @app.post('/api/projects/{pid}/predict', status_code=201)
    def predict(pid: str, body: PredictionIn):
        image = original(pid, body.image_id)
        version = store.version(body.version_id, pid)
        inference_details={'mode':'resize','runtime':'pytorch','global_resize':True}
        if body.inference_mode=='resize' and body.runtime=='pytorch':
            values, config, elapsed = infer(store, version, image)
        else:
            from inspection.advanced import infer_advanced
            values, config, elapsed, inference_details = infer_advanced(store,version,image,body.inference_mode,body.runtime,pid)
        rendered = render(image, values, body.threshold, body.opacity)
        rid = identity()
        folder = store.folder('predictions', rid)
        atomic_write(folder / 'probability.npy', lambda stream: np.save(stream, values, allow_pickle=False))
        for name, content in rendered.items():
            save_png(folder / (name + '.png'), content)
        mask = np.asarray(rendered['mask']) > 0
        corr = latest_correction(pid, body.image_id)
        metrics = None
        if corr and corr['complete']:
            with Image.open(store.root / 'corrections' / corr['id'] / 'mask.png') as truth:
                metrics = binary_metrics(rendered['mask'], truth)
        value = {'id': rid, 'image_id': body.image_id, 'version_id': version['id'],
                 'checkpoint_sha256': version['sha256'], 'threshold': body.threshold, 'opacity': body.opacity,
                 'created': now(), 'width': image.width, 'height': image.height,
                 'crack_pixels': int(mask.sum()), 'area_fraction': float(mask.mean()),
                 'latency_ms': elapsed, 'latency_scope': 'CPU model loading, preprocessing and inference; excludes export rendering',
                 'preprocessing': {'size': config['image_size'], 'resize': 'bilinear square', 'rgb': True,
                                   'normalization': 'divide_by_255', 'probability_resize': 'bilinear to original dimensions'},
                 'correction_id': corr['id'] if corr and corr['complete'] else None, 'metrics': metrics}
        value['inference']=inference_details
        if body.inference_mode=='tiled':
            value['preprocessing']['resize']='per-tile model input; original spatial scale retained'
        return store.put('prediction', value, pid)

    @app.get('/api/projects/{pid}/predictions/{rid}/{view}')
    def prediction_view(pid: str, rid: str, view: str):
        project(pid)
        store.get('prediction', rid, pid)
        if view not in ('mask', 'overlay', 'heatmap'):
            raise ValueError('Unknown image view')
        return FileResponse(store.root / 'predictions' / rid / (view + '.png'), media_type='image/png')

    @app.post('/api/projects/{pid}/predictions/{rid}/ai-suggestions')
    def ai_suggestions(pid: str, rid: str):
        prediction = store.get('prediction', rid, pid)
        image = original(pid, prediction['image_id'])
        with Image.open(store.root / 'predictions' / rid / 'mask.png') as stored_mask:
            prediction_mask = stored_mask.convert('RGB')
        from inspection.vision_agent import suggest_cracks
        return suggest_cracks(image, prediction_mask)

    @app.post('/api/projects/{pid}/images/{iid}/ai-suggestions')
    def ai_image_suggestions(pid: str, iid: str):
        image = original(pid, iid)
        from inspection.vision_agent import suggest_cracks
        return suggest_cracks(image, Image.new('RGB', image.size, 'black'))

    @app.get('/api/projects/{pid}/export')
    def export(pid: str):
        metadata = project(pid)
        buffer = io.BytesIO()
        predictions = store.list('prediction', pid)
        corrections = store.list('correction', pid)
        images = store.list('image', pid)
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
            payload = {'project': metadata, 'images': images, 'predictions': predictions,
                       'corrections': corrections, 'jobs': store.list('job', pid),
                       'versions': versions(pid), 'units': 'pixels and area fractions; no physical calibration',
                       'exported': now()}
            payload['rankings']=store.list('ranking',pid)
            payload['onnx_exports']=store.list('onnx',pid)
            z.writestr('metadata.json', json.dumps(payload, indent=2))
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=['id', 'image_id', 'version_id', 'threshold', 'crack_pixels', 'area_fraction', 'width', 'height'])
            writer.writeheader()
            writer.writerows({key: p[key] for key in writer.fieldnames} for p in predictions)
            z.writestr('measurements.csv', stream.getvalue())
            for row in images:
                z.write(store.root / 'images' / row['id'] / 'original.png', f'images/{row["id"]}.png')
            for row in predictions:
                for name in ('mask.png', 'overlay.png', 'heatmap.png', 'probability.npy'):
                    z.write(store.root / 'predictions' / row['id'] / name, f'predictions/{row["id"]}/{name}')
            for row in corrections:
                z.write(store.root / 'corrections' / row['id'] / 'mask.png', f'corrections/{row["id"]}.png')
        return Response(buffer.getvalue(), media_type='application/zip',
                        headers={'Content-Disposition': 'attachment; filename="cracklab-project.zip"'})

    @app.get('/api/projects/{pid}/jobs')
    def jobs(pid: str):
        project(pid)
        with mutation_lock:
            reconcile_jobs()
        return store.list('job', pid)

    @app.get('/api/projects/{pid}/jobs/{jid}/log')
    def job_log(pid: str, jid: str):
        project(pid)
        store.get('job', jid, pid)
        path = store.root / 'jobs' / jid / 'execution.log'
        return Response(path.read_bytes()[-32000:] if path.exists() else b'', media_type='text/plain')

    @app.get('/api/projects/{pid}/jobs/{jid}/export')
    def export_job(pid: str, jid: str):
        project(pid)
        job = store.get('job', jid, pid)
        folder = store.root / 'jobs' / jid
        pointer_path = folder / 'checkpoints/latest.json'
        snapshot = None
        if pointer_path.exists():
            pointer = json.loads(pointer_path.read_text())
            snapshot = (folder / 'checkpoints' / pointer['file']).resolve()
            if snapshot.parent != (folder / 'checkpoints').resolve() or hashlib.sha256(snapshot.read_bytes()).hexdigest() != pointer['sha256']:
                raise ValueError('Invalid recovery checkpoint')
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('job.json', json.dumps(job, indent=2))
            for path in sorted(folder.rglob('*')):
                if path.is_file() and path.name not in ('cancel', 'latest.json') and path.suffix != '.pt':
                    z.write(path, path.relative_to(folder).as_posix())
            if snapshot:
                z.writestr('checkpoints/latest.json', json.dumps(pointer, indent=2))
                z.write(snapshot, 'checkpoints/' + snapshot.name)
        return Response(stream.getvalue(), media_type='application/zip',
                        headers={'Content-Disposition': f'attachment; filename="cracklab-job-{jid[:6]}.zip"'})

    @app.post('/api/projects/{pid}/jobs', status_code=201)
    def new_job(pid: str, body: JobIn):
        with mutation_lock:
            project(pid)
            reconcile_jobs()
            parent = store.version(body.parent_id, pid)
            if parent['state'] not in ('reference', 'validated'):
                raise ValueError('Choose a reference or validated parent')
            if any(j['state'] in ('queued', 'running') for j in store.list('job')):
                raise ValueError('Wait for or cancel the current adaptation job')
            all_ids = body.support_ids + body.validation_ids
            if len(set(all_ids)) != len(all_ids):
                raise ValueError('Support and validation images must be distinct, without duplicates')
            rows = []
            for iid in all_ids:
                image = image_record(pid, iid)
                corr = latest_correction(pid, iid)
                if not corr or not corr['complete']:
                    raise ValueError('Every selected image needs a saved, complete correction')
                if not image['training_eligible']:
                    raise ValueError('Frozen evaluation images cannot enter local adaptation or validation')
                rows.append((image, corr))
            if len({image['pixel_sha256'] for image, _ in rows}) != len(rows):
                raise ValueError('Duplicate image content across support and validation')
            replay_root=replay_buffer=None
            if body.source_replay:
                from inspection.replay import verified_buffer
                replay_root,replay_buffer=verified_buffer(store)
            jid = identity()
            folder = store.folder('jobs', jid)
            records = []
            for image, corr in rows:
                data = (store.root / 'images' / image['id'] / 'original.png').read_bytes()
                mask = (store.root / 'corrections' / corr['id'] / 'mask.png').read_bytes()
                records.append({'image_id': image['id'], 'image': image['id'] + '.png',
                                'mask': image['id'] + '_mask.png', 'correction_id': corr['id'],
                                'image_sha256': hashlib.sha256(data).hexdigest(), 'mask_sha256': hashlib.sha256(mask).hexdigest(),
                                'split': 'target_acquisition' if image['id'] in body.support_ids else 'local_validation'})
                atomic_write(folder / (image['id'] + '.png'), lambda stream, data=data: stream.write(data))
                atomic_write(folder / (image['id'] + '_mask.png'), lambda stream, data=mask: stream.write(data))
            manifest = {'protocol': 'phase6_user_corrections_v1', 'records': records}
            if body.source_replay:
                for index,source in enumerate(replay_buffer['records']):
                    entry=dict(source)
                    for field in ('image','mask'):
                        data=(replay_root/source[field]).read_bytes()
                        name=f'source_{index}_{field}'+Path(source[field]).suffix
                        atomic_write(folder/name,lambda stream,data=data:stream.write(data))
                        entry[field]=name
                    records.append(entry)
                manifest['protocol']='phase7_source_replay_v1'
                manifest['source_buffer_sha256']=hashlib.sha256((replay_root/'manifest.json').read_bytes()).hexdigest()
            write_json(folder / 'manifest.json', manifest)
            job = {'id': jid, 'project': pid, 'parent_id': parent['id'], 'parent_sha256': parent['sha256'],
                   'support_ids': body.support_ids, 'validation_ids': body.validation_ids,
                   'epochs': body.epochs, 'seed': body.seed, 'state': 'queued', 'step': 0,
                   'total_steps': body.epochs * ((len(body.support_ids) + 1) // 2), 'created': now(), 'updated': now(),
                   'manifest_sha256': hashlib.sha256((folder / 'manifest.json').read_bytes()).hexdigest(),
                   'settings': {'device': 'cpu', 'learning_rate': .0001, 'batch_size': 2,
                                'freeze_encoder': True, 'validation_threshold': .5,
                                'allowed_validation_dice_drop': .02},
                   'validation': None, 'candidate_id': None, 'error': None}
            if body.source_replay:
                job['protocol']='phase7_source_replay_v1'
                job['settings'].update(source_replay=True,source_per_support=1,allowed_source_validation_dice_drop=.02)
                job['source_replay_images']=16
                job['source_validation_images']=8
            store.put('job', job, pid)
            if start_workers:
                launch(job)
            return job

    @app.post('/api/projects/{pid}/jobs/{jid}/cancel')
    def cancel_job(pid: str, jid: str):
        project(pid)
        job = store.get('job', jid, pid)
        if job['state'] not in ('queued', 'running'):
            raise ValueError('Job is no longer running')
        atomic_write(store.folder('jobs', jid) / 'cancel', lambda stream: stream.write(b'cancel'))
        return {'state': 'cancellation_requested', 'detail': 'The worker will checkpoint after its current optimizer step.'}

    @app.post('/api/projects/{pid}/jobs/{jid}/resume')
    def resume_job(pid: str, jid: str):
        with mutation_lock:
            project(pid)
            reconcile_jobs()
            job = store.get('job', jid, pid)
            if job['state'] not in ('cancelled', 'failed'):
                raise ValueError('Only cancelled or failed jobs can be resumed')
            if any(j['state'] in ('queued', 'running') for j in store.list('job')):
                raise ValueError('Another job is already running')
            job.update(state='queued', error=None, updated=now())
            store.put('job', job, pid)
            if start_workers:
                launch(job)
            return job

    @app.get('/')
    def index():
        entry = ROOT / 'web/dist/index.html'
        if not entry.exists():
            return JSONResponse({'detail': 'Build the React interface with npm ci and npm run build in web/, or use scripts/run_inspection.py.'}, status_code=503)
        return FileResponse(entry)

    app.mount('/static', StaticFiles(directory=ROOT / 'web/dist', check_dir=False), name='static')
    return app


app = create_app()
