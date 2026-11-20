"""Isolated, bounded CPU adaptation for user corrections; never changes research runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
from inspection.engine import binary_metrics, load_state, model_config, probability, tensor_image
from inspection.store import Store, now
from src.atomic import atomic_write, write_json
from src.boundaries import boundary_loss
from src.checkpoints import load_checkpoint, save_checkpoint
from src.dataset import verify_files
from src.model_factory import build_model
from src.segmentation import region_loss


def run(workspace, jid, stop_after_steps=None):
    store = Store(workspace)
    job = store.get('job', jid)
    folder = store.folder('jobs', jid)
    cancelled = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: cancelled.__setitem__(0, True))

    def update(**fields):
        job.update(fields, updated=now())
        store.put('job', job, job['project'])
        print(json.dumps({k: job[k] for k in ('state', 'step', 'total_steps', 'updated')}), flush=True)

    try:
        update(state='running', pid=os.getpid(), error=None)
        manifest_path = folder / 'manifest.json'
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != job['manifest_sha256']:
            raise ValueError('Correction snapshot manifest changed')
        manifest = json.loads(manifest_path.read_text())
        verify_files(manifest['records'], folder)
        settings = job['settings']
        if settings != {'device': 'cpu', 'learning_rate': .0001, 'batch_size': 2, 'freeze_encoder': True,
                        'validation_threshold': .5, 'allowed_validation_dice_drop': .02}:
            raise ValueError('Unsupported adaptation preset')
        if not 1 <= job['epochs'] <= 5:
            raise ValueError('Invalid epoch budget')
        support = [r for r in manifest['records'] if r['split'] == 'target_acquisition']
        validation = [r for r in manifest['records'] if r['split'] == 'local_validation']
        if not support or not validation or {r['image_id'] for r in support} & {r['image_id'] for r in validation}:
            raise ValueError('Separate support and validation images are required')
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed'])
        version = store.version(job['parent_id'], job['project'])
        if version['sha256'] != job['parent_sha256']:
            raise ValueError('Parent model changed')
        parent = load_state(store, version)
        config = model_config(parent)
        model = build_model(config)
        model.load_state_dict(parent['model'])
        for p in model.encoder.parameters():
            p.requires_grad = False
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.0001)
        source_root = Path(__file__).resolve().parents[1]
        code_hash = hashlib.sha256(b''.join((source_root / n).read_bytes() for n in
            ('scripts/inspection_worker.py', 'inspection/engine.py', 'src/litecrack.py', 'src/checkpoints.py',
             'src/model_factory.py', 'src/segmentation.py', 'src/boundaries.py'))).hexdigest()
        recovery_config = {'protocol': 'phase6_user_corrections_v1', 'model_config': config,
                           'parent_sha256': version['sha256'], 'settings': settings,
                           'epochs': job['epochs'], 'seed': job['seed'], 'code_sha256': code_hash,
                           'torch': str(torch.__version__), 'support_ids': job['support_ids'],
                           'validation_ids': job['validation_ids']}
        epoch = step = cursor = 0
        order = None
        history = []
        if (folder / 'checkpoints/latest.json').exists():
            state = load_checkpoint(folder / 'checkpoints', model=model, optimizer=optimizer,
                                    scheduler=None, scaler=None, expected_manifest_hash=job['manifest_sha256'],
                                    expected_config=recovery_config)
            epoch, step = state['epoch'], state['global_step']
            order, cursor, history = (state['sampler_state'][k] for k in ('order', 'cursor', 'history'))
        write_json(folder / 'config.json', recovery_config)

        def snapshot():
            save_checkpoint(folder / 'checkpoints', model=model, optimizer=optimizer, scheduler=None,
                            scaler=None, epoch=epoch, global_step=step,
                            sampler_state={'order': order, 'cursor': cursor, 'history': history},
                            config=recovery_config, manifest_hash=job['manifest_sha256'],
                            metrics=history[-1] if history else {}, best_metric=None)

        def stop_requested():
            return cancelled[0] or (folder / 'cancel').exists()

        def stop():
            snapshot()
            update(state='cancelled', step=step, epoch=epoch)

        while epoch < job['epochs']:
            if stop_requested():
                stop(); return
            if order is None:
                order = torch.randperm(len(support)).tolist()
                cursor = 0
            model.train(); model.encoder.eval()
            while cursor < len(order):
                if stop_requested():
                    stop(); return
                batch = [support[i] for i in order[cursor:cursor + 2]]
                samples = []
                for row in batch:
                    with Image.open(folder / row['image']) as image:
                        x = tensor_image(image.convert('RGB'), config['image_size'])[0]
                    with Image.open(folder / row['mask']) as mask:
                        y = torch.from_numpy((np.array(mask.resize((config['image_size'], config['image_size']), Image.Resampling.NEAREST)) > 0).astype(np.float32))[None]
                    samples.append((x, y))
                x, y = (torch.stack([s[i] for s in samples]) for i in (0, 1))
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x)
                logits = prediction[0] if isinstance(prediction, tuple) else prediction
                loss = region_loss(logits, y, 'bce_dice')
                if isinstance(prediction, tuple):
                    loss = loss + config.get('boundary_weight', .2) * boundary_loss(prediction[1], y)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite adaptation loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                cursor += len(batch); step += 1
                history.append({'step': step, 'epoch': epoch, 'loss': float(loss.detach())})
                snapshot()
                update(step=step, epoch=epoch, loss=history[-1]['loss'])
                if stop_after_steps is not None and step >= stop_after_steps:
                    stop(); return
            epoch += 1
            cursor = 0; order = None
            snapshot()
        if stop_requested():
            stop(); return
        for key, value in model.state_dict().items():
            if key.startswith('encoder.') and not torch.equal(value.cpu(), parent['model'][key].cpu()):
                raise ValueError('Encoder changed')
        baseline = build_model(config)
        baseline.load_state_dict(parent['model'])
        scores = []
        for row in validation:
            if stop_requested():
                stop(); return
            with Image.open(folder / row['image']) as image, Image.open(folder / row['mask']) as truth:
                image = image.convert('RGB')
                parent_mask = (probability(baseline, image, config['image_size']) >= .5).astype(np.uint8) * 255
                candidate_mask = (probability(model, image, config['image_size']) >= .5).astype(np.uint8) * 255
                scores.append({'image_id': row['image_id'], 'parent': binary_metrics(parent_mask, truth),
                               'candidate': binary_metrics(candidate_mask, truth)})
        parent_dice = float(np.mean([r['parent']['dice'] for r in scores]))
        candidate_dice = float(np.mean([r['candidate']['dice'] for r in scores]))
        passed = candidate_dice >= parent_dice - settings['allowed_validation_dice_drop']
        report = {'passed': passed, 'parent_dice': parent_dice, 'candidate_dice': candidate_dice,
                  'dice_change_points': 100 * (candidate_dice - parent_dice), 'images': scores,
                  'threshold': .5, 'encoder_unchanged': True,
                  'gate': 'Finite predictions, unchanged encoder, local validation Dice drop at most 2 points.',
                  'scope': 'User-designated local validation corrections; not the frozen research benchmark.'}
        write_json(folder / 'validation.json', report)
        if stop_requested():
            stop(); return
        pointer = json.loads((folder / 'checkpoints/latest.json').read_text())
        data = (folder / 'checkpoints' / pointer['file']).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != pointer['sha256']:
            raise ValueError('Final checkpoint hash mismatch')
        model_path = store.folder('models') / (sha + '.pt')
        if not model_path.exists():
            atomic_write(model_path, lambda stream: stream.write(data))
        candidate = {'id': sha[:32], 'sha256': sha, 'project': job['project'], 'phase': 6,
                     'label': f'Local adaptation · {jid[:6]}', 'created': now(),
                     'state': 'validated' if passed else 'rejected', 'parent_id': job['parent_id'],
                     'job_id': jid, 'validation': report, 'description': 'Candidate from saved user corrections; activation is manual.'}
        store.put('version', candidate)
        update(state='completed', step=step, epoch=epoch, validation=report, candidate_id=candidate['id'])
    except BaseException as exc:
        update(state='failed', error=str(exc))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    run(args.workspace, args.job)
