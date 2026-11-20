"""Decoder adaptation with replay/distillation and exact step-boundary recovery."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import shutil

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F
from src.atomic import write_json
from src.boundaries import boundary_loss
from src.checkpoints import load_checkpoint, save_checkpoint
from src.dataset import CrackDataset, verify_files
from src.model_factory import build_model
from src.segmentation import region_loss


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def code_identity():
    root = Path(__file__).resolve().parents[1]
    names = ['followup/adaptation.py', 'followup/study.py', 'followup/scenes.py', 'scripts/experiments/run_followup.py',
             'scripts/experiments/train_baseline.py', 'src/checkpoints.py', 'src/atomic.py', 'src/dataset.py', 'src/models.py',
             'inspection/advanced.py', 'inspection/engine.py', 'configs/phase7.json',
             'src/model_factory.py', 'src/litecrack.py', 'src/segmentation.py', 'src/boundaries.py']
    return {n: sha(root / n) for n in names}


def region(output):
    return output[0] if isinstance(output, tuple) else output


def distillation_loss(student, teacher):
    """Bernoulli KL at T=1, averaged over pixels; zero at the unchanged parent."""
    target = teacher.detach().sigmoid()
    cross = F.binary_cross_entropy_with_logits(student, target)
    entropy = -(target * F.logsigmoid(teacher.detach()) +
                (1 - target) * F.logsigmoid(-teacher.detach())).mean()
    return (cross - entropy).clamp_min(0)


def select_rows(manifest, ids, allowed):
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Nonempty unique IDs required')
    lookup = {r['image_id']: r for r in manifest['records']}
    if len(lookup) != len(manifest['records']):
        raise ValueError('Duplicate manifest IDs')
    if any(i not in lookup or lookup[i]['split'] not in allowed for i in ids):
        raise ValueError('Training split violation: ' + str(sorted(allowed)))
    return [lookup[i] for i in ids]


def adapt(parent, parent_sha, manifest, manifest_sha, data_root, supports,
          settings, seed, run, resume=False, stop_after_steps=None, extra=None):
    """Extra is an audited external scene_train manifest/root; never test scenes."""
    if settings['source_per_support'] < 0 or not isinstance(settings['source_per_support'], int):
        raise ValueError('Replay ratio must be a nonnegative integer')
    if settings['distill_weight'] < 0 or settings['learning_rate'] <= 0:
        raise ValueError('Invalid loss weight or learning rate')
    if settings['distill_weight'] and not settings['source_per_support']:
        raise ValueError('Source distillation requires replay')
    if min(settings['epochs'], settings['batch_size']) < 1:
        raise ValueError('Positive support budget required')
    support_rows = select_rows(manifest, supports, {'target_acquisition'})
    replay_rows = [r for r in manifest['records'] if r['split'] == 'source_train']
    if settings['source_per_support'] and not replay_rows:
        raise ValueError('Source-training replay required')
    verify_files(support_rows + (replay_rows if settings['source_per_support'] else []), data_root)
    run = Path(run)
    if not resume and run.exists() and any(run.iterdir()):
        raise ValueError('Existing run requires explicit resume')
    if resume and not (run/'checkpoints/latest.json').exists():
        raise ValueError('No recoverable checkpoint')
    run.mkdir(parents=True, exist_ok=True)
    config = {'protocol': 'followup_decoder_v1', 'settings': settings, 'seed': seed,
              'parent_sha256': parent_sha, 'support_ids': supports,
              'model_config': parent['config'], 'code': code_identity(),
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'extra_manifest_sha256': extra['sha256'] if extra else None}
    device = torch.device(settings['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU required: reconnect the CUDA notebook kernel')
    torch.set_num_threads(settings.get('threads', 2))
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(seed)
    model = build_model(parent['config']).to(device)
    model.load_state_dict(parent['model'])
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    for p in model.encoder.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=settings['learning_rate'])
    size = parent['config']['image_size']
    support_data = CrackDataset({'records': support_rows}, data_root, 'target_acquisition', size)
    replay_data = CrackDataset({'records': replay_rows}, data_root, 'source_train', size) if settings['source_per_support'] else None
    extra_data = None
    if extra:
        from followup.scenes import audit_scenes
        audited = audit_scenes(extra['manifest'], extra['root'], manifest)
        if sha(extra['manifest']) != extra['sha256']: raise ValueError('External manifest changed')
        extra_data = CrackDataset(audited, extra['root'], 'scene_train', size)
    epoch = step = cursor = 0; order = None; history = []
    if resume:
        state = load_checkpoint(run/'checkpoints', model=model, optimizer=optimizer, scheduler=None,
                                scaler=None, expected_manifest_hash=manifest_sha, expected_config=config)
        epoch, step = state['epoch'], state['global_step']
        order, cursor, history = (state['sampler_state'][k] for k in ('order', 'cursor', 'history'))
    else:
        write_json(run/'config.json', config)
    stop = [False]
    previous = {s: signal.signal(s, lambda *_: stop.__setitem__(0, True)) for s in (signal.SIGINT, signal.SIGTERM)}
    def snapshot():
        # Keep room for a recovery export rather than filling ephemeral storage.
        estimated=sum(p.numel()*p.element_size() for p in model.parameters())*4
        if shutil.disk_usage(run).free < estimated+2*1024**3:
            raise RuntimeError('Checkpoint storage low: preserve/export latest and best, then resume in restored storage')
        save_checkpoint(run/'checkpoints', model=model, optimizer=optimizer, scheduler=None,
                        scaler=None, epoch=epoch, global_step=step,
                        sampler_state={'order': order, 'cursor': cursor, 'history': history},
                        config=config, manifest_hash=manifest_sha,
                        metrics=history[-1] if history else {}, best_metric=None)
    try:
        while epoch < settings['epochs']:
            if order is None: order = torch.randperm(len(support_data)).tolist(); cursor = 0
            model.train(); model.encoder.eval()
            while cursor < len(order):
                indices = order[cursor:cursor + settings['batch_size']]
                samples = [support_data[i] for i in indices]
                source_count = len(indices) * settings['source_per_support']
                source = [replay_data[random.randrange(len(replay_data))] for _ in range(source_count)]
                # Each support example receives one audited broad-scene training example.
                background = [extra_data[random.randrange(len(extra_data))] for _ in indices] if extra_data else []
                all_samples = samples + source + background
                x = torch.stack([s[0] for s in all_samples]).to(device)
                y = torch.stack([s[1] for s in all_samples]).to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(x); logits = region(output)
                supervised = region_loss(logits, y, 'bce_dice')
                if isinstance(output, tuple):
                    supervised = supervised + parent['config'].get('boundary_weight', .2) * boundary_loss(output[1], y)
                kd = logits.new_zeros(())
                if settings['distill_weight']:
                    source_slice = slice(len(samples), len(samples) + source_count)
                    with torch.no_grad(): target = region(teacher(x[source_slice]))
                    kd = distillation_loss(logits[source_slice], target)
                loss = supervised + settings['distill_weight'] * kd
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite adaptation loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step(); cursor += len(indices); step += 1
                history.append({'step': step, 'epoch': epoch, 'loss': float(loss.detach()),
                                'distillation': float(kd.detach()), 'support_examples': len(samples),
                                'source_examples': source_count, 'scene_examples': len(background)})
                snapshot()
                if stop[0] or (stop_after_steps is not None and step >= stop_after_steps):
                    write_json(run/'status.json', {'state': 'stopped_safely', 'step': step}); return None
            epoch += 1; cursor = 0; order = None; snapshot()
        for key, value in model.state_dict().items():
            if key.startswith('encoder.') and not torch.equal(value.cpu(), parent['model'][key].cpu()):
                raise ValueError('Frozen encoder changed')
        write_json(run/'history.json', history)
        write_json(run/'status.json', {'state': 'complete', 'step': step, 'encoder_unchanged': True,
                    'support_examples': sum(r['support_examples'] for r in history),
                    'source_examples': sum(r['source_examples'] for r in history),
                    'scene_examples': sum(r['scene_examples'] for r in history)})
        return model
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)
