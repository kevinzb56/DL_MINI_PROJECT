"""Shared preprocessing, trusted-checkpoint inference and original-size exports."""
import hashlib
import io
from pathlib import Path
import threading
import time
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from src.atomic import atomic_write

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 16_000_000
INFERENCE_LOCK = threading.Lock()


def decode_image(data, mask=False):
    if not data or len(data) > MAX_BYTES:
        raise ValueError('Choose a PNG/JPEG image no larger than 20 MB')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in (('PNG',) if mask else ('PNG', 'JPEG')):
                    raise ValueError('Masks must be PNG' if mask else 'Only PNG and JPEG images are supported')
                if source.width * source.height > MAX_PIXELS or min(source.size) < 8:
                    raise ValueError('Image dimensions must be at least 8 pixels and at most 16 megapixels')
                source.load()
                if mask:
                    # Browser canvases encode RGB/RGBA PNGs. Require opaque grayscale
                    # pixels so conversion preserves every binary pixel exactly.
                    rgba = np.asarray(source.convert('RGBA'))
                    if (not np.all(rgba[..., 3] == 255) or
                        not np.all(rgba[..., 0] == rgba[..., 1]) or
                        not np.all(rgba[..., 0] == rgba[..., 2]) or
                        not set(np.unique(rgba[..., 0]).tolist()) <= {0, 255}):
                        raise ValueError('Correction must be a grayscale binary PNG with values 0 and 255')
                    return source.convert('L').copy()
                return ImageOps.exif_transpose(source).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('The image is corrupt or cannot be safely decoded') from exc


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


def save_png(path, image):
    atomic_write(path, lambda stream: image.save(stream, format='PNG'))


def model_config(state):
    config = state['config']
    return config if 'architecture' in config else config['model_config']


def load_state(store, version):
    import torch
    path = store.root / 'models' / (version['sha256'] + '.pt')
    if hashlib.sha256(path.read_bytes()).hexdigest() != version['sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    # Registry contains only locally imported, hash-pinned project backups and our own jobs.
    return torch.load(path, map_location='cpu', weights_only=False)


def tensor_image(image, size):
    import torch
    array = np.array(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def probability(model, image, size):
    import torch
    model.eval()
    with torch.inference_mode():
        output = model(tensor_image(image, size))
        logits = output[0] if isinstance(output, tuple) else output
        if not torch.isfinite(logits).all():
            raise ValueError('Model returned nonfinite probabilities')
        small = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return np.asarray(Image.fromarray(small).resize(image.size, Image.Resampling.BILINEAR)).copy()


def infer(store, version, image):
    import torch
    from src.model_factory import build_model
    started = time.perf_counter()
    with INFERENCE_LOCK:
        torch.set_num_threads(2)
        state = load_state(store, version)
        config = model_config(state)
        model = build_model(config)
        model.load_state_dict(state['model'])
        values = probability(model, image, config['image_size'])
    return values, config, round((time.perf_counter() - started) * 1000, 1)


def render(image, values, threshold, opacity):
    mask = values >= threshold
    rgb = np.array(image, dtype=np.float32)
    rgb[mask] = rgb[mask] * (1 - opacity) + np.array([33, 213, 158]) * opacity
    heat = np.zeros((*values.shape, 3), dtype=np.uint8)
    heat[..., 0] = (values * 245).astype(np.uint8)
    heat[..., 1] = (values * 210).astype(np.uint8)
    heat[..., 2] = ((1 - values) * 140 + 30).astype(np.uint8)
    return {'mask': Image.fromarray(mask.astype(np.uint8) * 255),
            'overlay': Image.fromarray(rgb.astype(np.uint8)), 'heatmap': Image.fromarray(heat)}


def binary_metrics(predicted, truth):
    a, b = np.asarray(predicted) > 0, np.asarray(truth) > 0
    intersection = int((a & b).sum())
    total, union = int(a.sum() + b.sum()), int((a | b).sum())
    return {'dice': 2 * intersection / total if total else 1.0,
            'iou': intersection / union if union else 1.0}
