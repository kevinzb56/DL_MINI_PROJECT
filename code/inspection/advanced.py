"""Phase 7 inference and acquisition, isolated from the Phase 6 recovery code."""
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from inspection.engine import INFERENCE_LOCK, load_state, model_config, probability, tensor_image

CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'configs/phase7.json').read_text())


def entropy(values):
    p = np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1-1e-7)
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))


def tiled_probability(predict, image, tile_size=256, overlap=64, max_tiles=256):
    if not 0 <= overlap < tile_size or tile_size < 8:
        raise ValueError('Invalid tile size or overlap')
    width, height = image.size
    stride = tile_size - overlap
    columns = max(1, math.ceil((width-tile_size)/stride)+1)
    rows = max(1, math.ceil((height-tile_size)/stride)+1)
    if columns*rows > max_tiles:
        raise ValueError(f'This image needs {columns*rows} tiles; the local limit is {max_tiles}')
    padded_w, padded_h = (columns-1)*stride+tile_size, (rows-1)*stride+tile_size
    pixels = np.pad(np.asarray(image), ((0,padded_h-height),(0,padded_w-width),(0,0)), mode='edge')
    total = np.zeros((padded_h,padded_w), dtype=np.float32)
    weights = np.zeros_like(total)
    window = np.outer(np.maximum(np.hanning(tile_size), .05), np.maximum(np.hanning(tile_size), .05)).astype(np.float32)
    for row in range(rows):
        for column in range(columns):
            x,y = column*stride,row*stride
            tile = Image.fromarray(pixels[y:y+tile_size,x:x+tile_size])
            values = np.asarray(predict(tile), dtype=np.float32)
            if values.shape != (tile_size,tile_size) or not np.isfinite(values).all():
                raise ValueError('Invalid tile output')
            total[y:y+tile_size,x:x+tile_size] += values*window
            weights[y:y+tile_size,x:x+tile_size] += window
    result = (total/weights)[:height,:width].copy()
    return result, {'mode':'tiled', 'tile_size':tile_size, 'overlap':overlap, 'stride':stride,
                    'tiles':rows*columns, 'padding':'edge', 'pad_right':padded_w-width,
                    'pad_bottom':padded_h-height, 'blend':'hann_floor_0.05', 'global_resize':False}


def checked_export(store, project_id, version):
    exports = [e for e in store.list('onnx',project_id) if e['version_id']==version['id'] and e['state']=='verified']
    if not exports:
        raise ValueError('Export and validate this model before selecting ONNX Runtime')
    entry = exports[-1]
    path = store.root / 'onnx_exports' / entry['id'] / 'model.onnx'
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['onnx_sha256'] or entry['checkpoint_sha256'] != version['sha256']:
        raise ValueError('ONNX export checksum mismatch')
    return path, entry


def predictor(store, version, runtime='pytorch', project_id=None):
    import torch
    torch.set_num_threads(2)
    state = load_state(store, version)
    config = model_config(state)
    if runtime == 'pytorch':
        from src.model_factory import build_model
        model = build_model(config).eval()
        model.load_state_dict(state['model'])
        return lambda image: probability(model,image,config['image_size']), config
    if runtime != 'onnx':
        raise ValueError('Unknown inference runtime')
    import onnxruntime as ort
    path, _ = checked_export(store,project_id,version)
    options = ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
    session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    def predict(image):
        small = session.run(['probability'],{'image':tensor_image(image,config['image_size']).numpy()})[0][0,0]
        return np.asarray(Image.fromarray(small).resize(image.size,Image.Resampling.BILINEAR)).copy()
    return predict, config


def infer_advanced(store,version,image,mode='resize',runtime='pytorch',project_id=None):
    started=time.perf_counter()
    with INFERENCE_LOCK:
        predict,config=predictor(store,version,runtime,project_id)
        if mode=='resize':
            values=predict(image);details={'mode':'resize','global_resize':True}
        elif mode=='tiled':
            values,details=tiled_probability(predict,image,config['image_size'],min(CONFIG['tiling']['overlap'],config['image_size']//4),CONFIG['tiling']['max_tiles'])
        else:
            raise ValueError('Unknown inference mode')
    return values,config,round((time.perf_counter()-started)*1000,1),dict(details,runtime=runtime)


def rank_images(store,project_id,version,policy='entropy',seed=7301):
    if policy not in ('entropy','random'):
        raise ValueError('Unknown ranking policy')
    corrections=store.list('correction',project_id)
    complete={c['image_id'] for c in corrections if c['complete']}
    # Respect the latest revision: a later draft makes the image eligible again.
    latest={c['image_id']:c for c in corrections}
    complete={iid for iid,c in latest.items() if c['complete']}
    reserved={iid for j in store.list('job',project_id) for iid in j['validation_ids']}
    pool=sorted((i for i in store.list('image',project_id)
                 if i['training_eligible'] and i['id'] not in complete|reserved
                 and all(s=='target_acquisition' for s in i.get('frozen_splits',[]))),key=lambda i:i['id'])
    if len(pool)>CONFIG['ranking']['max_pool']:
        raise ValueError('Ranking is limited to 100 eligible images per project')
    rows=[]
    if policy=='random':
        random.Random(seed).shuffle(pool)
        rows=[{'image_id':i['id'],'name':i['name'],'score':None} for i in pool]
    elif pool:
        with INFERENCE_LOCK:
            predict,_=predictor(store,version)
            for image in pool:
                with Image.open(store.root/'images'/image['id']/'original.png') as source:
                    score=float(entropy(predict(source.convert('RGB'))).mean())
                rows.append({'image_id':image['id'],'name':image['name'],'score':score})
        rows.sort(key=lambda r:(-r['score'],r['image_id']))
    return {'policy':policy,'seed':seed,'version_id':version['id'],'checkpoint_sha256':version['sha256'],
            'pool_size':len(pool),'ranked':rows,'scope':'Eligible incomplete/unlabeled project images only; validation images excluded.',
            'score_definition':'Mean binary predictive entropy (bits), original-size bilinear probabilities; not calibrated confidence.',
            'warning':'Confident all-background predictions can still be wrong. Ranking is a suggestion; reviewer choice remains manual.'}
