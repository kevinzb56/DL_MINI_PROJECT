"""FP32 ONNX export with mandatory native-runtime parity checks."""
import hashlib
import io
import json
import time
import zipfile
import numpy as np
from PIL import Image
from inspection.advanced import CONFIG
from inspection.engine import INFERENCE_LOCK, binary_metrics, load_state, model_config, tensor_image
from inspection.replay import verified_buffer
from inspection.store import identity,now
from src.atomic import write_json


def export_model(store,project_id,version):
    import torch
    import onnx
    import onnxruntime as ort
    from src.model_factory import build_model
    started=time.perf_counter()
    export_id=identity();folder=store.folder('onnx_exports',export_id)
    samples=[]
    for image in sorted(store.list('image',project_id),key=lambda i:i['id'])[:3]:
        with Image.open(store.root/'images'/image['id']/'original.png') as original:
            samples.append((image['id'],original.convert('RGB'),None))
    audited=False
    try:
        replay_root,buffer=verified_buffer(store)
        for row in buffer['records']:
            if row['split']=='source_validation':
                with Image.open(replay_root/row['image']) as original,Image.open(replay_root/row['mask']) as truth:
                    samples.append((row['image_id'],original.convert('RGB'),truth.convert('L').copy()))
        audited=True
    except (ValueError,OSError,KeyError):
        pass
    class RegionProbability(torch.nn.Module):
        def __init__(self,model):
            super().__init__();self.model=model
        def forward(self,x):
            result=self.model(x)
            return torch.sigmoid(result[0] if isinstance(result,tuple) else result)
    with INFERENCE_LOCK:
        torch.set_num_threads(2)
        state=load_state(store,version);config=model_config(state);size=config['image_size']
        model=build_model(config).eval();model.load_state_dict(state['model'])
        wrapper=RegionProbability(model).eval()
        graph=folder/'model.onnx'
        torch.onnx.export(wrapper,torch.zeros(1,3,size,size),str(graph),input_names=['image'],output_names=['probability'],
                          opset_version=CONFIG['onnx']['opset'],dynamo=False,external_data=False)
        onnx.checker.check_model(str(graph))
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(graph),sess_options=options,providers=['CPUExecutionProvider'])
        rng=np.random.default_rng(7301)
        samples.extend([('synthetic_black',Image.new('RGB',(size,size),0),None),
                        ('synthetic_white',Image.new('RGB',(size,size),255),None),
                        ('synthetic_noise',Image.fromarray(rng.integers(0,256,(size+17,size+31,3),dtype=np.uint8)),None)])
        checks=[]
        for name,image,truth in samples:
            x=tensor_image(image,size)
            with torch.inference_mode():native=wrapper(x).numpy()
            portable=session.run(['probability'],{'image':x.numpy()})[0]
            if portable.shape!=native.shape or not np.isfinite(portable).all():raise ValueError('ONNX returned invalid output')
            error=float(np.max(np.abs(native-portable)))
            row={'sample':name,'max_probability_difference':error,'mask_agreement':float(np.mean((native>=.5)==(portable>=.5)))}
            if error>CONFIG['onnx']['max_probability_error']:raise ValueError('ONNX probability parity gate failed')
            if truth is not None:
                metrics=[]
                for values in (native,portable):
                    restored=np.asarray(Image.fromarray(values[0,0]).resize(image.size,Image.Resampling.BILINEAR))
                    metrics.append(binary_metrics((restored>=.5).astype(np.uint8)*255,truth))
                row['native_metrics'],row['onnx_metrics']=metrics
                for metric in ('dice','iou'):
                    if abs(metrics[0][metric]-metrics[1][metric])>CONFIG['onnx']['max_metric_difference']:
                        raise ValueError('ONNX Dice/IoU parity gate failed')
            checks.append(row)
    report={'id':export_id,'version_id':version['id'],'checkpoint_sha256':version['sha256'],
            'onnx_sha256':hashlib.sha256(graph.read_bytes()).hexdigest(),'state':'verified','created':now(),
            'input':{'name':'image','shape':[1,3,size,size],'dtype':'float32','normalization':'RGB divided by 255','resize':'bilinear square'},
            'output':{'name':'probability','shape':[1,1,size,size],'dtype':'float32','range':[0,1]},
            'opset':CONFIG['onnx']['opset'],'pytorch':str(torch.__version__),'onnx':onnx.__version__,'onnxruntime':ort.__version__,
            'provider':'CPUExecutionProvider','checks':checks,'audited_source_validation':audited,
            'max_probability_difference':max(c['max_probability_difference'] for c in checks),
            'probability_gate':CONFIG['onnx']['max_probability_error'],'metric_difference_gate':CONFIG['onnx']['max_metric_difference'],
            'metric_gate_scope':'Eight audited source-validation samples' if audited else 'Not assessed: no audited ground-truth samples available',
            'elapsed_seconds':time.perf_counter()-started}
    write_json(folder/'parity.json',report)
    store.put('onnx',report,project_id)
    return report


def export_zip(store,project_id,export_id):
    entry=store.get('onnx',export_id,project_id)
    folder=store.root/'onnx_exports'/export_id
    if hashlib.sha256((folder/'model.onnx').read_bytes()).hexdigest()!=entry['onnx_sha256']:
        raise ValueError('Export graph changed')
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(folder/'model.onnx','model.onnx');z.write(folder/'parity.json','parity.json')
        z.writestr('README.txt','Input: FP32 RGB/255 NCHW, batch 1, fixed spatial dimensions from parity.json.\nOutput: crack probability, threshold 0.50. Resize probability bilinearly to original size.\nNo claim of universal road-scene accuracy follows from runtime parity.\n')
    return stream.getvalue()
