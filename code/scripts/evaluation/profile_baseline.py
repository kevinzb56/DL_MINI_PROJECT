"""Profile validation-best weights; does not use query labels or change training."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from src.atomic import write_json
from src.model_factory import build_model


def conv_linear_macs(model,image):
    total=[0]
    handles=[]
    def count(module,inputs,output):
        if isinstance(module,torch.nn.Conv2d):
            total[0]+=output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
        elif isinstance(module,torch.nn.Linear):
            total[0]+=output.numel()*module.in_features
    for module in model.modules():
        if isinstance(module,(torch.nn.Conv2d,torch.nn.Linear)):
            handles.append(module.register_forward_hook(count))
    try:
        with torch.inference_mode(): model(image)
    finally:
        for handle in handles: handle.remove()
    return total[0]


def profile(run_dir,device_name):
    run=Path(run_dir)
    pointer=json.loads((run/'checkpoints/best.json').read_text())
    checkpoint=run/'checkpoints'/pointer['file']
    assert checkpoint.resolve().parent==(run/'checkpoints').resolve()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==pointer['sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=state['config']
    device=torch.device(device_name)
    torch.set_num_threads(4)
    model=build_model(config).to(device).eval()
    model.load_state_dict(state['model'])
    image=torch.zeros(1,3,config['image_size'],config['image_size'],device=device)
    def sync():
        if device.type=='cuda': torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(20): model(image)
        sync()
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        samples=[]
        for _ in range(200):
            sync(); start=time.perf_counter(); model(image); sync()
            samples.append((time.perf_counter()-start)*1000)
    report={'architecture':config['architecture'],'device':str(device),
            'hardware':torch.cuda.get_device_name(device) if device.type=='cuda' else __import__('platform').processor(),
            'threads':torch.get_num_threads(),'batch_size':1,'image_size':config['image_size'],
            'parameters':sum(p.numel() for p in model.parameters()),
            'p50_ms':statistics.median(samples),'p95_ms':sorted(samples)[189],
            'warmups':20,'timed_iterations':200,'scope':'model pass only; no image decoding/transfer',
            'recovery_checkpoint_bytes':checkpoint.stat().st_size,
            'fp32_weight_bytes':sum(p.numel()*p.element_size() for p in model.state_dict().values()),
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
            'conv_linear_macs':conv_linear_macs(model,image),
            'mac_convention':'One MAC per multiply-accumulate; Conv2d/Linear only; excludes normalization, activation, pooling, resize and elementwise operations'}
    write_json(run/f'profile_{device.type}.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],required=True)
    a=p.parse_args(); profile(a.run_dir,a.device)
