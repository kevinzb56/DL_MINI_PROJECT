"""Fixed-budget adaptation; only explicit acquisition IDs enter optimizer updates."""
import hashlib,json,os,random,sys,signal
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from src.atomic import write_json
from src.checkpoints import save_checkpoint,load_checkpoint
from src.dataset import CrackDataset
from src.model_factory import build_model
from src.segmentation import region_loss
from src.boundaries import boundary_loss


def support_manifest(manifest,ids):
    lookup={r['image_id']:r for r in manifest['records']}
    if len(set(ids))!=len(ids) or not ids: raise ValueError('Support IDs must be nonempty and unique')
    records=[]
    for identity in ids:
        if identity not in lookup or lookup[identity]['split']!='target_acquisition':
            raise ValueError('Support must belong only to frozen target acquisition pool')
        records.append(lookup[identity])
    return {'records':records}


def adapt(parent_state,parent_hash,manifest,manifest_hash,data_root,ids,settings,seed,run,resume=False,stop_after_steps=None):
    run=Path(run)
    if not resume and run.exists() and any(run.iterdir()): raise ValueError('Existing run requires resume')
    if not settings['freeze_encoder'] or settings['augmentation'] or settings['replay']:
        raise ValueError('This protocol requires frozen encoder, no augmentation and no replay')
    run.mkdir(parents=True,exist_ok=True)
    source_root=Path(__file__).resolve().parents[2]
    code_hash=hashlib.sha256(b''.join((source_root/n).read_bytes() for n in
        ['scripts/experiments/adapt_fewshot.py','src/litecrack.py','src/model_factory.py','src/dataset.py','src/segmentation.py','src/boundaries.py','src/checkpoints.py'])).hexdigest()
    config={'settings':settings,'support_ids':ids,'parent_sha256':parent_hash,'seed':seed,
            'code_sha256':code_hash,'torch':torch.__version__,'cuda':torch.version.cuda,
            'model_config':parent_state['config']}
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=torch.device(settings['device'])
    if device.type=='cuda': torch.cuda.manual_seed_all(seed)
    model=build_model(parent_state['config']).to(device)
    model.load_state_dict(parent_state['model'])
    for parameter in model.encoder.parameters(): parameter.requires_grad=False
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=settings['learning_rate'])
    dataset=CrackDataset(support_manifest(manifest,ids),data_root,'target_acquisition',parent_state['config']['image_size'])
    epoch=step=cursor=0;order=None;history=[]
    if resume:
        state=load_checkpoint(run/'checkpoints',model=model,optimizer=optimizer,scheduler=None,scaler=None,
            expected_manifest_hash=manifest_hash,expected_config=config)
        epoch,step=state['epoch'],state['global_step']
        order,cursor,history=(state['sampler_state'][key] for key in ('order','cursor','history'))
    else: write_json(run/'config.json',config)
    def snapshot():
        save_checkpoint(run/'checkpoints',model=model,optimizer=optimizer,scheduler=None,scaler=None,
            epoch=epoch,global_step=step,sampler_state={'order':order,'cursor':cursor,'history':history},
            config=config,manifest_hash=manifest_hash,metrics=history[-1] if history else {},best_metric=None)
    stopping=[False]
    def stop(*_): stopping[0]=True
    previous={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        while epoch<settings['epochs']:
            if order is None: order=torch.randperm(len(dataset)).tolist();cursor=0
            model.train();model.encoder.eval()
            while cursor<len(order):
                selected=order[cursor:cursor+settings['batch_size']]
                samples=[dataset[i] for i in selected]
                x=torch.stack([s[0] for s in samples]).to(device); y=torch.stack([s[1] for s in samples]).to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction=model(x)
                logits=prediction[0] if isinstance(prediction,tuple) else prediction
                loss=region_loss(logits,y,'bce_dice')
                if isinstance(prediction,tuple): loss=loss+parent_state['config'].get('boundary_weight',.2)*boundary_loss(prediction[1],y)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite adaptation loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip_norm'],error_if_nonfinite=True)
                optimizer.step();cursor+=len(selected);step+=1
                history.append({'step':step,'epoch':epoch,'support_loss':float(loss.detach())})
                snapshot()
                if stopping[0] or (stop_after_steps is not None and step>=stop_after_steps):
                    write_json(run/'status.json',{'state':'stopped_safely','step':step,'epoch':epoch});return None
            epoch+=1;order=None;cursor=0;snapshot()
        # Exact comparison enforces that the encoder really stayed frozen.
        for key,value in model.state_dict().items():
            if key.startswith('encoder.') and not torch.equal(value.cpu(),parent_state['model'][key].cpu()):
                raise ValueError('Encoder changed during decoder-only adaptation')
        write_json(run/'history.json',history)
        write_json(run/'status.json',{'state':'adaptation_complete','epoch':epoch,'step':step,
            'support_images':len(ids),'encoder_unchanged':True,'selection':'final_fixed_budget'})
        return model
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
