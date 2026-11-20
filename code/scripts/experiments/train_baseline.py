"""Deterministic single-device baseline training, resumable after optimizer steps."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import shutil
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from src.atomic import write_json
from src.checkpoints import save_checkpoint,load_checkpoint
from src.dataset import CrackDataset,verify_files
from src.model_factory import build_model
from src.segmentation import dice_loss,image_metrics,region_loss
from src.boundaries import boundary_loss,boundary_f1


def source_hash():
    root=Path(__file__).resolve().parents[2]
    names=['src/models.py','src/litecrack.py','src/boundaries.py','src/model_factory.py','src/dataset.py','src/segmentation.py','src/checkpoints.py','scripts/experiments/train_baseline.py']
    if (root/'src/archive_io.py').exists(): names.append('src/archive_io.py')
    return hashlib.sha256(b''.join((root/n).read_bytes() for n in names)).hexdigest()


@torch.no_grad()
def evaluate(model,dataset,device,batch_size):
    model.eval(); totals={key:0. for key in ('dice','iou','precision','recall','boundary_f1')}
    per_image=[]; loss_sum=0; predicted_fraction=0.; truth_fraction=0.
    for start in range(0,len(dataset),batch_size):
        indices=list(range(start,min(start+batch_size,len(dataset))))
        samples=[dataset[i] for i in indices]
        x=torch.stack([s[0] for s in samples]).to(device)
        y=torch.stack([s[1] for s in samples]).to(device)
        logits=model(x)
        if isinstance(logits,tuple): logits=logits[0]
        predicted_fraction+=float((logits>=0).float().mean())*len(indices)
        truth_fraction+=float(y.mean())*len(indices)
        loss_sum+=float(dice_loss(logits,y))*len(indices)
        values=image_metrics(logits,y)
        values['boundary_f1']=boundary_f1(logits,y)
        for j,index in enumerate(indices):
            record={'image_id':dataset.rows[index]['image_id']}
            for key,vector in values.items():
                value=float(vector[j]); totals[key]+=value; record[key]=value
            per_image.append(record)
    return {'loss':loss_sum/len(dataset),**{k:v/len(dataset) for k,v in totals.items()},
            'predicted_foreground_fraction':predicted_fraction/len(dataset),
            'truth_foreground_fraction':truth_fraction/len(dataset),'n':len(dataset)},per_image


def train(config,manifest_path,data_root,run_dir,resume=False,stop_after_steps=None,final_evaluation=True):
    run=Path(run_dir)
    if not resume and run.exists() and any(run.iterdir()):
        raise ValueError('Run directory is not empty; use --resume or a new run')
    run.mkdir(parents=True,exist_ok=True)
    manifest_path=Path(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('status') not in ('FROZEN_PILOT','FROZEN_MILANO_V1','FROZEN_MULTISOURCE_ARCHIVE1_PILOT','FROZEN_MULTISOURCE_V1'):
        raise ValueError('Expected frozen training protocol')
    if manifest.get('status')=='FROZEN_MULTISOURCE_V1':
        from collections import Counter
        if config.get('protocol')!='multisource_v1_seed42' or final_evaluation:
            raise ValueError('Full multisource protocol requires explicit config and deferred test evaluation')
        if dict(Counter(r['publisher_split'] for r in manifest['records']))!={'train':2526,'val':542}:
            raise ValueError('Full multisource train/validation coverage mismatch')
        if any(r['split']!={'train':'source_train','val':'source_validation'}.get(r['publisher_split']) for r in manifest['records']):
            raise ValueError('Invalid publisher split mapping')
    if manifest.get('status')=='FROZEN_MILANO_V1' and config.get('protocol')!='milano_v1_publisher_splits':
        raise ValueError('Milano manifest requires its explicit training protocol')
    if manifest.get('status')=='FROZEN_MULTISOURCE_ARCHIVE1_PILOT':
        if config.get('protocol')!='multisource_archive1_pilot' or final_evaluation:
            raise ValueError('Archive-1 pilot requires explicit protocol and no test evaluation')
        mapping={'train':'source_train','val':'source_validation','test':'source_holdout'}
        if any(r.get('split')!=mapping.get(r.get('publisher_split')) or r.get('archive')!='01_ground_gopro.zip' for r in manifest['records']):
            raise ValueError('Invalid archive-1 pilot split mapping')
    manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    config={**config,'source_sha256':source_hash(),'torch_version':torch.__version__,
            'cuda_runtime':torch.version.cuda,
            'torchvision_version':importlib.metadata.version('torchvision') if config['architecture'].startswith('litecrack') else None}
    if config.get('loss') not in ('dice','bce_dice'): raise ValueError('Unknown loss')
    if config['batch_size']<1 or config['checkpoint_every_steps']<1 or config['image_size']<16:
        raise ValueError('Invalid batch/checkpoint frequency/image size')
    device=torch.device(config['device'])
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if config['deterministic']:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    if device.type=='cuda': torch.cuda.manual_seed_all(config['seed'])
    # Verify all hashes once before training; target data is never used for updates.
    verify_files(manifest['records'],data_root)
    training=CrackDataset(manifest,data_root,'source_train',config['image_size'],True)
    validation=CrackDataset(manifest,data_root,'source_validation',config['image_size'])
    model=build_model(config).to(device)
    if config.get('max_parameters') and sum(p.numel() for p in model.parameters())>config['max_parameters']:
        raise ValueError('Compact-model parameter budget exceeded')
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5,factor=.1)
    epoch=step=cursor=bad_epochs=0; order=None; best=-1.; history=[]
    loss_total=0.; seen=0
    if resume:
        state=load_checkpoint(run/'checkpoints',model=model,optimizer=optimizer,scheduler=scheduler,
            scaler=None,expected_manifest_hash=manifest_hash,expected_config=config)
        epoch,step,best=state['epoch'],state['global_step'],state['best_metric']
        sampler=state['sampler_state']
        order,cursor,bad_epochs=sampler['order'],sampler['cursor'],sampler['bad_epochs']
        history=sampler['history']; loss_total=sampler['loss_total']; seen=sampler['seen']
    else:
        write_json(run/'config.json',config)
        write_json(run/'manifest.json',manifest)
    write_json(run/'environment.json',{'python':sys.version,'torch':torch.__version__,
        'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        'parameters':sum(p.numel() for p in model.parameters())})
    stopping={'requested':False}
    def request_stop(*_): stopping['requested']=True
    old_handlers={}
    for sig in (signal.SIGINT,signal.SIGTERM):
        old_handlers[sig]=signal.signal(sig,request_stop)
    def snapshot(is_best=False):
        # Do not delete earlier snapshots to make space. Leave recoverable files
        # intact and fail explicitly before a likely out-of-space write.
        estimated=sum(p.numel()*p.element_size() for p in model.parameters())*4
        if shutil.disk_usage(run).free < estimated+256*1024**2:
            raise RuntimeError('Insufficient checkpoint disk space; export results and resume on persistent storage')
        save_checkpoint(run/'checkpoints',model=model,optimizer=optimizer,scheduler=scheduler,
            scaler=None,epoch=epoch,global_step=step,
            sampler_state={'order':order,'cursor':cursor,'bad_epochs':bad_epochs,
                           'history':history,'loss_total':loss_total,'seen':seen},
            config=config,manifest_hash=manifest_hash,metrics=history[-1] if history else {},
            best_metric=best,is_best=is_best)
    started=time.monotonic()
    try:
        while epoch<config['epochs'] and bad_epochs<config['patience']:
            if order is None:
                order=torch.randperm(len(training)).tolist(); cursor=0; loss_total=0.; seen=0
            model.train()
            while cursor<len(order):
                indices=order[cursor:cursor+config['batch_size']]
                samples=[training[i] for i in indices]
                x=torch.stack([s[0] for s in samples]).to(device)
                y=torch.stack([s[1] for s in samples]).to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction=model(x)
                if isinstance(prediction,tuple):
                    loss=region_loss(prediction[0],y,config['loss'])+config.get('boundary_weight',.2)*boundary_loss(prediction[1],y)
                else:
                    loss=region_loss(prediction,y,config['loss'])
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss; inspect run')
                loss.backward()
                if config.get('gradient_clip_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'],error_if_nonfinite=True)
                optimizer.step()
                cursor+=len(indices); step+=1
                loss_total+=float(loss.detach())*len(indices); seen+=len(indices)
                stop=stopping['requested'] or (stop_after_steps is not None and step>=stop_after_steps)
                if step%config['checkpoint_every_steps']==0 or stop: snapshot()
                if stop:
                    write_json(run/'status.json',{'state':'stopped_safely','epoch':epoch,'step':step,
                                                  'elapsed_seconds_this_session':time.monotonic()-started})
                    return
            metrics,_=evaluate(model,validation,device,config['batch_size'])
            scheduler.step(metrics['loss'])
            improved=metrics['dice']>best
            if improved: best=metrics['dice']; bad_epochs=0
            else: bad_epochs+=1
            history.append({'epoch':epoch+1,'step':step,'train_loss':loss_total/seen,
                            'validation':metrics,'learning_rate':optimizer.param_groups[0]['lr']})
            print(json.dumps(history[-1]),flush=True)
            epoch+=1; order=None; cursor=0; loss_total=0.; seen=0
            snapshot(improved)
            # Recovery history is authoritative; replace logs rather than append duplicates.
            write_json(run/'history.json',history)
        write_json(run/'status.json',{'state':'training_complete','epoch':epoch,'step':step,
            'best_validation_dice':best,'elapsed_seconds_this_session':time.monotonic()-started})
        if final_evaluation:
            load_checkpoint(run/'checkpoints',model=model,optimizer=optimizer,scheduler=scheduler,
                scaler=None,expected_manifest_hash=manifest_hash,expected_config=config,which='best')
            final={}
            for split in config.get('evaluation_splits',['source_holdout','target_query']):
                dataset=CrackDataset(manifest,data_root,split,config['image_size'])
                metrics,images=evaluate(model,dataset,device,config['batch_size'])
                final[split]=metrics
                write_json(run/(split+'_per_image.json'),images)
            write_json(run/'evaluation.json',{'checkpoint':'validation_best','threshold':.5,'results':final})
    except Exception as error:
        # Do not save partially executed optimizer steps over valid recovery state.
        write_json(run/'failure.json',{'error':str(error),'epoch':epoch,'last_step':step})
        raise
    finally:
        for sig,handler in old_handlers.items(): signal.signal(sig,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-steps',type=int)
    a=p.parse_args()
    train(json.loads(a.config.read_text()),a.manifest,a.data_root,a.run_dir,a.resume,a.stop_after_steps)
