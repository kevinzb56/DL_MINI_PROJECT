"""Separate train/validation diagnostic; immutable parent pilot, no test scoring."""
import hashlib
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from src.atomic import write_json
from src.checkpoints import load_checkpoint, save_checkpoint
from src.model_factory import build_model
from src.segmentation import region_loss, image_metrics, dice_loss
from scripts.experiments.train_baseline import source_hash


def sha(data): return hashlib.sha256(data).hexdigest()


def export():
    # Called only between optimizer steps; all checkpoint pointers are stable.
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path=ROOT.parent/f'multisource_diagnostic_recovery_{stamp}.zip'
    temporary=path.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ('src','scripts','artifacts','parent','cache','witness'):
            for item in sorted((ROOT/folder).rglob('*')):
                if item.is_file() and '__pycache__' not in item.parts:
                    archive.write(item,item.relative_to(ROOT).as_posix())
    temporary.replace(path)
    path.with_suffix('.zip.sha256').write_text(sha(path.read_bytes())+'  '+path.name+'\n')
    print('Export:',path.name,flush=True)


class Cache:
    def __init__(self):
        self.index=json.loads((ROOT/'cache/index.json').read_text())
        self.rows=self.index['rows']
        assert not self.index['test_samples_included']
        assert len(self.rows)==476 and len({r['image_id'] for r in self.rows})==476
        mapping={'train':'source_train','val':'source_validation'}
        assert all(r['publisher_split'] in mapping and r['split']==mapping[r['publisher_split']] for r in self.rows)
        parent_bytes=(ROOT/'parent/manifest.json').read_bytes()
        assert sha(parent_bytes)==self.index['parent_manifest_sha256']
        expected=[r for r in json.loads(parent_bytes)['records'] if r['publisher_split'] in mapping]
        assert self.rows==expected
        data=(ROOT/'cache/samples.npz').read_bytes()
        assert sha(data)==self.index['samples_sha256']
        with np.load(io.BytesIO(data),allow_pickle=False) as arrays:
            raw_x=arrays['images']; raw_y=arrays['masks']
        assert raw_x.shape==(476,256,256,3) and raw_y.shape==(476,256,256)
        assert raw_x.dtype==raw_y.dtype==np.uint8 and set(np.unique(raw_y).tolist())<={0,255}
        for witness in self.index['witnesses']:
            data=(ROOT/witness['member']).read_bytes()
            assert sha(data)==witness['sha256']
            with Image.open(io.BytesIO(data)) as image:
                if witness['field']=='image':
                    actual=np.asarray(image.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
                    expected=raw_x[witness['index']]
                else:
                    actual=np.asarray(image.convert('L').resize((256,256),Image.Resampling.NEAREST))
                    expected=raw_y[witness['index']]
                assert np.array_equal(actual,expected), 'Cache differs from raw decoding'
        self.x=torch.from_numpy(raw_x.astype(np.float32).transpose(0,3,1,2)/255)
        self.y=torch.from_numpy((raw_y>0).astype(np.float32)[:,None])
        self.train=[i for i,r in enumerate(self.rows) if r['publisher_split']=='train']
        self.val=[i for i,r in enumerate(self.rows) if r['publisher_split']=='val']
        assert len(self.train)==390 and len(self.val)==86
        self.hash=sha((ROOT/'cache/index.json').read_bytes())

    def batch(self,indices,augment=False):
        x=self.x[indices].clone(); y=self.y[indices].clone()
        if augment:
            for i in range(len(indices)):
                if random.random()<.5: x[i]=x[i].flip(-1); y[i]=y[i].flip(-1)
                if random.random()<.5: x[i]=x[i].flip(-2); y[i]=y[i].flip(-2)
        return x.cuda(),y.cuda()


@torch.no_grad()
def evaluate(model,cache,indices):
    model.eval()
    totals={k:0. for k in ('dice','iou','precision','recall','bce_dice','soft_dice_loss','predicted_fraction','truth_fraction')}
    sweep={str(t):0. for t in (.1,.2,.3,.4,.5,.6,.7,.8,.9)}
    probability_sums=[0.,0.]; pixel_counts=[0,0]
    for start in range(0,len(indices),8):
        chosen=indices[start:start+8]; x,y=cache.batch(chosen); logits=model(x)
        values=image_metrics(logits,y)
        for key,vector in values.items(): totals[key]+=float(vector.sum())
        totals['bce_dice']+=float(region_loss(logits,y,'bce_dice'))*len(chosen)
        totals['soft_dice_loss']+=float(dice_loss(logits,y))*len(chosen)
        totals['predicted_fraction']+=float((logits>=0).float().mean())*len(chosen)
        totals['truth_fraction']+=float(y.mean())*len(chosen)
        probability=logits.sigmoid(); truth=y.bool()
        for label,mask in enumerate((~truth,truth)):
            probability_sums[label]+=float(probability[mask].sum())
            pixel_counts[label]+=int(mask.sum())
        for threshold in sweep:
            p=probability>=float(threshold)
            tp=(p&truth).sum((1,2,3)).float(); den=p.sum((1,2,3))+truth.sum((1,2,3))
            sweep[threshold]+=float(torch.where(den==0,1,2*tp/den.clamp_min(1)).sum())
    return dict(n=len(indices),**{k:v/len(indices) for k,v in totals.items()},
                diagnostic_threshold_dice={k:v/len(indices) for k,v in sweep.items()},
                mean_probability_background=probability_sums[0]/max(1,pixel_counts[0]),
                mean_probability_crack=probability_sums[1]/max(1,pixel_counts[1]))


def new_model(config): return build_model(config).cuda()


def parent_restore(model,config,which='latest'):
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5,factor=.1)
    state=load_checkpoint(ROOT/'parent/checkpoints',model=model,optimizer=optimizer,scheduler=scheduler,
                          scaler=None,expected_manifest_hash=sha((ROOT/'parent/manifest.json').read_bytes()),
                          expected_config=config,which=which)
    return optimizer,scheduler,state


def snapshot(folder,model,optimizer,scheduler,epoch,step,history,config,cache,best,is_best=False):
    save_checkpoint(folder/'checkpoints',model=model,optimizer=optimizer,scheduler=scheduler,
                    scaler=None,epoch=epoch,global_step=step,sampler_state={'history':history},
                    config=config,manifest_hash=cache.hash,metrics=history[-1],best_metric=best,is_best=is_best)
    write_json(folder/'history.json',history)


def main():
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    artifacts=ROOT/'artifacts'; artifacts.mkdir(exist_ok=True)
    state_path=artifacts/'pipeline_status.json'
    write_json(state_path,dict(state='checking_cache_and_parent'))
    cache=Cache()
    base=json.loads((ROOT/'parent/config.json').read_text())
    assert base['source_sha256']==source_hash(), 'Parent model/data implementation changed'
    assert base['torch_version']==torch.__version__ and base['cuda_runtime']==torch.version.cuda
    parent_latest=json.loads((ROOT/'parent/checkpoints/latest.json').read_text())
    config=dict(protocol='multisource_archive1_diagnostic_v1',parent_checkpoint=parent_latest,
                parent_config=base,runner_sha256=sha(Path(__file__).read_bytes()),
                cache_index_sha256=cache.hash,continuation_total_epochs=20,
                memorization_steps=200,memorization_learning_rate=.001,
                reporting_threshold=.5,test_evaluation=False)
    config_path=artifacts/'config.json'
    if config_path.exists(): assert json.loads(config_path.read_text())==config
    else: write_json(config_path,config)
    # Replicate previous full-validation metrics before changing weights.
    if not (artifacts/'baseline.json').exists():
        model=new_model(base); parent_restore(model,base)
        baseline=dict(train=evaluate(model,cache,cache.train),validation=evaluate(model,cache,cache.val))
        old=json.loads((ROOT/'parent/history.json').read_text())[-1]['validation']
        assert abs(baseline['validation']['dice']-old['dice'])<1e-6
        assert abs(baseline['validation']['soft_dice_loss']-old['loss'])<1e-6
        write_json(artifacts/'baseline.json',baseline)
        print('Parent replication:',json.dumps(baseline),flush=True)
        del model
    # A fixed training-only memorization test, separate from the continuation.
    write_json(state_path,dict(state='memorization_check',test_evaluation=False))
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    tiny=[cache.train[i] for i in np.linspace(0,len(cache.train)-1,8,dtype=int)]
    model=new_model(base); optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    folder=artifacts/'memorization'; folder.mkdir(exist_ok=True)
    tiny_config={**config,'experiment':'memorization'}
    start=0; history=[]; best=-1.
    if (folder/'checkpoints/latest.json').exists():
        restored=load_checkpoint(folder/'checkpoints',model=model,optimizer=optimizer,scheduler=None,
                                 scaler=None,expected_manifest_hash=cache.hash,expected_config=tiny_config)
        start=restored['global_step']; history=restored['sampler_state']['history']; best=restored['best_metric']
    elif not (folder/'initial.json').exists():
        write_json(folder/'initial.json',dict(image_ids=[cache.rows[i]['image_id'] for i in tiny],metrics=evaluate(model,cache,tiny)))
    x,y=cache.batch(tiny)
    for step in range(start+1,201):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss=region_loss(model(x),y,'bce_dice'); assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
        if step%50==0:
            metrics=evaluate(model,cache,tiny); history.append(dict(step=step,train=metrics))
            improved=metrics['dice']>best; best=max(best,metrics['dice'])
            snapshot(folder,model,optimizer,None,0,step,history,tiny_config,cache,best,improved)
            print('Memorization:',json.dumps(history[-1]),flush=True)
    export()
    del model,optimizer,x,y
    # Branch from pilot step 147 with original optimizer/scheduler/RNG restored.
    write_json(state_path,dict(state='continuation',target_total_epochs=20,test_evaluation=False))
    model=new_model(base); optimizer,scheduler,parent=parent_restore(model,base)
    folder=artifacts/'continuation'; folder.mkdir(exist_ok=True)
    continuation_config={**config,'experiment':'continuation'}
    start=parent['epoch']; step=parent['global_step']; history=[]; best=-1.
    if (folder/'checkpoints/latest.json').exists():
        restored=load_checkpoint(folder/'checkpoints',model=model,optimizer=optimizer,scheduler=scheduler,
                                 scaler=None,expected_manifest_hash=cache.hash,expected_config=continuation_config)
        start=restored['epoch']; step=restored['global_step']; history=restored['sampler_state']['history']; best=restored['best_metric']
    for epoch in range(start,20):
        order=torch.randperm(len(cache.train)).tolist(); loss_sum=0.; seen=0
        model.train()
        for cursor in range(0,len(order),8):
            indices=[cache.train[i] for i in order[cursor:cursor+8]]
            x,y=cache.batch(indices,True); optimizer.zero_grad(set_to_none=True)
            loss=region_loss(model(x),y,'bce_dice'); assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
            step+=1; loss_sum+=float(loss.detach())*len(indices); seen+=len(indices)
        validation=evaluate(model,cache,cache.val)
        scheduler.step(validation['soft_dice_loss'])
        row=dict(epoch=epoch+1,step=step,augmented_training_loss=loss_sum/seen,
                 train=evaluate(model,cache,cache.train),validation=validation,learning_rate=optimizer.param_groups[0]['lr'])
        history.append(row); improved=validation['dice']>best; best=max(best,validation['dice'])
        snapshot(folder,model,optimizer,scheduler,epoch+1,step,history,continuation_config,cache,best,improved)
        write_json(state_path,dict(state='continuation',epoch=epoch+1,step=step,best_validation_dice=best,test_evaluation=False))
        print('Epoch:',json.dumps(row),flush=True)
        if (epoch+1)%3==0: export()
    write_json(state_path,dict(state='diagnostic_complete',epoch=20,step=step,best_validation_dice=best,test_evaluation=False))
    export()


if __name__=='__main__':
    try: main()
    except Exception as exc:
        write_json(ROOT/'artifacts/failure.json',dict(error_type=type(exc).__name__,error=str(exc)))
        print('DIAGNOSTIC FAILED:',type(exc).__name__,str(exc),flush=True)
        raise
