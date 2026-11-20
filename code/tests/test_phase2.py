import hashlib
import json
import os
from pathlib import Path
import pytest

torch=pytest.importorskip('torch')
np=pytest.importorskip('numpy')
Image=pytest.importorskip('PIL.Image')
from src.models import SegmentationUNet
from src.segmentation import image_metrics,region_loss
from scripts.experiments.train_baseline import train


def test_models_preserve_spatial_dimensions():
    torch.set_num_threads(1)
    for architecture in ('unet','r2au'):
        model=SegmentationUNet(architecture,2)
        assert model(torch.zeros(2,3,35,37)).shape==(2,1,35,37)


def test_empty_and_disjoint_metric_conventions():
    logits=torch.full((1,1,16,16),-100.)
    truth=torch.zeros_like(logits)
    assert image_metrics(logits,truth)['dice'].item()==1
    truth[:,:,2,2]=1
    assert image_metrics(logits,truth)['dice'].item()==0
    assert image_metrics(logits,truth)['iou'].item()==0


def test_bce_dice_recovers_gradient_for_saturated_wrong_prediction():
    logits=torch.full((1,1,16,16),-100.,requires_grad=True)
    truth=torch.ones_like(logits)
    region_loss(logits,truth,'bce_dice').backward()
    assert torch.isfinite(logits.grad).all() and (logits.grad<0).all()


def fixture_manifest(root):
    rng=np.random.default_rng(123)
    records=[]
    splits=['source_train']*4+['source_validation']*2+['source_holdout']*2+['target_query']*2
    for i,split in enumerate(splits):
        image=rng.integers(0,256,(32,32,3),dtype=np.uint8)
        mask=np.zeros((32,32),dtype=np.uint8); mask[:,i+2:i+4]=255
        ip=root/f'{i}.png'; mp=root/f'{i}_mask.png'
        Image.fromarray(image).save(ip); Image.fromarray(mask).save(mp)
        records.append({'image_id':str(i),'image':ip.name,'mask':mp.name,'split':split,
                        'image_sha256':hashlib.sha256(ip.read_bytes()).hexdigest(),
                        'mask_sha256':hashlib.sha256(mp.read_bytes()).hexdigest()})
    path=root/'manifest.json'
    path.write_text(json.dumps({'status':'FROZEN_PILOT','records':records}))
    return path


@pytest.mark.parametrize('architecture',['unet','r2au','litecrack','litecrack_ba'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_actual_trainer_mid_epoch_resume(tmp_path,architecture,device):
    if architecture.startswith('litecrack'): pytest.importorskip('torchvision')
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    manifest=fixture_manifest(tmp_path)
    config={'architecture':architecture,'base_channels':2,'normalization':'group','seed':42,'image_size':32,
            'batch_size':2,'epochs':2,'learning_rate':.001,'patience':10,
            'checkpoint_every_steps':1,'device':device,'loss':'bce_dice','gradient_clip_norm':1.,'deterministic':True}
    uninterrupted=tmp_path/'full'; recovered=tmp_path/'resumed'
    train(config,manifest,tmp_path,uninterrupted,final_evaluation=False)
    train(config,manifest,tmp_path,recovered,stop_after_steps=1,final_evaluation=False)
    assert json.loads((recovered/'status.json').read_text())['state']=='stopped_safely'
    train(config,manifest,tmp_path,recovered,resume=True,final_evaluation=False)
    def latest(run):
        pointer=json.loads((run/'checkpoints/latest.json').read_text())
        return torch.load(run/'checkpoints'/pointer['file'],map_location='cpu',weights_only=False)
    a,b=latest(uninterrupted),latest(recovered)
    assert a['global_step']==b['global_step']==4
    assert a['sampler_state']==b['sampler_state']
    assert a['scheduler']==b['scheduler']
    for key in a['model']: assert torch.equal(a['model'][key],b['model'][key]),key
    assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
    for key in a['optimizer']['state']:
        for field in a['optimizer']['state'][key]:
            assert torch.equal(a['optimizer']['state'][key][field],b['optimizer']['state'][key][field])
    changed={**config,'learning_rate':.1}
    with pytest.raises(ValueError,match='Configuration changed'):
        train(changed,manifest,tmp_path,recovered,resume=True,final_evaluation=False)
