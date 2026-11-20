import json,hashlib
import pytest
torch=pytest.importorskip('torch')
np=pytest.importorskip('numpy')
Image=pytest.importorskip('PIL.Image')
pytest.importorskip('torchvision')
from src.model_factory import build_model
from scripts.experiments.adapt_fewshot import adapt,support_manifest

def test_query_cannot_be_used_as_support():
    with pytest.raises(ValueError,match='acquisition'):
        support_manifest({'records':[{'image_id':'query','split':'target_query'}]},['query'])

@pytest.mark.parametrize('architecture',['litecrack','litecrack_ba'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_adaptation_resume_and_frozen_encoder(tmp_path,architecture,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1);torch.manual_seed(42)
    config={'architecture':architecture,'encoder_width_mult':.5,'image_size':32,'boundary_weight':.2}
    parent={'config':config,'model':build_model(config).state_dict()}
    rows=[]
    rng=np.random.default_rng(12)
    for i in range(3):
        image=rng.integers(0,256,(32,32,3),dtype=np.uint8)
        mask=np.zeros((32,32),dtype=np.uint8);mask[:,5+i:8+i]=255
        Image.fromarray(image).save(tmp_path/f'{i}.png');Image.fromarray(mask).save(tmp_path/f'{i}_mask.png')
        rows.append({'image_id':str(i),'split':'target_acquisition','image':f'{i}.png','mask':f'{i}_mask.png'})
    manifest={'records':rows};ids=[r['image_id'] for r in rows]
    settings={'freeze_encoder':True,'augmentation':False,'replay':False,'device':device,
              'learning_rate':.0001,'epochs':2,'batch_size':2,'gradient_clip_norm':1.}
    full=adapt(parent,'fixture',manifest,'fixture',tmp_path,ids,settings,42,tmp_path/'full')
    assert adapt(parent,'fixture',manifest,'fixture',tmp_path,ids,settings,42,tmp_path/'resumed',stop_after_steps=1) is None
    resumed=adapt(parent,'fixture',manifest,'fixture',tmp_path,ids,settings,42,tmp_path/'resumed',resume=True)
    for key,value in full.state_dict().items():
        assert torch.equal(value,resumed.state_dict()[key])
        if key.startswith('encoder.'):assert torch.equal(value.cpu(),parent['model'][key].cpu())
    assert any(not torch.equal(value.cpu(),parent['model'][key].cpu()) for key,value in full.state_dict().items() if not key.startswith('encoder.'))
