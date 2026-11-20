import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('torchvision')
from src.litecrack import LiteCrack
from scripts.evaluation.profile_baseline import conv_linear_macs


def test_grouped_convolution_mac_convention():
    model=torch.nn.Conv2d(4,6,3,padding=1,groups=2)
    image=torch.zeros(1,4,8,8)
    assert conv_linear_macs(model,image)==6*8*8*2*3*3
    assert not model._forward_hooks


def test_compact_parameter_budget_and_shapes():
    torch.set_num_threads(1)
    model=LiteCrack(.5)
    assert sum(p.numel() for p in model.parameters())<=1055014
    assert not any(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules())
    for size in ((32,32),(35,37),(256,256)):
        with torch.no_grad(): result=model(torch.rand(1,3,*size))
        assert result.shape==(1,1,*size)
        assert torch.isfinite(result).all()


def test_compact_encoder_and_decoder_receive_gradients():
    model=LiteCrack(.5)
    loss=model(torch.rand(2,3,32,32)).square().mean()
    loss.backward()
    for component in (model.encoder,model.decoder,model.output):
        gradients=[p.grad for p in component.parameters() if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
