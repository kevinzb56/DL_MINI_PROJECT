import pytest
torch=pytest.importorskip('torch')
from src.boundaries import boundary_target,boundary_f1,boundary_loss

def test_empty_and_full_masks_have_no_artificial_edges():
    for value in (0.,1.):
        assert boundary_target(torch.full((1,1,16,16),value)).sum()==0

def test_perfect_edge_and_auxiliary_gradient():
    mask=torch.zeros(1,1,16,16); mask[:,:,4:12,7:9]=1
    assert boundary_f1(mask*20-10,mask).item()==1
    logits=torch.zeros_like(mask,requires_grad=True)
    boundary_loss(logits,mask).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
