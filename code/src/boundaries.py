import torch
from torch.nn import functional as F

def boundary_target(mask):
    # Replicated borders avoid inventing edges at the image perimeter.
    padded=F.pad(mask,(1,1,1,1),mode='replicate')
    return (F.max_pool2d(padded,3,1)+F.max_pool2d(-padded,3,1)).clamp(0,1)

def boundary_loss(logits,mask):
    target=boundary_target(mask)
    # Fixed weight avoids unstable image-specific weighting on empty masks.
    return F.binary_cross_entropy_with_logits(logits,target,pos_weight=logits.new_tensor(5.))

def boundary_f1(logits,mask):
    pred=boundary_target((logits>=0).float())>0
    truth=boundary_target(mask)>0
    # Symmetric many-to-one matching within a two-pixel Chebyshev radius.
    near_truth=F.max_pool2d(truth.float(),5,1,2)>0
    near_pred=F.max_pool2d(pred.float(),5,1,2)>0
    dims=(1,2,3)
    pcount=pred.sum(dims); tcount=truth.sum(dims)
    precision=(pred&near_truth).sum(dims)/pcount.clamp_min(1)
    recall=(truth&near_pred).sum(dims)/tcount.clamp_min(1)
    score=2*precision*recall/(precision+recall).clamp_min(1e-8)
    return torch.where((pcount==0)&(tcount==0),1.,score)
