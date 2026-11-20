"""Binary loss/metrics with explicit per-image averaging and empty-mask rules."""
import torch
from torch.nn import functional as F


def dice_loss(logits,target):
    probability=logits.sigmoid()
    intersection=(probability*target).sum((1,2,3))
    denominator=(probability+target).sum((1,2,3))
    return (1-(2*intersection+1e-6)/(denominator+1e-6)).mean()


def region_loss(logits,target,kind='dice'):
    if kind=='dice': return dice_loss(logits,target)
    if kind=='bce_dice': return F.binary_cross_entropy_with_logits(logits,target)+dice_loss(logits,target)
    raise ValueError('Unknown region loss')


def image_metrics(logits,target):
    predicted=logits.sigmoid()>=.5
    truth=target>=.5
    dimensions=(1,2,3)
    tp=(predicted & truth).sum(dimensions).float()
    fp=(predicted & ~truth).sum(dimensions).float()
    fn=(~predicted & truth).sum(dimensions).float()
    dice_den=2*tp+fp+fn
    union=tp+fp+fn
    return {'dice':torch.where(dice_den==0,1,2*tp/dice_den.clamp_min(1)),
            'iou':torch.where(union==0,1,tp/union.clamp_min(1)),
            'precision':torch.where(tp+fp==0,(fn==0).float(),tp/(tp+fp).clamp_min(1)),
            'recall':torch.where(tp+fn==0,1,tp/(tp+fn).clamp_min(1))}
