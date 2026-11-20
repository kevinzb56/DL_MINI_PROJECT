"""Region-only compact model; no boundary head or pretrained external data."""
import torch
from torch import nn
from torch.nn import functional as F


def replace_batchnorm(module):
    for name,child in list(module.named_children()):
        if isinstance(child,nn.BatchNorm2d):
            setattr(module,name,nn.GroupNorm(1,child.num_features))
        else: replace_batchnorm(child)


class SeparableBlock(nn.Module):
    def __init__(self,incoming,outgoing):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Conv2d(incoming,incoming,3,padding=1,groups=incoming,bias=False),
            nn.Conv2d(incoming,outgoing,1,bias=False),nn.GroupNorm(1,outgoing),nn.ReLU(),
            nn.Conv2d(outgoing,outgoing,3,padding=1,groups=outgoing,bias=False),
            nn.Conv2d(outgoing,outgoing,1,bias=False),nn.GroupNorm(1,outgoing),nn.ReLU())
    def forward(self,x): return self.layers(x)


class LiteCrack(nn.Module):
    def __init__(self,width_mult=.5,boundary_aware=False):
        super().__init__()
        from torchvision.models import mobilenet_v3_small
        # weights=None prevents both downloads and unreported ImageNet exposure.
        self.encoder=mobilenet_v3_small(weights=None,width_mult=width_mult).features
        replace_batchnorm(self.encoder)
        self.indices=(0,1,3,8,12)
        channels=[self.encoder[i].out_channels for i in self.indices]
        decoder_widths=(128,64,32,16)
        current=channels[-1]
        self.decoder=nn.ModuleList()
        for skip,width in zip(reversed(channels[:-1]),decoder_widths):
            self.decoder.append(SeparableBlock(current+skip,width)); current=width
        self.refine=SeparableBlock(current,current)
        self.output=nn.Conv2d(current,1,1)
        self.boundary=None
        if boundary_aware:
            # Keep the region initialization and subsequent sampler RNG identical
            # to the region-only control for the same seed.
            with torch.random.fork_rng(devices=[]):
                self.boundary=nn.Conv2d(current,1,1)
    def forward(self,x):
        original_size=x.shape[-2:]
        skips=[]
        for index,layer in enumerate(self.encoder):
            x=layer(x)
            if index in self.indices: skips.append(x)
        for block,skip in zip(self.decoder,reversed(skips[:-1])):
            x=F.interpolate(x,size=skip.shape[-2:],mode='nearest')
            x=block(torch.cat([x,skip],dim=1))
        x=F.interpolate(x,size=original_size,mode='nearest')
        features=self.refine(x)
        region=self.output(features)
        return (region,self.boundary(features)) if self.boundary is not None else region
