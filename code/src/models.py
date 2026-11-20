"""U-Net and a clearly labeled paper-inspired recurrent/attention baseline."""
import torch
from torch import nn
from torch.nn import functional as F


def normalization_layer(channels,normalization):
    return nn.GroupNorm(1,channels) if normalization=='group' else nn.Identity()


class ConvBlock(nn.Module):
    def __init__(self, incoming, outgoing,normalization='none'):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(incoming,outgoing,3,padding=1),normalization_layer(outgoing,normalization),nn.ReLU(),
                                  nn.Conv2d(outgoing,outgoing,3,padding=1),normalization_layer(outgoing,normalization),nn.ReLU())
    def forward(self,x): return self.layers(x)


class RecurrentResidual(nn.Module):
    def __init__(self,incoming,outgoing,normalization='none'):
        super().__init__()
        self.project=nn.Conv2d(incoming,outgoing,1)
        self.units=nn.ModuleList([nn.Conv2d(outgoing,outgoing,3,padding=1) for _ in range(2)])
        self.norms=nn.ModuleList([normalization_layer(outgoing,normalization) for _ in range(2)])
    def forward(self,x):
        residual=self.project(x)
        h=residual
        for unit,norm in zip(self.units,self.norms):
            fixed=h
            h=F.relu(norm(unit(fixed)))
            for _ in range(2): h=F.relu(norm(unit(fixed+h)))
        return residual+h


class AttentionGate(nn.Module):
    def __init__(self,skip_channels,gate_channels):
        super().__init__()
        hidden=max(1,skip_channels//2)
        self.skip=nn.Conv2d(skip_channels,hidden,1)
        self.gate=nn.Conv2d(gate_channels,hidden,1)
        self.score=nn.Conv2d(hidden,1,1)
    def forward(self,skip,gate):
        return skip*torch.sigmoid(self.score(F.relu(self.skip(skip)+self.gate(gate))))


class SegmentationUNet(nn.Module):
    def __init__(self,architecture='unet',base_channels=16,normalization='none'):
        super().__init__()
        if architecture not in ('unet','r2au'): raise ValueError('Unknown architecture')
        if normalization not in ('none','group'): raise ValueError('Unknown normalization')
        block=ConvBlock if architecture=='unet' else RecurrentResidual
        widths=[base_channels*2**i for i in range(5)]
        self.encoders=nn.ModuleList([block(3 if i==0 else widths[i-1],w,normalization) for i,w in enumerate(widths)])
        self.decoders=nn.ModuleList([block(widths[i+1]+widths[i],widths[i],normalization) for i in range(3,-1,-1)])
        self.attention=nn.ModuleList([AttentionGate(widths[i],widths[i+1]) for i in range(3,-1,-1)]) if architecture=='r2au' else None
        self.output=nn.Conv2d(widths[0],1,1)
    def forward(self,x):
        skips=[]
        for i,encoder in enumerate(self.encoders):
            if i: x=F.max_pool2d(x,2)
            x=encoder(x); skips.append(x)
        for i,decoder in enumerate(self.decoders):
            skip=skips[-2-i]
            # Nearest resizing avoids CUDA bilinear-backward nondeterminism in
            # the strict recovery baseline; compact-model experiments are separate.
            x=F.interpolate(x,size=skip.shape[-2:],mode='nearest')
            if self.attention is not None: skip=self.attention[i](skip,x)
            x=decoder(torch.cat([x,skip],dim=1))
        return self.output(x)
