"""Manifest-only data access; query records cannot enter the training split."""
import hashlib
import io
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageOps
import torch
from src.archive_io import read_sample


def verify_files(records,root):
    root=Path(root).resolve()
    for row in records:
        for field in ('image','mask'):
            if hashlib.sha256(read_sample(root,row,field)).hexdigest()!=row[field+'_sha256']:
                raise ValueError(f"Dataset checksum changed: {row['image_id']} {field}")


class CrackDataset:
    def __init__(self,manifest,root,split,size=256,augment=False):
        if augment and split!='source_train': raise ValueError('Only source_train may augment')
        self.rows=[r for r in manifest['records'] if r['split']==split]
        if not self.rows: raise ValueError('Empty split: '+split)
        self.root=Path(root); self.size=size; self.augment=augment
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(io.BytesIO(read_sample(self.root,row,'image'))) as source: image=source.convert('RGB')
        with Image.open(io.BytesIO(read_sample(self.root,row,'mask'))) as source:
            raw=np.asarray(source)
            if raw.ndim==3:
                if not np.all(raw[...,0]==raw[...,1]) or not np.all(raw[...,0]==raw[...,2]):
                    raise ValueError('Colored mask requires explicit mapping')
                raw=raw[...,0]
            if not set(np.unique(raw).tolist())<={0,255}: raise ValueError('Nonbinary mask')
            mask=Image.fromarray(raw.astype(np.uint8))
        image=image.resize((self.size,self.size),Image.Resampling.BILINEAR)
        mask=mask.resize((self.size,self.size),Image.Resampling.NEAREST)
        if self.augment:
            if random.random()<.5: image,mask=ImageOps.mirror(image),ImageOps.mirror(mask)
            if random.random()<.5: image,mask=ImageOps.flip(image),ImageOps.flip(mask)
        return (torch.from_numpy(np.array(image,dtype=np.float32).transpose(2,0,1)/255),
                torch.from_numpy((np.array(mask)>0).astype(np.float32)[None]))
