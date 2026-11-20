"""Build a train/validation-only cache tied to the immutable pilot recovery."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]


def sha(data): return hashlib.sha256(data).hexdigest()


def build():
    output=ROOT/'output/multisource_diagnostic'
    output.mkdir(parents=True,exist_ok=True)
    recovery=ROOT/'output/multisource_archive1_pilot/recovery/multisource_pilot_recovery_20261114T184535Z.zip'
    assert sha(recovery.read_bytes())==recovery.with_suffix('.zip.sha256').read_text().split()[0]
    target=output/'Multisource_Diagnostic_v1.zip'
    if target.exists(): raise ValueError('Preserve existing diagnostic bundle; choose a new version')
    with zipfile.ZipFile(recovery) as old, zipfile.ZipFile(target.with_suffix('.zip.partial'),'w',zipfile.ZIP_DEFLATED) as bundle:
        manifest_bytes=old.read('artifacts/manifest.json')
        manifest=json.loads(manifest_bytes)
        rows=[r for r in manifest['records'] if r['publisher_split'] in ('train','val')]
        assert len(rows)==476 and not any(r['publisher_split']=='test' for r in rows)
        bundle.writestr('parent/manifest.json',manifest_bytes)
        for name in old.namelist():
            if name.startswith('artifacts/litecrack_seed42/') and not name.endswith('/manifest.json'):
                bundle.writestr('parent/'+name.removeprefix('artifacts/litecrack_seed42/'),old.read(name))
        images=[]; masks=[]; witnesses=[]
        archive=ROOT/'data/external/multisource_cracks_v1/01_ground_gopro.zip'
        with archive.open('rb') as stream:
            h=hashlib.sha256()
            for block in iter(lambda:stream.read(4*1024**2),b''): h.update(block)
        assert h.hexdigest()==manifest['archives'][archive.name]
        with zipfile.ZipFile(archive) as source:
            for i,row in enumerate(rows):
                for field in ('image','mask'):
                    data=source.read(row[field])
                    assert sha(data)==row[field+'_sha256']
                    with Image.open(io.BytesIO(data)) as im:
                        if field=='image':
                            value=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
                            images.append(value)
                        else:
                            raw=np.asarray(im)
                            if raw.ndim==3:
                                assert np.all(raw==raw[...,:1]); raw=raw[...,0]
                            assert set(np.unique(raw).tolist())<={0,255}
                            value=np.asarray(Image.fromarray(raw.astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST))
                            masks.append(value)
                    if i in (0,len(rows)//2,len(rows)-1):
                        member=f'witness/{i}_{field}.bin'
                        bundle.writestr(member,data)
                        witnesses.append(dict(index=i,field=field,member=member,sha256=sha(data)))
                if (i+1)%100==0: print('Cached',i+1,'/',len(rows),flush=True)
        cache=io.BytesIO(); np.savez_compressed(cache,images=np.stack(images),masks=np.stack(masks))
        cache_bytes=cache.getvalue(); bundle.writestr('cache/samples.npz',cache_bytes)
        index=dict(schema_version=1,rows=rows,witnesses=witnesses,size=256,
                   parent_manifest_sha256=sha(manifest_bytes),archive_sha256=h.hexdigest(),
                   samples_sha256=sha(cache_bytes),test_samples_included=False,
                   preprocessing='RGB Pillow bilinear; binary-mask nearest; 256x256; augmentation after cache')
        bundle.writestr('cache/index.json',json.dumps(index,indent=2))
        for path in sorted((ROOT/'src').glob('*.py')): bundle.write(path,path.relative_to(ROOT).as_posix())
        for name in ('scripts/experiments/run_multisource_diagnostic.py','scripts/experiments/train_baseline.py'):
            bundle.write(ROOT/name,name)
    target.with_suffix('.zip.partial').replace(target)
    target.with_suffix('.zip.sha256').write_text(sha(target.read_bytes())+'  '+target.name+'\n')
    print('Built',target.name,target.stat().st_size,'bytes',flush=True)


if __name__=='__main__': build()
