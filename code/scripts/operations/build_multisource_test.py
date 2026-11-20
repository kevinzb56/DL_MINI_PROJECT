"""Build lossless 256px training/validation cache from the fully audited release."""
from pathlib import Path
import hashlib, io, json, zipfile, argparse
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
def sha(data): return hashlib.sha256(data).hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',type=int,choices=(256,512),default=512)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--output',default='output/multisource_test_v1/Full_Training_v1.zip')
    args=parser.parse_args()
    path=ROOT/'output/multisource_full_v1/manifest.json'
    parent=json.loads(path.read_text()); assert parent['status']=='AUDITED_MULTISOURCE_V1'
    assert len(parent['records'])==3564 and parent['decoded_cross_split_duplicates']==0
    out=ROOT/args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    assert not out.exists()
    records=[]
    with zipfile.ZipFile(out.with_suffix('.zip.partial'),'w',zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr('audit_manifest.json',path.read_bytes())
        for archive_name in parent['archives']:
            with zipfile.ZipFile(ROOT/'data/external/multisource_cracks_v1'/archive_name) as archive:
                for original in parent['records']:
                    if original['archive']!=archive_name or original['publisher_split']!='test': continue
                    row=dict(original); row.pop('archive')
                    for field in ('image','mask'):
                        raw=archive.read(original[field]); assert sha(raw)==original[field+'_sha256']
                        with Image.open(io.BytesIO(raw)) as source:
                            image=source.convert('RGB') if field=='image' else source.convert('L')
                            if field=='mask': assert set(np.unique(np.asarray(image)).tolist())<={0,255}
                            image=image.resize((args.size,args.size),Image.Resampling.BILINEAR if field=='image' else Image.Resampling.NEAREST)
                            data=io.BytesIO(); image.save(data,format='PNG'); data=data.getvalue()
                        name=f'data/{field}/{row["image_id"]}.png'
                        bundle.writestr(name,data); row[field]=name; row[field+'_sha256']=sha(data)
                    records.append(row)
                    if len(records)%200==0: print('Prepared',len(records),'/496',flush=True)
        assert len(records)==496
        manifest=dict(status='FROZEN_MULTISOURCE_V1',records=records,parent_manifest_sha256=sha(path.read_bytes()),
                      preprocessing=f'RGB bilinear / binary nearest {args.size}px, lossless PNG; flips at training',test_evaluation=False)
        bundle.writestr('manifest.json',json.dumps(manifest,indent=2))
        config=json.loads((ROOT/'configs/litecrack.json').read_text()); config.update(protocol='multisource_v1_seed42',evaluation_splits=[],image_size=args.size,epochs=args.epochs)
        bundle.writestr('config.json',json.dumps(config,indent=2))
        for file in (ROOT/'src').glob('*.py'): bundle.write(file,file.relative_to(ROOT).as_posix())
        for name in ('scripts/experiments/train_baseline.py','scripts/experiments/run_multisource_full.py'): bundle.write(ROOT/name,name)
    out.with_suffix('.zip.partial').replace(out)
    out.with_suffix('.zip.sha256').write_text(sha(out.read_bytes())+'\n')
    print('Built',out.stat().st_size,flush=True)
if __name__=='__main__': main()
