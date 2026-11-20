"""Training-only mask detail audit; no test/validation pixels are opened."""
from pathlib import Path
import io,json,zipfile
from collections import defaultdict
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
manifest=json.loads((ROOT/'output/multisource_full_v1/manifest.json').read_text())
groups=defaultdict(list)
for row in manifest['records']:
 if row['publisher_split']=='train': groups[row['archive']].append(row)
results=[]
for archive_name,rows in groups.items():
 # Deterministic evenly spaced sample, bounded at 24 per archive.
 chosen=[rows[i] for i in np.linspace(0,len(rows)-1,min(24,len(rows)),dtype=int)]
 with zipfile.ZipFile(ROOT/'data/external/multisource_cracks_v1'/archive_name) as archive:
  for row in chosen:
   with Image.open(io.BytesIO(archive.read(row['mask']))) as source:
    raw=np.asarray(source.convert('L'))>0
    record=dict(image_id=row['image_id'],archive=archive_name,group=row['group'],
                original_width=source.width,original_height=source.height,original_foreground_fraction=float(raw.mean()))
    for size in (256,512):
     small=source.convert('L').resize((size,size),Image.Resampling.NEAREST)
     restored=np.asarray(small.resize(source.size,Image.Resampling.NEAREST))>0
     tp=np.count_nonzero(raw&restored); denominator=np.count_nonzero(raw)+np.count_nonzero(restored)
     record[str(size)]=dict(roundtrip_dice=2*tp/denominator if denominator else 1.,
                           original_foreground_recall=tp/max(1,np.count_nonzero(raw)),
                           resized_foreground_fraction=float((np.asarray(small)>0).mean()))
    results.append(record)
 print('Audited',archive_name,len(chosen),flush=True)
summary={}
for name in groups:
 rows=[r for r in results if r['archive']==name]
 summary[name]=dict(n=len(rows),**{str(size):{key:sum(r[str(size)][key] for r in rows)/len(rows)
  for key in ('roundtrip_dice','original_foreground_recall','resized_foreground_fraction')} for size in (256,512)})
out=ROOT/'output/multisource_resolution';out.mkdir(exist_ok=True)
(out/'audit.json').write_text(json.dumps(dict(scope='deterministic training-only sample, up to 24 per archive',
 interpretation='Nearest resize/down-up label fidelity; not model accuracy or proof of causation',results=results,summary=summary),indent=2))
print(json.dumps(summary,indent=2))
