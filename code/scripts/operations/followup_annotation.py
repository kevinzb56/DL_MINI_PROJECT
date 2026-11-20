"""Record real human correction sessions without inventing annotation effort."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from followup.adaptation import sha
from followup.scenes import binary
from PIL import Image
from src.atomic import write_json


def start(output,image,annotator,starting_mask,task_id):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    record={'id':uuid.uuid4().hex,'image':str(Path(image).resolve()),'image_sha256':sha(image),
            'annotator_code':annotator,'task_id':task_id,'starting_mask':starting_mask,
            'started_utc':datetime.now(timezone.utc).isoformat(),'start_monotonic_ns':time.monotonic_ns(),
            'state':'running','scope':'Manual timed human correction session; self-reported provenance.'}
    path=output/(record['id']+'.json');write_json(path,record);return path


def finish(path,mask,origin):
    path=Path(path);record=json.loads(path.read_text())
    if record['state']!='running':raise ValueError('Session already ended')
    if origin!='human_corrected':raise ValueError('Imported reference masks cannot count as human correction effort')
    if sha(record['image'])!=record['image_sha256']:raise ValueError('Source image changed')
    truth=binary(mask)
    with Image.open(record['image']) as image:
        if truth.shape!=(image.height,image.width):raise ValueError('Correction dimensions differ from original')
    elapsed=(time.monotonic_ns()-record['start_monotonic_ns'])/1e9
    if elapsed<=0:raise ValueError('Invalid timer; do not resume timing across an OS reboot')
    record.update(state='complete',elapsed_seconds=elapsed,finished_utc=datetime.now(timezone.utc).isoformat(),
                  mask_sha256=sha(mask),mask=str(Path(mask).resolve()),annotation_origin=origin)
    write_json(path,record);return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('start');a.add_argument('--image',type=Path,required=True)
    a.add_argument('--annotator',required=True);a.add_argument('--task-id',required=True)
    a.add_argument('--starting-mask',choices=['prediction','blank'],required=True)
    a.add_argument('--output',type=Path,default=Path('artifacts/followup_v1/annotation_sessions'))
    a=s.add_parser('finish');a.add_argument('--session',type=Path,required=True)
    a.add_argument('--mask',type=Path,required=True);a.add_argument('--origin',choices=['human_corrected','existing_dataset'],required=True)
    a=p.parse_args()
    if a.action=='start':print(start(a.output,a.image,a.annotator,a.starting_mask,a.task_id))
    else:print(json.dumps(finish(a.session,a.mask,a.origin),indent=2))
