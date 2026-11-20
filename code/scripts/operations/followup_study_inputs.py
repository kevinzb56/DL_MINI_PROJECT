"""Prepare real external-scene manifests and summarize actual annotation sessions."""
import argparse
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from followup.adaptation import sha
from followup.scenes import audit_scenes,inside,require_scene_size
from src.atomic import write_json

COLUMNS=['image_id','site_id','group_id','split','image','mask','road_mask','annotation_complete','annotation_origin']


def initialize(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for name in ('images','cracks','roads','annotation_sessions'):(root/name).mkdir(exist_ok=True)
    path=root/'scene_index.csv'
    if not path.exists():
        with path.open('x',encoding='utf-8',newline='') as stream:csv.writer(stream).writerow(COLUMNS)
    for name in ('usage_terms.txt','physical_sites.txt'):
        path=root/name
        if not path.exists():path.write_text('',encoding='utf-8')
    report={'status':'awaiting_real_data','index':'scene_index.csv',
            'required_images':{'scene_train':24,'scene_validation':12,'scene_test':30},
            'required_disjoint_sites':{'scene_train':3,'scene_validation':2,'scene_test':5},
            'scope':'No labels, site independence or human effort were fabricated.'}
    write_json(root/'READINESS.json',report);return report


def freeze_manifest(root,index,terms,provenance,output,pilot_path=ROOT/'data/manifests/pilot_v1.json',config_path=ROOT/'configs/followup_v1.json'):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Choose a new manifest filename; frozen inputs are immutable')
    with Path(index).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if not set(COLUMNS).issubset(reader.fieldnames or []):raise ValueError('Scene CSV columns incomplete')
        rows=list(reader)
    if not rows:raise ValueError('No real scene records supplied')
    records=[]
    for row in rows:
        row={k:row[k].strip() for k in COLUMNS}
        if row['annotation_complete'].lower()!='true':raise ValueError('Only complete crack and road annotations may be frozen')
        row['annotation_complete']=True
        for field in ('image','mask','road_mask'):row[field+'_sha256']=sha(inside(root,row[field]))
        records.append(row)
    data={'protocol':'followup_external_scenes_v1','records':records,
          'usage_terms':Path(terms).read_text(encoding='utf-8').strip(),
          'physical_site_provenance':Path(provenance).read_text(encoding='utf-8').strip(),
          'index_sha256':sha(index),'created_utc':datetime.now(timezone.utc).isoformat()}
    # Audit a temporary manifest first; incomplete data never leaves a frozen output.
    import tempfile
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        candidate=Path(scratch)/'candidate.json';write_json(candidate,data)
        audited=audit_scenes(candidate,root,json.loads(Path(pilot_path).read_text()))
        require_scene_size(audited,json.loads(Path(config_path).read_text())['scene_requirements'])
    # Exclusive creation makes duplicate submissions explicit.
    with output.open('x',encoding='utf-8') as stream:stream.write(json.dumps(data,indent=2)+'\n')
    output.with_suffix(output.suffix+'.sha256').write_text(sha(output)+'  '+output.name+'\n')
    return {'status':'frozen','manifest':str(output),'sha256':sha(output),'images':len(records),
            'physical_site_independence':'Supplied collection provenance; not independently certified.'}


def annotation_report(folder):
    records=[];seen=set();incomplete=0
    for path in sorted(Path(folder).glob('*.json')):
        row=json.loads(path.read_text())
        if 'started_utc' not in row:continue
        if row['id'] in seen:raise ValueError('Duplicate annotation session')
        seen.add(row['id'])
        if row.get('state')!='complete':incomplete+=1;continue
        if row.get('annotation_origin')!='human_corrected':raise ValueError('Imported labels cannot count as human effort')
        if row.get('starting_mask') not in ('blank','prediction'):raise ValueError('Unknown annotation starting condition')
        value=row.get('elapsed_seconds')
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:raise ValueError('Invalid recorded correction time')
        for field in ('image_sha256','mask_sha256'):
            if not isinstance(row.get(field),str) or len(row[field])!=64:raise ValueError('Missing annotation asset identity')
        for field,hash_field in [('image','image_sha256'),('mask','mask_sha256')]:
            # Relocated sessions retain their recorded hashes but are marked below.
            if Path(row[field]).is_file() and sha(row[field])!=row[hash_field]:raise ValueError('Annotation asset changed')
        records.append(row)
    groups={}
    for condition in ('blank','prediction'):
        values=[r['elapsed_seconds'] for r in records if r['starting_mask']==condition]
        groups[condition]={'sessions':len(values),'mean_seconds':statistics.mean(values) if values else None,
                           'median_seconds':statistics.median(values) if values else None}
    tasks={}
    for row in records:tasks.setdefault((row['task_id'],row['image_sha256']),[]).append(row)
    pairs=[];practice=[]
    for (task,image),rows in tasks.items():
        blank=[r for r in rows if r['starting_mask']=='blank'];prediction=[r for r in rows if r['starting_mask']=='prediction']
        if blank and prediction:
            overlap={r['annotator_code'] for r in blank}&{r['annotator_code'] for r in prediction}
            if overlap:practice.append(task)
            pairs.append({'task_id':task,'image_sha256':image,
                          'blank_seconds_mean':statistics.mean(r['elapsed_seconds'] for r in blank),
                          'prediction_seconds_mean':statistics.mean(r['elapsed_seconds'] for r in prediction),
                          'same_annotator_repeated_image':bool(overlap)})
    assets_available=all(Path(r[f]).is_file() for r in records for f in ('image','mask')) if records else False
    return {'status':'descriptive_sessions_available' if records else 'awaiting_real_human_sessions',
            'complete_sessions':len(records),'incomplete_sessions':incomplete,'by_starting_condition':groups,
            'paired_tasks':pairs,'practice_effect_task_ids':practice,
            'recorded_assets_available_for_local_hash_check':assets_available,
            'expert_effort_goal_established':False,
            'scope':'Self-reported human sessions; descriptive times only. No causal effort claim or expert-quality validation is inferred.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('init');a.add_argument('--root',type=Path,default=ROOT/'data/followup_external')
    a=s.add_parser('freeze-scenes');a.add_argument('--root',type=Path,required=True)
    a.add_argument('--index',type=Path,required=True);a.add_argument('--usage-terms',type=Path,required=True)
    a.add_argument('--site-provenance',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=s.add_parser('annotation-report');a.add_argument('--sessions',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='init':report=initialize(a.root)
    elif a.command=='freeze-scenes':report=freeze_manifest(a.root,a.index,a.usage_terms,a.site_provenance,a.output)
    else:report=annotation_report(a.sessions);write_json(a.output,report)
    print(json.dumps(report,indent=2))
