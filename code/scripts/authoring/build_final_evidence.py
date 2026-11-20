"""Consolidate measured evidence; retrospective proxy-group intervals are descriptive."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.project_layout import archive_path
import sys
import xml.etree.ElementTree as ET
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from src.atomic import write_json
from src.model_factory import build_model
from scripts.release_integrity import digest


def paired_interval(differences,records,seed=881,repetitions=10000):
    groups={}
    for iid,value in differences.items():groups.setdefault(records[iid]['group_id'],[]).append(value)
    ordered=sorted(groups)
    sums=np.array([sum(groups[g]) for g in ordered]);sizes=np.array([len(groups[g]) for g in ordered])
    choices=np.random.default_rng(seed).integers(0,len(ordered),(repetitions,len(ordered)))
    estimates=sums[choices].sum(axis=1)/sizes[choices].sum(axis=1)*100
    return {'difference_points':float(np.mean(list(differences.values()))*100),
            'lower_95':float(np.percentile(estimates,2.5)),'upper_95':float(np.percentile(estimates,97.5)),
            'proxy_groups':len(ordered),'images':len(differences),'seed':seed,'replicates':repetitions,
            'method':'Retrospective paired proxy-group percentile bootstrap; group resampling with image-weighted mean.',
            'limits':'Proxy groups are not verified road/site independence. Intervals are descriptive, unadjusted, not confirmatory significance tests.'}


def build(root):
    root=Path(root);out=root/'output/phase8';out.mkdir(exist_ok=True)
    frozen=json.loads((root/'data/manifests/pilot_v1.json').read_text());records={r['image_id']:r for r in frozen['records']}
    baseline=json.loads((root/'data/manifests/baseline_registry_v1.json').read_text())
    phase5=json.loads((root/'output/phase5_archive_verification.json').read_text())
    phase7=json.loads((root/'output/phase7/real_validation.json').read_text())
    demo=json.loads((out/'demo/demo.json').read_text())
    clean=json.loads((out/'release_validation.json').read_text())
    tests=ET.parse(root/'output/phase8-clean-tests.xml').findall('.//testcase')
    counts={'total':len(tests),'passed':sum(not any(t.find(tag) is not None for tag in ('failure','error','skipped')) for t in tests),
            'skipped':sum(t.find('skipped') is not None for t in tests),'failed':sum(any(t.find(tag) is not None for tag in ('failure','error')) for t in tests)}
    if counts['failed']:raise ValueError('Clean release tests did not pass')
    models=[]
    for key,label in [('unet','U-Net'),('r2au','Paper-inspired R2AU')]:
        config=json.loads((root/f'configs/stabilized_{key}.json').read_text())
        parameters=sum(p.numel() for p in build_model(config).parameters())
        row=baseline['models'][key]
        models.append({'name':label,'parameters':parameters,'source':row['evaluation']['source_holdout'],
                       'target':row['evaluation']['target_query'],'checkpoint_sha256':row['checkpoint_sha256']})
    intervals={}
    with zipfile.ZipFile(archive_path(root, 'phase5_results.zip')) as z:
        result=json.loads(z.read('results/results.json'))
        def scores(name,split):
            rows=json.loads(z.read(f'results/{name}_{split}.json'))
            expected={r['image_id'] for r in frozen['records'] if r['split']==split}
            if len(rows)!=len(expected) or {r['image_id'] for r in rows}!=expected:raise ValueError('Per-image split coverage changed')
            return {r['image_id']:r['dice'] for r in rows}
        for key,label in [('control','LiteCrack'),('boundary','LiteCrack-BA')]:
            config=json.loads((root/f'configs/{"litecrack" if key=="control" else "litecrack_ba"}.json').read_text())
            parameters=sum(p.numel() for p in build_model(config).parameters())
            models.append({'name':label,'parameters':parameters,'source':result['static'][key]['source_holdout'],
                           'target':result['static'][key]['target_query'],'checkpoint_sha256':result['protocol']['parents'][key]})
        for split in ('source_holdout','target_query'):
            control,boundary=scores('control_k0',split),scores('boundary_k0',split)
            intervals['boundary_minus_region_'+split]=paired_interval({i:boundary[i]-control[i] for i in control},records)
        for k,split in [(5,'target_query'),(10,'target_query'),(5,'source_holdout'),(10,'source_holdout')]:
            base=scores('boundary_k0',split)
            draws=[scores(f'boundary_draw{draw}_k{k}',split) for draw in range(5)]
            differences={iid:float(np.mean([d[iid] for d in draws]))-base[iid] for iid in base}
            intervals[f'boundary_k{k}_minus_static_{split}']=paired_interval(differences,records)
    gates=[
        {'id':'G1','status':'partial','finding':'Frozen split/hash and exclusion checks preserved; physical capture/site independence is unverified.'},
        {'id':'G2','status':'passed','finding':'339,265 region-only parameters; 83.92% fewer than the implemented R2AU baseline, below 5M.'},
        {'id':'G3','status':'pilot_pass','finding':'Static target Dice meets the within-two-point tradeoff in this single-seed pilot; source accuracy is lower.'},
        {'id':'G4','status':'partial','finding':'Target boundary F1 +6.22 points and Dice +7.71; source/validation do not show a universal improvement. Multi-seed confirmation absent.'},
        {'id':'G5','status':'pilot_pass','finding':'Five-shot boundary mean Dice 57.23% vs static 41.72%; five support draws share the same query set.'},
        {'id':'G6','status':'unmet','finding':'Boundary source-holdout Dice changes -2.81/-4.87 points at K=5/10; replay smoke still fails the source-subset gate.'},
        {'id':'G7','status':'passed_on_named_cpu','finding':'Native FP32 p95 44.88/52.59 ms, 20 warmups and 200 timed model passes, i5-1335U, two threads.'},
        {'id':'G8','status':'passed_on_audited_samples','finding':'Both static ONNX exports pass probability and labeled sample metric parity gates; no global accuracy or ONNX speed claim.'},
        {'id':'G9','status':'passed_demonstration','finding':'Real acquisition-image workflow, reference annotation import, actual adaptation, comparison and exports recorded; no independent accuracy study.'},
        {'id':'G10','status':'partial','finding':'Fresh venv/app bootstrap and CPU tests verified. Full original GPU training has not been rerun from the release.'}]
    report={'version':'1.0.0-pilot','date':'2026-11-01','scope':'Final local research/education pilot release; functional delivery does not imply all research goals passed.',
            'manifest_sha256':digest(root/'data/manifests/pilot_v1.json'),'split_counts':frozen['counts'],
            'static_models':models,'fewshot':phase5['summary'],'phase5_archive_sha256':phase5['sha256'],
            'descriptive_intervals':intervals,'phase7':phase7,'clean_release':clean,'tests':counts,'demonstration':demo,
            'gates':gates,'archives':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(root.glob('phase*.zip'))},
            'limitations':['Single training seed; support draws are not independent new test sets.',
                           'Target query was inspected in earlier phases; results are exploratory.',
                           'Capture/site independence is unverified; bootstrap groups are proxies.',
                           'Wide road scenes still produce vegetation/sky/lamp false positives.',
                           'No measured human annotation time or independent expert-correction study.',
                           'CrackMap usage terms remain unconfirmed; no public redistribution permission is asserted.']}
    write_json(out/'evidence.json',report)
    write_json(out/'acceptance_gates.json',gates)
    print(json.dumps({'tests':counts,'intervals':intervals,'gate_statuses':{g['id']:g['status'] for g in gates}},indent=2))


if __name__=='__main__':build(Path(__file__).resolve().parents[2])
