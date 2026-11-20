"""Assess every fixed smoke candidate on the same source-validation subset.

This adds diagnostic comparison evidence; it never changes job activation gates.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import torch
from inspection.engine import binary_metrics,load_state,model_config,probability
from inspection.replay import verified_buffer
from inspection.store import Store
from src.atomic import write_json
from src.model_factory import build_model


def evaluate(report_path):
    report=json.loads(report_path.read_text())
    store=Store(report['workspace'])
    folder,buffer=verified_buffer(store)
    rows=[r for r in buffer['records'] if r['split']=='source_validation']
    torch.set_num_threads(2)
    cache={}
    def score(vid):
        if vid in cache:return cache[vid]
        version=store.version(vid,report['project_id']);state=load_state(store,version);config=model_config(state)
        model=build_model(config).eval();model.load_state_dict(state['model'])
        scores=[]
        for row in rows:
            with Image.open(folder/row['image']) as image,Image.open(folder/row['mask']) as truth:
                predicted=(probability(model,image.convert('RGB'),config['image_size'])>=.5).astype(np.uint8)*255
                scores.append({'image_id':row['image_id'],'dice':binary_metrics(predicted,truth)['dice']})
        result={'mean_dice':float(np.mean([r['dice'] for r in scores])),'rows':scores}
        cache[vid]=result;return result
    comparison=[]
    for job in report['jobs']:
        parent=score(job['parent_id']);candidate=score(job['candidate_id'])
        comparison.append({'job_id':job['id'],'parent_id':job['parent_id'],
                           'source_replay':job['settings'].get('source_replay',False),
                           'parent_dice':parent['mean_dice'],'candidate_dice':candidate['mean_dice'],
                           'source_change_points':100*(candidate['mean_dice']-parent['mean_dice']),
                           'scores':candidate['rows'],'scope':'Same fixed eight-image source-validation subset; not source holdout.'})
    report['paired_source_validation']=comparison
    write_json(report_path,report)
    for row in comparison:print(row['parent_id'][:7],row['source_replay'],round(row['source_change_points'],3))


if __name__=='__main__':evaluate(Path('output/phase7/real_validation.json'))
