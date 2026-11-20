"""Summarize training-seed variation separately from shared support-draw variation."""
import argparse
import json
from pathlib import Path
import sys
import statistics
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.atomic import write_json


def summarize(result):
    groups={}
    for row in result['adapted']:
        architecture,separator,seed=row['model'].rpartition('_seed')
        if not separator:architecture=row['model'];seed='original_single_seed'
        for split in ('source_holdout','target_query'):
            key=(architecture,row['condition'],row['k'],split)
            groups.setdefault(key,{}).setdefault(seed,[]).append(row['scores'][split]['summary']['dice'])
    rows=[]
    for (architecture,condition,k,split),seeds in sorted(groups.items()):
        per_seed=[{'training_seed':seed,'draws':len(values),'mean_dice':statistics.mean(values),
                   'support_draw_sd':statistics.stdev(values) if len(values)>1 else None} for seed,values in sorted(seeds.items())]
        means=[r['mean_dice'] for r in per_seed]
        rows.append({'architecture':architecture,'condition':condition,'k':k,'split':split,
                     'training_seeds':len(means),'mean_of_seed_means':statistics.mean(means),
                     'training_seed_sd':statistics.stdev(means) if len(means)>1 else None,'per_seed':per_seed})
    return {'scope':'Exploratory reused-query results; training-seed SD and support-draw SD are separate, not confidence intervals.',
            'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();write_json(a.output,summarize(json.loads(a.results.read_text())))
