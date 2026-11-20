"""Real-weight, one-step CPU integration check on previously known release subsets."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from followup.adaptation import adapt,sha
from followup.study import original_parents,score,gate,freeze
from src.atomic import write_json
from src.model_factory import build_model


def smoke(output):
    assets=ROOT/'release_assets';output=Path(output)
    source=json.loads((assets/'source_replay/manifest.json').read_text())['records']
    demo=json.loads((assets/'demo/manifest.json').read_text())['records']
    records=[]
    for rows,prefix in [(source,'source_replay/'),(demo,'demo/')]:
        for original in rows:
            row=copy.deepcopy(original)
            for field in ('image','mask'):row[field]=prefix+row[field]
            records.append(row)
    manifest={'records':records}
    selected={r['role']:r for r in demo}
    config=json.loads((ROOT/'configs/followup_v1.json').read_text())
    state,parent_sha=original_parents(ROOT,config['manifest_sha256'])['litecrack_ba']
    condition=next(c for c in config['conditions'] if c['name']=='distill_5')
    settings={**config['adaptation'],**condition,'epochs':1,'device':'cpu','threads':2}
    freeze(output/'protocol.json',{'scope':'One-step real-checkpoint CPU smoke on known release assets; not full retention/independent test.',
            'parent_sha256':parent_sha,'settings':settings,'source_manifest_sha256':sha(assets/'source_replay/manifest.json'),
            'demo_manifest_sha256':sha(assets/'demo/manifest.json')})
    torch.set_num_threads(2)
    model=build_model(state['config']);model.load_state_dict(state['model'])
    def evaluate(m):
        return {'source_validation_subset':score(m,manifest,assets,'source_validation',256,'cpu'),
                'target_local_validation':score(m,manifest,assets,'target_acquisition',256,'cpu',{selected['validation']['image_id']})}
    baseline=evaluate(model)
    run=output/'adaptation'
    candidate=adapt(state,parent_sha,manifest,config['manifest_sha256'],assets,[selected['support']['image_id']],
                    settings,42,run,resume=(run/'checkpoints/latest.json').exists())
    if candidate is None:return
    observed=evaluate(candidate)
    result={'status':'passed','scope':'CPU integration only; 1 known acquisition support, 1 separate local validation, 16 replay and 8 source validation images.',
            'scientific_goals_met':False,'activation_performed':False,'baseline':baseline,'candidate':observed,
            'diagnostic_gate':gate(baseline['source_validation_subset']['summary']['dice'],observed['source_validation_subset']['summary']['dice'],
                baseline['target_local_validation']['summary']['dice'],observed['target_local_validation']['summary']['dice'],config['gates'])}
    write_json(output/'results.json',result)
    print(json.dumps({k:result[k] for k in ('status','scope','scientific_goals_met','activation_performed','diagnostic_gate')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'artifacts/followup_smoke_v1')
    a=p.parse_args();smoke(a.output)
