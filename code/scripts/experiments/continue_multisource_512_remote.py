"""Explicit checkpoint branch: extend only epoch horizon, preserving parent backup."""
from pathlib import Path
import hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from src.atomic import write_json
from src.checkpoints import save_checkpoint
from src.model_factory import build_model
from scripts.experiments.train_baseline import source_hash

parent=ROOT/'parent_recovery.zip'
original_config=json.loads((ROOT/'config.json').read_text())
with zipfile.ZipFile(parent) as archive:
    pointer=json.loads(archive.read('artifacts/run/checkpoints/latest.json'))
    payload=archive.read('artifacts/run/checkpoints/'+pointer['file'])
    assert hashlib.sha256(payload).hexdigest()==pointer['sha256']
    import io
    state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    assert state['epoch']==20 and state['global_step']==6320
    assert state['manifest_hash']==hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
    assert state['config']['source_sha256']==source_hash()
    assert state['config']['torch_version']==torch.__version__ and state['config']['cuda_runtime']==torch.version.cuda
    config={**original_config,'epochs':100}
    expected={**state['config'],'epochs':100}
    for key in original_config:
        if key!='epochs': assert original_config[key]==state['config'][key]
    assert state['best_metric']==state['metrics']['validation']['dice']
    model=build_model(config)
    model.load_state_dict(state['model'])
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate']);optimizer.load_state_dict(state['optimizer'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5,factor=.1);scheduler.load_state_dict(state['scheduler'])
    run=ROOT/'artifacts/run'; assert not run.exists();run.mkdir(parents=True)
    # Copy the parent best payload with its original identity for provenance;
    # the branched snapshot below is both best/latest because epoch 20 was best.
    torch.set_rng_state(state['rng']['torch'])
    import random,numpy as np
    random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
    torch.cuda.set_rng_state_all(state['rng']['cuda'])
    save_checkpoint(run/'checkpoints',model=model,optimizer=optimizer,scheduler=scheduler,scaler=None,
                    epoch=state['epoch'],global_step=state['global_step'],sampler_state=state['sampler_state'],
                    config=expected,manifest_hash=state['manifest_hash'],metrics=state['metrics'],
                    best_metric=state['best_metric'],is_best=True)
    write_json(run/'config.json',expected)
    write_json(run/'manifest.json',json.loads((ROOT/'manifest.json').read_text()))
    write_json(run/'history.json',state['sampler_state']['history'])
    write_json(ROOT/'continuation_lineage.json',dict(parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
               parent_pointer=pointer,changed_field='epochs',old_value=20,new_value=100,
               preserved='weights optimizer scheduler sampler RNG best metric; official splits unchanged'))
    write_json(ROOT/'config.json',config)
from scripts.experiments.run_multisource_full import main
main()
