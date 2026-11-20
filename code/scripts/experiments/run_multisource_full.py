"""Full single-seed LiteCrack run, deferred test scoring, periodic recovery exports."""
import os, sys, json, time, hashlib, zipfile, threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.experiments.train_baseline import train
def export():
    # Capture immutable checkpoint payloads first and the matching pointers next.
    run=ROOT/'artifacts/run'
    pointers={p.name:json.loads(p.read_text()) for p in (run/'checkpoints').glob('*.json')}
    if not pointers: return
    target=ROOT.parent/f'multisource_full_recovery_{time.time_ns()}.zip'
    with zipfile.ZipFile(target.with_suffix('.zip.partial'),'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('src','scripts'):
            for p in (ROOT/folder).glob('*.py'): z.write(p,p.relative_to(ROOT).as_posix())
        for name in ('manifest.json','config.json','audit_manifest.json'): z.write(ROOT/name,name)
        for p in run.glob('*.json'): z.write(p,p.relative_to(ROOT).as_posix())
        for name,pointer in pointers.items():
            payload=run/'checkpoints'/pointer['file']; data=payload.read_bytes()
            assert hashlib.sha256(data).hexdigest()==pointer['sha256']
            member='artifacts/run/checkpoints/'+payload.name
            if member not in z.namelist(): z.writestr(member,data)
            z.writestr('artifacts/run/checkpoints/'+name,json.dumps(pointer))
    target.with_suffix('.zip.partial').replace(target)
    target.with_suffix('.zip.sha256').write_text(hashlib.sha256(target.read_bytes()).hexdigest()+'\n')
    print('Exported',target.name,flush=True)
def main():
    import torch
    torch.set_num_threads(2)
    stop=threading.Event()
    def backup():
        while not stop.wait(120):
            try: export()
            except Exception as exc: print('Backup retry:',type(exc).__name__,flush=True)
    worker=threading.Thread(target=backup,daemon=True); worker.start()
    try:
        train(json.loads((ROOT/'config.json').read_text()),ROOT/'manifest.json',ROOT,ROOT/'artifacts/run',
              resume=(ROOT/'artifacts/run/checkpoints/latest.json').exists(),final_evaluation=False)
    finally:
        stop.set(); worker.join(); export()
if __name__=='__main__': main()
