"""Orchestrate the frozen follow-up code without modifying completed experiments."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import signal
import stat
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.atomic import write_json
from scripts.release_integrity import safe_member,digest


class WorkerStillRunning(RuntimeError):pass


def read(path):return json.loads(Path(path).read_text())


def verify_backup(path):
    path=Path(path)
    if path.is_dir():
        manifest=read(path/'FOLLOWUP_BACKUP_MANIFEST.json')
        for name,entry in manifest['files'].items():
            safe_member(name);asset=(path/name).resolve()
            if not asset.is_relative_to(path.resolve()) or not asset.is_file() or digest(asset)!=entry['sha256'] or asset.stat().st_size!=entry['bytes']:
                raise ValueError('Extracted backup checksum/path mismatch: '+name)
        return manifest
    with zipfile.ZipFile(path) as z:
        infos=z.infolist();names=[safe_member(i.filename) for i in infos]
        if len(names)!=len(set(n.casefold() for n in names)):raise ValueError('Duplicate backup paths')
        if sum(i.file_size for i in infos)>4*1024**3 or any(stat.S_ISLNK(i.external_attr>>16) for i in infos):
            raise ValueError('Backup exceeds bound or contains symlinks')
        marker='FOLLOWUP_BACKUP_MANIFEST.json'
        if z.getinfo(marker).file_size>8*1024**2:raise ValueError('Oversized inventory')
        manifest=json.loads(z.read(marker));entries=manifest['files']
        if set(names)!=(set(entries)|{marker}):raise ValueError('Backup inventory differs')
        for name,entry in entries.items():
            if z.getinfo(name).file_size!=entry['bytes']:raise ValueError('Backup member size mismatch')
            h=hashlib.sha256()
            with z.open(name) as stream:
                for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
            if h.hexdigest()!=entry['sha256']:raise ValueError('Backup checksum mismatch: '+name)
    return manifest


def backup_plan(path):
    path=Path(path)
    if path.is_dir():return read(path/'MULTISEED_PLAN.json')
    with zipfile.ZipFile(path) as z:return json.loads(z.read('MULTISEED_PLAN.json'))


def restore(path,destination,expected_plan):
    """Validate everything and detect conflicts before copying any backup file."""
    path=Path(path);destination=Path(destination).resolve()
    manifest=verify_backup(path)
    if backup_plan(path)!=expected_plan:raise ValueError('Backup belongs to a different multi-seed protocol')
    entries=manifest['files']
    for name,entry in entries.items():
        target=(destination/name).resolve()
        if not target.is_relative_to(destination):raise ValueError('Restore path escapes destination')
        if target.exists() and (not target.is_file() or digest(target)!=entry['sha256']):
            raise ValueError('Existing result differs; preserve it and restore into a fresh folder: '+name)
    def put(name,content):
        target=destination/name
        if not target.exists():
            from src.atomic import atomic_write
            atomic_write(target,lambda stream:stream.write(content))
    if path.is_dir():
        for name in entries:put(name,(path/name).read_bytes())
    else:
        with zipfile.ZipFile(path) as z:
            for name in entries:put(name,z.read(name))
    return {'status':'restored','files':len(entries),'source_form':'directory' if path.is_dir() else 'zip'}


def latest_backup(paths,expected_plan):
    candidates=[]
    for path in paths:
        try:
            if backup_plan(path)==expected_plan:
                manifest=verify_backup(path)
                candidates.append((manifest.get('created',''),str(path),Path(path)))
        except (ValueError,KeyError,FileNotFoundError,zipfile.BadZipFile):continue
    return max(candidates)[2] if candidates else None


def validate_source_index(project,config,require_complete=True):
    source=Path(project)/'source_multiseed';index=read(source/'parents.json')
    wanted={(n,s) for n in config['source_models'] for s in config['training_seeds']}
    rows=index['entries'];actual=[(r['name'],r['seed']) for r in rows]
    if len(actual)!=len(set(actual)) or not set(actual).issubset(wanted):raise ValueError('Duplicate or unexpected training seed/model')
    if require_complete and (set(actual)!=wanted or not index.get('complete')):raise ValueError('All nine source runs are required')
    if index.get('protocol')!='followup_multiseed_source_v1' or index['manifest_sha256']!=config['manifest_sha256']:
        raise ValueError('Parent matrix protocol mismatch')
    for row in rows:
        folder=(source/row['checkpoint_dir']).resolve()
        if not folder.is_relative_to(source.resolve()):raise ValueError('Parent checkpoint escapes source tree')
        pointer=read(folder/'best.json');snapshot=(folder/pointer['file']).resolve()
        if snapshot.parent!=folder or digest(snapshot)!=pointer['sha256'] or pointer['sha256']!=row['sha256']:
            raise ValueError('Parent checkpoint checksum mismatch')
        status=read(folder.parent/'status.json');saved=read(folder.parent/'config.json')
        if status.get('state')!='training_complete' or saved['seed']!=row['seed']:
            raise ValueError('Parent source training incomplete')
        template=read(Path(project)/'configs'/f"{row['name']}.json")
        expected={**template,'seed':row['seed'],'device':config['device'],
                  'protocol':'followup_multiseed_source_v1','evaluation_splits':['source_validation']}
        if any(saved.get(k)!=v for k,v in expected.items()):raise ValueError('Source run config differs from predeclared template')
        import torch
        state=torch.load(snapshot,map_location='cpu',weights_only=False)
        if state['config']!=saved or state['manifest_hash']!=config['manifest_sha256']:
            raise ValueError('Source snapshot seed/config/manifest differs from its run record')
    return index


def verify_project(project):
    project=Path(project);plan=read(project/'MULTISEED_PLAN.json')
    for name,sha in {**plan['frozen_source'],**plan['orchestration_source']}.items():
        safe_member(name)
        if digest(project/name)!=sha:raise ValueError('Frozen workflow code changed: '+name)
    config=read(project/'configs/followup_v1.json')
    if config['training_seeds']!=[42,73,101] or config['source_models']!=['litecrack','litecrack_ba','stabilized_r2au']:
        raise ValueError('Expected the predeclared three-seed, three-model matrix')
    return config


def run_worker(project,arguments):
    project=Path(project)
    with (project/'multiseed_execution.log').open('a',encoding='utf-8',buffering=1) as log:
        worker=subprocess.Popen([sys.executable,'scripts/experiments/run_followup.py',*arguments],cwd=project,
                                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in worker.stdout:print(line,end='',flush=True);log.write(line)
            code=worker.wait()
        except KeyboardInterrupt:
            worker.send_signal(signal.SIGTERM)
            try:worker.wait(timeout=120)
            except subprocess.TimeoutExpired:
                raise WorkerStillRunning('Worker has not stopped; wait for it before making a recovery ZIP')
            raise
    if code:raise RuntimeError('Worker failed; preserve the recovery ZIP and inspect multiseed_execution.log')


def export(project,backup_dir,label):
    from scripts.operations.package_followup import package
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    target=Path(backup_dir)/f'multiseed_{label}_{stamp}.zip'
    report=package(project,target)
    print('RECOVERY ZIP: '+str(target),flush=True)
    print('SHA256: '+report['sha256'],flush=True)
    return report


def execute(project,data_root,stage,seed,backup_dir,worker=run_worker,exporter=export):
    project=Path(project);config=verify_project(project)
    import importlib.metadata
    import torch
    if config['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('GPU required for the production three-seed protocol')
    runtime={'torch':str(torch.__version__),'torchvision':importlib.metadata.version('torchvision'),
             'numpy':importlib.metadata.version('numpy'),'cuda':torch.version.cuda,
             'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    lock=project/'RUNTIME_LOCK.json'
    if lock.exists() and read(lock)!=runtime:raise ValueError('Runtime changed; restore the original compatible GPU environment')
    if not lock.exists():write_json(lock,runtime)
    common=['--data-root',str(Path(data_root).resolve())]
    label=f'seed{seed}' if stage=='train' else stage
    error=None
    try:
        if stage=='train':
            if seed not in config['training_seeds']:raise ValueError('Unknown training seed')
            worker(project,[*common,'--stage','source-train','--only-seed',str(seed),
                           '--output',str((project/'source_multiseed').resolve())])
            index=validate_source_index(project,config,False)
            if {(r['name'],r['seed']) for r in index['entries'] if r['seed']==seed}!={(n,seed) for n in config['source_models']}:
                raise RuntimeError('Seed block stopped safely before completion; export retained, rerun this cell to resume')
        elif stage=='evaluate':
            validate_source_index(project,config)
            worker(project,[*common,'--stage','source-evaluate','--parents-index',str((project/'source_multiseed/parents.json').resolve()),
                           '--output',str((project/'source_multiseed_evaluation').resolve())])
            results=read(project/'source_multiseed_evaluation/results.json')['results']
            wanted={f'{n}_seed{s}' for n in config['source_models'] for s in config['training_seeds']}
            if set(results)!=wanted:raise RuntimeError('Source evaluation incomplete')
        elif stage=='adapt':
            validate_source_index(project,config)
            parent=['--parents-index',str((project/'source_multiseed/parents.json').resolve())]
            development=(project/'multiseed_development').resolve()
            worker(project,[*common,*parent,'--stage','develop','--output',str(development)])
            if not (development/'selection.json').exists():raise RuntimeError('Development stopped safely; rerun to resume')
            matrix=(project/'multiseed_matrix').resolve()
            worker(project,[*common,*parent,'--stage','confirm','--development',str(development),'--output',str(matrix)])
            if not (matrix/'status.json').exists() or read(matrix/'status.json').get('state')!='complete':
                raise RuntimeError('Locked matrix stopped safely; rerun to resume')
            from scripts.evaluation.summarize_followup import summarize
            write_json(matrix/'summary.json',summarize(read(matrix/'results.json')))
        else:raise ValueError('Unknown multi-seed stage')
        write_json(project/f'COMPLETED_{label}.json',{'stage':stage,'seed':seed if stage=='train' else None,
                   'state':'complete','scientific_scope':'Repeated seeds on the old pilot; independent scenes and human effort remain separate.'})
    except BaseException as caught:
        error=caught
        raise
    finally:
        # Worker interruptions are waited on before export; never package a live writer.
        if stage in ('train','evaluate','adapt') and not isinstance(error,WorkerStillRunning):
            try:exporter(project,backup_dir,label)
            except Exception as backup_error:
                if error is None:raise
                print('Recovery export failed; original snapshots remain intact: '+type(backup_error).__name__,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=ROOT);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--stage',choices=['train','evaluate','adapt'],required=True)
    p.add_argument('--seed',type=int,choices=[42,73,101]);p.add_argument('--backup-dir',type=Path,required=True)
    a=p.parse_args();execute(a.project,a.data_root,a.stage,a.seed,a.backup_dir)
