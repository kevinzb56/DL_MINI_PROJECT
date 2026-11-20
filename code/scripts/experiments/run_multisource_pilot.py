"""Run the bounded archive-1 pipeline pilot; never evaluate official test."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from src.archive_io import file_sha256
from src.atomic import write_json


def export():
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    target=ROOT.parent/f'multisource_pilot_recovery_{stamp}.zip'
    temporary=target.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ('src','scripts','tests','configs','artifacts'):
            for path in sorted((ROOT/folder).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    archive.write(path,path.relative_to(ROOT).as_posix())
    temporary.replace(target)
    target.with_suffix('.zip.sha256').write_text(file_sha256(target)+'  '+target.name+'\n')
    print('Exported:',target.name,flush=True)


def main():
    os.chdir(ROOT)
    artifacts=ROOT/'artifacts'; artifacts.mkdir(exist_ok=True)
    state=artifacts/'pipeline_status.json'
    try:
        write_json(state,dict(state='runtime_and_tests',scope='archive1_pipeline_pilot'))
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_multisource.py','-v'],check=True)
        import torch
        from src.model_factory import build_model
        config=json.loads((ROOT/'configs/multisource_archive1_pilot.json').read_text())
        assert torch.cuda.is_available()
        torch.set_num_threads(2)
        model=build_model(config).cuda()
        model(torch.zeros(2,3,256,256,device='cuda')).mean().backward()
        del model
        torch.cuda.empty_cache()
        write_json(state,dict(state='downloading_archive1'))
        subprocess.run([sys.executable,'scripts/data/download_multisource_cracks.py','--root','data',
                        '--files','01_ground_gopro.zip'],check=True)
        write_json(state,dict(state='auditing_archive1'))
        from scripts.data.prepare_multisource import prepare
        manifest=artifacts/'manifest.json'
        if not manifest.exists(): prepare(ROOT/'data',manifest)
        write_json(state,dict(state='training_archive1_pilot',epochs=config['epochs'],test_evaluation=False))
        from scripts.experiments.train_baseline import train
        run=artifacts/'litecrack_seed42'
        # Exercise a real optimizer/RNG/sampler checkpoint restore before continuing.
        train(config,manifest,ROOT/'data',run,stop_after_steps=2,final_evaluation=False)
        export()
        train(config,manifest,ROOT/'data',run,resume=True,final_evaluation=False)
        write_json(state,dict(state='pilot_complete',test_evaluation=False,epochs=config['epochs']))
        export()
    except Exception as exc:
        write_json(state,dict(state='failed',error_type=type(exc).__name__,error=str(exc)))
        export()
        raise


if __name__=='__main__': main()
