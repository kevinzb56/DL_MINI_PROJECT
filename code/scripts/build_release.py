"""Build a versioned application/evidence bundle without copying user workspaces."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.frontend_build import fingerprint
from scripts.release_integrity import MANIFEST,MODEL_ARCHIVES,digest,verify_archive,safe_member
from scripts.project_layout import archive_path


def build(root,target,candidate=False):
    root=Path(root).resolve();target=Path(target).resolve()
    selected={}
    def add(path,name=None):
        path=Path(path)
        if not path.is_file():return
        name=name or path.relative_to(root).as_posix();safe_member(name)
        if path.suffix in ('.py','.md','.txt','.json','.jsx','.js','.html','.ipynb'):
            if re.search(rb'https://kkb-production[.]jupyter-proxy[.]kaggle[.]net/k/[0-9]+/[^\s/]{32,}',path.read_bytes()):
                raise ValueError('Potential authenticated Jupyter URL in '+name)
        selected[name]=path
    for folder in ('inspection','src','scripts','tests','configs','data/manifests','docs','web/src','web/public','web/dist','release_assets','output/kaggle'):
        for path in (root/folder).rglob('*'):
            if '__pycache__' not in path.parts and path.suffix!='.pyc':add(path)
    for folder in ('output/phase6/react-browser','output/phase7','output/phase8','output/pdf'):
        for path in (root/folder).rglob('*'):
            if any(part in ('renders','raw_video','release_assets','package_extract') for part in path.relative_to(root/folder).parts):continue
            add(path)
    for name in ('README.md','START_HERE.md','MODEL_CARD.md','LICENSE','NOTICE.md','Setup_CrackLab.ps1','Start_CrackLab.cmd',
                 'PRD_Lightweight_Few_Shot_Road_Crack_Segmentation.md','web/README.md','web/index.html','web/package.json','web/package-lock.json','web/vite.config.js',
                 'requirements.txt','requirements-app.txt','requirements-phase7.txt','requirements-phase7.lock.txt','requirements-release.lock.txt',
                 'requirements-report.txt','requirements-authoring.lock.txt','output/phase5_archive_verification.json','output/phase6-tests.xml',
                 'output/phase6-react-tests.xml','output/phase7-tests.xml','output/phase8-tests.xml','output/phase8-clean-tests.xml','phase7_results.zip','phase7_results.zip.sha256'):
        add(archive_path(root,name) if name.endswith(('.zip','.zip.sha256')) else root/name,name)
    for name,sha in MODEL_ARCHIVES.items():
        path=archive_path(root,name)
        if digest(path)!=sha:raise ValueError('Model archive changed: '+name)
        add(path,name)
    if json.loads((root/'web/dist/build_inputs.json').read_text())!=fingerprint(root/'web'):
        raise ValueError('Compiled frontend is stale')
    required=['Setup_CrackLab.ps1','requirements-release.lock.txt','release_assets/source_replay/manifest.json','release_assets/demo/manifest.json','web/dist/index.html']
    if not candidate:required+=['MODEL_CARD.md','START_HERE.md','docs/FINAL_REPORT.md','output/pdf/CrackLab_Final_Report.pdf','output/phase8/evidence.json','output/phase8/release_validation.json','output/phase8/demo/demo.json']
    if any(name not in selected for name in required):raise ValueError('Missing required release files: '+str([n for n in required if n not in selected]))
    entries={name:{'sha256':digest(path),'bytes':path.stat().st_size} for name,path in sorted(selected.items())}
    manifest={'version':'1.0.0-rc1' if candidate else '1.0.0-pilot','phase':8,'scope':'Local research/education application release; not scientific confirmation of every target.',
              'files':entries,'privacy':'Main user workspace, credentials, venv, browser binaries and node_modules are not included.',
              'runtime_requirements':'Windows Python 3.12; network for first dependency install; bundled trained model archives and compiled frontend.'}
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for name,path in sorted(selected.items()):
            z.write(path,name,compress_type=zipfile.ZIP_STORED if path.suffix in ('.zip','.webm','.png','.jpg','.pdf') else zipfile.ZIP_DEFLATED)
        z.writestr(MANIFEST,json.dumps(manifest,indent=2))
    result=verify_archive(target)
    target.with_suffix(target.suffix+'.sha256').write_text(digest(target)+'  '+target.name+'\n')
    print(json.dumps(dict(result,archive=str(target),sha256=digest(target),archive_bytes=target.stat().st_size),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'results/releases/CrackLab_Phase8_v1.0.0.zip');parser.add_argument('--candidate',action='store_true')
    args=parser.parse_args();build(Path(__file__).resolve().parents[1],args.output,args.candidate)
