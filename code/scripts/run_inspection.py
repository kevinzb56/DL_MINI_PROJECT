"""Prepare trusted models and serve the local Phase 6 application."""
import argparse
import os
from pathlib import Path
import sys
import shutil
import subprocess
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_inspection_models import prepare
from scripts.frontend_build import fingerprint


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    # Load local, gitignored demo credentials before preparing the server.
    secrets_file = root / '.env'
    if secrets_file.exists():
        for line in secrets_file.read_text(encoding='utf-8').splitlines():
            name, separator, value = line.partition('=')
            if separator and name.strip() and name.strip() not in os.environ:
                os.environ[name.strip()] = value.strip().strip('"\'')
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--workspace', type=Path, default=root / 'artifacts/phase6/workspace')
    args = parser.parse_args()
    web = root / 'web'
    entry = web / 'dist/index.html'
    sources = [web / 'index.html', web / 'package.json', web / 'vite.config.js',
               *list((web / 'src').rglob('*')), *list((web / 'public').rglob('*'))]
    inputs=web/'dist/build_inputs.json'
    stale=(json.loads(inputs.read_text())!=fingerprint(web)) if inputs.exists() else (not entry.exists() or any(path.is_file() and path.stat().st_mtime > entry.stat().st_mtime for path in sources))
    if not entry.exists() or stale:
        npm = shutil.which('npm.cmd' if os.name == 'nt' else 'npm')
        if not npm or not (web / 'node_modules').exists():
            raise SystemExit('React setup needed: open web/ in a terminal and run npm ci, then npm run build. Node.js 22.12+ is supported.')
        subprocess.run([npm, 'run', 'build'], cwd=web, check=True)
    prepare(root, args.workspace)
    os.environ['CRACK_WORKSPACE'] = str(args.workspace.resolve())
    import uvicorn
    print(f'CrackLab is ready at http://127.0.0.1:{args.port}', flush=True)
    uvicorn.run('inspection.api:app', host='127.0.0.1', port=args.port, workers=1)
