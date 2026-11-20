"""Wait for the active download queue, then audit all six ZIPs without extraction."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
status=ROOT/'output/multisource_downloads/queue_status.json'
output=ROOT/'output/multisource_full_v1/manifest.json'
while True:
    state=json.loads(status.read_text())['state']
    if state=='all_archives_downloaded_and_verified': break
    if state in ('paused_after_error','blocked_disk_space'):
        raise RuntimeError('Download queue needs attention: '+state)
    time.sleep(30)
output.parent.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,'-u','scripts/data/prepare_multisource.py','--full',
                '--root','data/external/multisource_cracks_v1','--output',str(output)],cwd=ROOT,check=True)
