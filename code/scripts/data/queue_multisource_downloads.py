"""Download the remaining archives sequentially with a durable per-file ledger."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/external/multisource_cracks_v1'
OUT=ROOT/'output/multisource_downloads'
FILES=[('02_ground_iphone.zip',1637391230),('06_uav_other.zip',1906268391),
       ('03_uav_low_3_10m.zip',2307063695),('04_uav_mid_15_30m.zip',3445893965),
       ('05_uav_high_35_100m.zip',4357800972)]


def status(state,**fields):
    value=dict(state=state,pid=os.getpid(),checked_at_utc=datetime.now(timezone.utc).isoformat(),**fields)
    temporary=OUT/'queue_status.json.tmp'
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(OUT/'queue_status.json')
    print(json.dumps(value),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'queue.lock').open('a+b') as lock:
        if lock.tell()==0: lock.write(b'0'); lock.flush()
        lock.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=sum(max(0,size-(DATA/(name+'.part')).stat().st_size) if (DATA/(name+'.part')).exists() else size
                    for name,size in FILES if not (DATA/name).exists())
        free=shutil.disk_usage(DATA).free
        if free<pending+2*1024**3:
            status('blocked_disk_space',free_bytes=free,pending_bytes=pending,reserve_bytes=2*1024**3)
            raise RuntimeError('Need room for remaining archives plus 2 GiB reserve')
        completed=[]
        for name,size in FILES:
            status('downloading',archive=name,completed_this_queue=completed,remaining_total_bytes=pending)
            result=subprocess.run([sys.executable,'-u','scripts/data/download_multisource_cracks.py',
                                   '--root',str(DATA),'--files',name],cwd=ROOT)
            if result.returncode:
                status('paused_after_error',archive=name,completed_this_queue=completed,returncode=result.returncode)
                return result.returncode
            ledger=json.loads((DATA/'DOWNLOAD_STATUS.json').read_text())
            entry=next(r for r in ledger['files'] if r['file']==name)
            assert entry.get('sha256_verified') is True
            completed.append(name)
        status('all_archives_downloaded_and_verified',completed_this_queue=completed)
    return 0


if __name__=='__main__': sys.exit(main())
