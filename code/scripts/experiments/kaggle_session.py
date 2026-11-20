"""Jupyter bridge. Connection URL is read from stdin, never persisted or printed."""
import argparse
import json
import sys
import urllib.request
import urllib.error


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--code-file')
    p.add_argument('--upload')
    p.add_argument('--remote-name')
    p.add_argument('--download')
    p.add_argument('--output')
    p.add_argument('--fetch-latest-recovery',action='store_true')
    p.add_argument('--recovery-prefix',default='milano_recovery_',help='Export filename prefix for --fetch-latest-recovery')
    p.add_argument('--capture',help='Save remote execution output locally, excluding connection input')
    p.add_argument('--watch-recovery',action='store_true')
    p.add_argument('--poll-seconds',type=int,default=60)
    p.add_argument('--completion-path',default='cracklab_milano_v1/artifacts/comparison.json')
    p.add_argument('--completion-state',help='Require this JSON state before stopping a recovery watch')
    a=p.parse_args()
    if sys.stdin.isatty():
        import getpass
        base=getpass.getpass('Kaggle connection (hidden): ').strip().rstrip('/')
    else:
        base=sys.stdin.readline().strip().rstrip('/')
    if not base.startswith('https://kkb-production.jupyter-proxy.kaggle.net/'):
        raise ValueError('Expected Kaggle Jupyter connection')
    def request(path, data=None, method=None):
        body=json.dumps(data).encode() if data is not None else None
        req=urllib.request.Request(base+path,data=body,method=method,
                                   headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=30) as r:
            return None if r.status==204 else json.load(r)
    if a.watch_recovery:
        import subprocess
        import time
        from pathlib import Path
        output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
        errors=0
        while errors<30:
            try:
                entries=request('/api/contents?content=1')['content']
                names={r['name'] for r in entries}
                backups=[r for r in entries if r['name'].startswith(a.recovery_prefix)
                         and r['name'].endswith('.zip') and r['name']+'.sha256' in names]
                latest=max(backups,key=lambda r:r['last_modified']) if backups else None
                if latest and not (output/latest['name']).exists():
                    child=subprocess.run([sys.executable,__file__,'--fetch-latest-recovery',
                        '--output',str(output),'--recovery-prefix',a.recovery_prefix],input=base+'\n',text=True,capture_output=True)
                    print(child.stdout.strip(),flush=True)
                    if child.returncode: raise RuntimeError('Recovery download failed')
                state={'checked_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                       'state':'watching','latest_verified_backup':latest['name'] if latest else None}
                (output.parent/'backup_sync_status.json').write_text(json.dumps(state,indent=2))
                try:
                    comparison=request('/api/contents/'+a.completion_path+'?content='+('1' if a.completion_state else '0'))
                    if comparison and a.completion_state:
                        if json.loads(comparison['content']).get('state')!=a.completion_state:
                            comparison=None
                except urllib.error.HTTPError as exc:
                    if exc.code!=404: raise
                    comparison=None
                if comparison and latest and latest['last_modified']>=comparison['last_modified']:
                    state['state']='completed_final_backup_downloaded'
                    (output.parent/'backup_sync_status.json').write_text(json.dumps(state,indent=2))
                    print('Completion confirmed; final recovery downloaded',flush=True)
                    return
                errors=0
            except Exception as exc:
                errors+=1
                print('Recovery sync retry:',type(exc).__name__,errors,flush=True)
            time.sleep(max(10,a.poll_seconds))
        raise RuntimeError('Recovery sync stopped after repeated connection failures')
    if a.fetch_latest_recovery:
        from pathlib import Path
        import hashlib
        import shutil
        from urllib.parse import quote
        entries=request('/api/contents?content=1')['content']
        available_names={row['name'] for row in entries}
        names=sorted(row['name'] for row in entries
                     if row['name'].startswith(a.recovery_prefix) and row['name'].endswith('.zip')
                     and row['name']+'.sha256' in available_names)
        if not names: raise ValueError('No exported recovery ZIP yet')
        # File timestamps, rather than architecture-prefixed lexical ordering.
        selected=max((r for r in entries if r['name'] in names),key=lambda r:r['last_modified'])['name']
        output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
        for name in (selected+'.sha256',selected):
            target=output/name
            if target.exists(): raise ValueError('Recovery output exists')
            temporary=target.with_name(target.name+'.partial')
            with urllib.request.urlopen(base+'/files/'+quote(name),timeout=60) as r, temporary.open('wb') as f:
                shutil.copyfileobj(r,f)
        path=output/(selected+'.partial')
        expected=(output/(selected+'.sha256.partial')).read_text().split()[0]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Downloaded recovery checksum mismatch')
        import zipfile
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None: raise ValueError('Recovery ZIP CRC failed')
            for name in z.namelist():
                if name.endswith(('/latest.json','/best.json')):
                    pointer=json.loads(z.read(name))
                    member=str(Path(name).parent/pointer['file']).replace('\\','/')
                    if hashlib.sha256(z.read(member)).hexdigest()!=pointer['sha256']:
                        raise ValueError('Recovery pointer checksum mismatch')
        path.replace(output/selected)
        (output/(selected+'.sha256.partial')).replace(output/(selected+'.sha256'))
        print('Downloaded and verified recovery:',output/selected,flush=True)
        return
    if a.download:
        from pathlib import Path
        from urllib.parse import quote
        path=Path(a.output)
        if path.exists(): raise ValueError('Local output already exists')
        path.parent.mkdir(exist_ok=True,parents=True)
        with urllib.request.urlopen(base+'/files/'+quote(a.download,safe='/'),timeout=60) as r, path.open('wb') as f:
            import shutil
            shutil.copyfileobj(r,f)
        print('Downloaded',path.name,path.stat().st_size,'bytes')
        return
    if a.upload:
        import base64
        import hashlib
        import time
        from pathlib import Path
        path=Path(a.upload)
        name=a.remote_name or path.name
        if '/' in name or '\\' in name: raise ValueError('Use a plain remote filename')
        size=path.stat().st_size; transferred=0; chunk=1; start=time.monotonic()
        with path.open('rb') as f:
            while block:=f.read(4*1024**2):
                transferred+=len(block)
                request('/api/contents/'+name,{'type':'file','format':'base64',
                    'content':base64.b64encode(block).decode(),
                    'chunk':-1 if transferred==size else chunk},method='PUT')
                if chunk==1 or chunk%8==0 or transferred==size:
                    print(f'Uploaded {transferred}/{size} bytes in {time.monotonic()-start:.0f}s',flush=True)
                chunk+=1
        return
    if not a.code_file:
        status=request('/api/status')
        kernels=request('/api/kernels')
        print(json.dumps({'server_status':status,'kernels':kernels}))
        return
    import websocket
    import uuid
    kernel=request('/api/kernels',{'name':'python3'})
    session=uuid.uuid4().hex; msg_id=uuid.uuid4().hex
    ws=websocket.create_connection(base.replace('https://','wss://',1)+
        '/api/kernels/'+kernel['id']+'/channels?session_id='+session,timeout=60)
    message={'header':{'msg_id':msg_id,'username':'codex','session':session,
                       'msg_type':'execute_request','version':'5.3'},
             'parent_header':{},'metadata':{},'channel':'shell',
             'content':{'code':open(a.code_file,encoding='utf-8').read(),'silent':False,
                        'store_history':False,'user_expressions':{},'allow_stdin':False,
                        'stop_on_error':True}}
    ws.send(json.dumps(message))
    failed=False
    captured=[]
    try:
        while True:
            msg=json.loads(ws.recv())
            if msg.get('parent_header',{}).get('msg_id')!=msg_id: continue
            kind=msg.get('msg_type',msg.get('header',{}).get('msg_type'))
            content=msg['content']
            if kind=='stream':
                output=content.get('text',''); captured.append(output)
                print(output,end='',flush=True)
            elif kind=='error':
                print('REMOTE ERROR:',content.get('ename'),content.get('evalue')); failed=True
            elif kind=='execute_result': print(content.get('data',{}).get('text/plain',''))
            elif kind=='status' and content.get('execution_state')=='idle': break
    finally:
        ws.close()
        request('/api/kernels/'+kernel['id'],method='DELETE')
        if a.capture:
            from pathlib import Path
            target=Path(a.capture); target.parent.mkdir(exist_ok=True,parents=True)
            target.write_text(''.join(captured),encoding='utf-8')
    if failed: sys.exit(1)


if __name__=='__main__':
    try: main()
    except Exception as exc:
        # Never echo exception URLs or remote response bodies containing credentials.
        print('Connection operation failed:',type(exc).__name__,
              getattr(exc,'code',''),flush=True)
        if isinstance(exc,ModuleNotFoundError):
            print('Missing local module:',exc.name,'Interpreter:',sys.executable,flush=True)
        sys.exit(1)
