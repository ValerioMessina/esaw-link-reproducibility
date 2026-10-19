"""Local-only immutable per-record inference; stop/restart preserves every attempt."""
from pathlib import Path
import json,time,hashlib,urllib.request,os,fcntl,subprocess,shutil,statistics,traceback
R=Path(__file__).resolve().parent;OUT=R/'results';P=json.loads((R/'protocol.json').read_text())
SSD=Path(P['ssd_directory']);lock=(R/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));os.replace(tmp,p)
def sync(p):
    q=SSD/p.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);tmp=q.with_suffix('.tmp');shutil.copy2(p,tmp);assert sha(tmp)==sha(p);os.replace(tmp,q)
def api(path,body=None,timeout=600):
    req=urllib.request.Request('http://127.0.0.1:11434'+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as f:return json.load(f)
def progress(**kw):
    save(R/'progress.json',{'time':time.time(),**kw});sync(R/'progress.json')
def main():
    assert SSD.parent.exists(),'SSD must be mounted'
    assert sha(Path(P['tasks_path']))==P['tasks_sha256']
    for n,h in P['source_sha256'].items():assert sha(Path(n))==h,n
    assert sha(Path.home()/'.ollama/models/manifests/registry.ollama.ai/library/gemma4/12b')==P['manifest_sha256']
    version=api('/api/version');assert version['version']=='0.34.0'
    tags=api('/api/tags');tag=next(m for m in tags['models'] if m['name']==P['model']);assert tag['digest']==P['manifest_sha256']
    save(R/'runtime.json',{'time':time.time(),'ollama':version,'model':tag,'platform':os.uname().sysname,'machine':os.uname().machine});sync(R/'runtime.json')
    subprocess.Popen(['caffeinate','-i','-m','-w',str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    tasks=[json.loads(l) for l in Path(P['tasks_path']).open() if l.strip()];assert len(tasks)==5000
    durations=[];done=0;started=time.time();protocol_sha=sha(R/'protocol.json')
    for t in tasks:
        assert hashlib.sha256(t['prompt'].encode()).hexdigest()==t['prompt_sha256']
        path=OUT/f"{t['pair_index']:05d}.json"
        body={'model':P['model'],'messages':[{'role':'user','content':t['prompt']}],'stream':False,'think':False,'format':'json','options':P['options']}
        body_sha=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
        if path.exists():
            old=json.loads(path.read_text());assert old['request_sha256']==body_sha and old['protocol_sha256']==protocol_sha
            sync(path);done+=1;continue
        progress(event='request_started',completed_records=done,total_records=5000,pair_index=t['pair_index'])
        now=time.time();row={'pair_index':t['pair_index'],'record_key':t['record_key'],'prompt_sha256':t['prompt_sha256'],'request_sha256':body_sha,'protocol_sha256':protocol_sha,'started_epoch':now,'think':False,'options':P['options']}
        try:row.update(response=api('/api/chat',body),transport_success=True)
        except Exception as e:row.update(transport_success=False,error=repr(e))
        row['elapsed_seconds']=time.time()-now;save(path,row);sync(path);durations.append(row['elapsed_seconds']);done+=1
        progress(event='request_completed',completed_records=done,total_records=5000,last_transport_success=row['transport_success'],elapsed_seconds=time.time()-started,mean_request_seconds=statistics.mean(durations),projected_remaining_hours=statistics.mean(durations)*(5000-done)/3600)
        # A transport failure is an infrastructure issue, not permission to burn through all cases.
        if not row['transport_success']:raise RuntimeError('Transport failure retained; stopped without retries. Investigate before resuming.')
    progress(event='completed',completed_records=done,total_records=5000)
    subprocess.run([P['python'],str(R/'analyze.py')],check=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(R/'failure.json',{'time':time.time(),'error':repr(e),'traceback':traceback.format_exc()});raise
