"""実モデルで繰り返し生成し、ASGI API・cache上限を時間をかけて確認する。
ブラウザ描画・物理的なオフライン・Windows実機確認を代替しない。
"""
import argparse
import gc
import json
try:
    import resource
except ImportError:  # Windowsではcache上限は検査し、OS側のRSSは未計測にする。
    resource=None
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
import server
from ai.trajectory import validate_payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seconds',type=float,default=900)
    p.add_argument('--interval',type=float,default=10)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--out',default='docs/soak-result.json')
    a=p.parse_args()
    if a.seconds<=0 or not 0<a.interval<=60:p.error('seconds>0, 0<interval<=60を指定')
    assert server.model is not None,'実checkpointを配置してください'
    client=TestClient(server.app)
    started=time.perf_counter();count=0;times=[];peak_cache=0;peak_jobs=0;error=None
    try:
        while time.perf_counter()-started<a.seconds:
            iteration=time.perf_counter()
            request={'seed':10000+count,'steps':a.steps,'count':4}
            response=client.post('/api/generate',json=request);response.raise_for_status();key=response.json()['id']
            for _ in range(6000):
                r=client.get('/api/jobs/'+key);r.raise_for_status();state=r.json()
                if state['state']!='running':break
                time.sleep(.02)
            assert state['state']=='ready',state
            response=client.get('/api/paths/'+key);response.raise_for_status()
            data=validate_payload(response.json())
            assert data['checkpoint_sha256']==server.model_sha
            assert data['seed']==request['seed'] and data['count']==4
            hit=client.post('/api/generate',json=request);hit.raise_for_status();assert hit.json()['state']=='ready'
            peak_cache=max(peak_cache,server.cache_bytes);peak_jobs=max(peak_jobs,len(server.jobs))
            assert server.cache_bytes<=server.MAX_CACHE_BYTES
            assert len(server.jobs)<=64
            assert server.cache_bytes==sum(len(v) for v in server.cache.values())
            times.append(time.perf_counter()-iteration);count+=1
            del data,response
            if count%10==0:
                gc.collect();print(f'PASS {count} generated / elapsed {time.perf_counter()-started:.0f}s / cache {server.cache_bytes/1048576:.1f}MiB',flush=True)
            time.sleep(max(0,min(a.interval-(time.perf_counter()-iteration),a.seconds-(time.perf_counter()-started))))
    except BaseException as e:
        error=f'{type(e).__name__}: {e}'
        raise
    finally:
        result={'result':'PASS' if error is None else 'FAIL','error':error,
                'elapsed_seconds':round(time.perf_counter()-started,3),'generated_requests':count,
                'cache_hit_checks':count,'checkpoint_sha256':server.model_sha,'epoch':server.info.get('epoch'),
                'steps':a.steps,'count':4,'peak_cache_bytes':peak_cache,'cache_limit_bytes':server.MAX_CACHE_BYTES,
                'peak_jobs':peak_jobs,'peak_process_rss_kib':(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024 if sys.platform=='darwin' else 1)) if resource else None,
                'request_seconds_median':sorted(times)[len(times)//2] if times else None,
                'note':'ASGI TestClient with the real model. Not a browser or physical offline test.'}
        path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
