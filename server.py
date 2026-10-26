"""ローカル専用。学習なし・ダウンロードなし。単一推論workerと上限付きcache。"""
import gzip
import json
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
app = FastAPI(title='Flow Matching — 白鷺祭')
executor = ThreadPoolExecutor(max_workers=1)
lock = threading.Lock()
jobs = OrderedDict()
cache = OrderedDict()
cache_bytes = 0
MAX_CACHE_BYTES = 96 * 1024 * 1024
model = None
info = {}
model_sha = None
load_error = None
checkpoint = Path(os.getenv('FLOW_CHECKPOINT',str(ROOT/'checkpoints/mnist.pt')))
if not checkpoint.is_absolute(): checkpoint=ROOT/checkpoint

def initialize():
    global model, info, model_sha, load_error
    if not checkpoint.exists(): return
    try:
        import torch
        from ai.trajectory import load_model, fingerprint
        torch.set_num_threads(int(os.getenv('FLOW_THREADS','4')))
        model, info = load_model(checkpoint)
        if info.get('smoke_only'): raise ValueError('動作確認用モデルです。本学習を実行してください。')
        if type(info.get('epoch')) is not int or info['epoch']<1:raise ValueError('学習回数の記録がないモデルです。学習済みcheckpointを指定してください。')
        model_sha = info['_source_sha256']
    except Exception as e:
        model=None; load_error=f'{type(e).__name__}: {e}'

initialize()

class Request(BaseModel):
    seed: int = Field(default=42,ge=0,le=2147483647)
    steps: int = Field(default=50,ge=5,le=100)
    count: int = Field(default=4,ge=1,le=4)

@app.get('/api/noise')
def noise(seed:int=Query(default=42,ge=0,le=2147483647),
          count:int=Query(default=4,ge=1,le=4)):
    if model is None:
        raise HTTPException(503,'リアルタイム生成には学習済みモデルが必要です。')
    from ai.trajectory import initial_noise
    # No model forward pass. The same x0 is later used in /api/generate.
    return {'seed':seed, 'count':count, 'pixels':initial_noise(seed,count).flatten(1).tolist()}


@app.get('/api/status')
def status():
    return {'ready':model is not None, 'checkpoint':checkpoint.name, 'checkpoint_sha256':model_sha,
            'epoch':info.get('epoch'), 'weights':info.get('_weights_source','raw'), 'device':'cpu', 'error':load_error,
            'demos':[p.name for p in sorted((ROOT/'demos').glob('*.json.gz'))]}

def pack(data):
    return gzip.compress(json.dumps(data,separators=(',',':'),allow_nan=False).encode(),compresslevel=1)

def work(key, request):
    global cache_bytes
    try:
        from ai.trajectory import generate_path,payload
        def publish(step, frame):
            # UIは直近の実計算フレームだけを取得する。古いフレームの大量送信を避ける。
            preview = frame.flatten(1).tolist()
            with lock:
                current = jobs.get(key)
                if current is not None and current['state'] == 'running':
                    current['step'] = step
                    current['preview'] = preview
        x,v=generate_path(model,request.seed,request.count,request.steps,on_frame=publish)
        data=pack(payload(x,v,seed=request.seed,steps=request.steps,checkpoint_sha256=model_sha,
                          epoch=info.get('epoch',0),loss_history=info.get('loss_history'),weights=info.get('_weights_source','raw')))
        if len(data)>MAX_CACHE_BYTES:
            raise ValueError('生成経路がキャッシュ上限を超えました。枚数かステップを減らしてください。')
        with lock:
            replaced=cache.pop(key,None)
            if replaced is not None:cache_bytes-=len(replaced)
            while cache and cache_bytes+len(data)>MAX_CACHE_BYTES:
                _, old=cache.popitem(last=False); cache_bytes-=len(old)
            cache[key]=data;cache_bytes+=len(data)
            jobs[key]={'state':'ready'}
    except Exception as e:
        with lock: jobs[key]={'state':'error','error':str(e)}

@app.post('/api/generate')
def generate(request:Request):
    if model is None: raise HTTPException(503,'チェックポイントがありません。READMEの学習・配置手順を実行してください。')
    key=f'{request.seed}-{request.steps}-{request.count}'
    with lock:
        if key in cache:
            cache.move_to_end(key); return {'id':key,'state':'ready'}
        if key in jobs and jobs[key]['state']=='running': return {'id':key,'state':'running'}
        if any(j['state']=='running' for j in jobs.values()): raise HTTPException(429,'生成中です。完了後にお試しください。')
        jobs[key]={'state':'running','step':-1,'total':request.steps,'preview':None}
        while len(jobs)>64: jobs.popitem(last=False)
    executor.submit(work,key,request)
    return {'id':key,'state':'running'}

@app.get('/api/jobs/{key}')
def job(key:str, after:int=Query(default=-1,ge=-1,le=100)):
    with lock:
        if key in cache: return {'state':'ready'}
        if key not in jobs: raise HTTPException(404,'生成履歴がありません')
        result=jobs[key]
        if result['state']=='ready': raise HTTPException(410,'キャッシュが削除されました。生成し直してください。')
        if result['state']=='running':
            # after以降に新しいフレームができた場合のみ返す。
            return {'state':'running','step':result['step'],'total':result['total'],
                    'preview':result['preview'] if result['step']>after else None}
        return {'state':'error','error':result.get('error','生成に失敗しました')}

@app.get('/api/paths/{key}')
def path(key:str):
    with lock:
        if key not in cache: raise HTTPException(404,'生成し直してください')
        data=cache[key]; cache.move_to_end(key)
    return Response(data,media_type='application/json',headers={'Content-Encoding':'gzip'})

@app.get('/api/demos/{name}')
def demo(name:str):
    # ファイル名を一覧と照合しパストラバーサルを防止
    if name not in status()['demos']: raise HTTPException(404)
    return Response((ROOT/'demos'/name).read_bytes(),media_type='application/json',headers={'Content-Encoding':'gzip'})

@app.get('/')
def index(): return FileResponse(ROOT/'web/index.html')
app.mount('/web',StaticFiles(directory=ROOT/'web'),name='web')
