"""ローカル専用。学習なし・ダウンロードなし。単一推論workerと上限付きcache。"""
import gzip
import json
import os
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from fastapi import FastAPI, HTTPException
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
        model_sha = fingerprint(checkpoint)
    except Exception as e:
        model=None; load_error=f'{type(e).__name__}: {e}'

initialize()

class Request(BaseModel):
    seed: int = Field(default=42,ge=0,le=2147483647)
    steps: int = Field(default=50,ge=5,le=100)
    count: int = Field(default=4,ge=1,le=4)

@app.get('/api/status')
def status():
    return {'ready':model is not None, 'checkpoint':checkpoint.name, 'checkpoint_sha256':model_sha,
            'epoch':info.get('epoch'), 'device':'cpu', 'error':load_error,
            'demos':[p.name for p in sorted((ROOT/'demos').glob('*.json.gz'))]}

def pack(data):
    return gzip.compress(json.dumps(data,separators=(',',':'),allow_nan=False).encode(),compresslevel=1)

def work(key, request):
    global cache_bytes
    try:
        from ai.trajectory import generate_path,payload
        x,v=generate_path(model,request.seed,request.count,request.steps)
        data=pack(payload(x,v,seed=request.seed,steps=request.steps,checkpoint_sha256=model_sha,
                          epoch=info.get('epoch',0),loss_history=info.get('loss_history')))
        with lock:
            cache[key]=data; cache_bytes+=len(data)
            while cache_bytes>MAX_CACHE_BYTES and len(cache)>1:
                _, old=cache.popitem(last=False); cache_bytes-=len(old)
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
        jobs[key]={'state':'running'}
        while len(jobs)>64: jobs.popitem(last=False)
    executor.submit(work,key,request)
    return {'id':key,'state':'running'}

@app.get('/api/jobs/{key}')
def job(key:str):
    with lock:
        if key in cache: return {'state':'ready'}
        if key not in jobs: raise HTTPException(404,'生成履歴がありません')
        result=jobs[key]
        if result['state']=='ready': raise HTTPException(410,'キャッシュが削除されました。生成し直してください。')
        return result

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
