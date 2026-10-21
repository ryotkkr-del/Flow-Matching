import gzip
import json
import time
import torch
from fastapi.testclient import TestClient
import server
from ai.trajectory import validate_payload
from test_core import Constant


def test_missing_model_and_static(monkeypatch):
    monkeypatch.setattr(server,'model',None)
    c=TestClient(server.app)
    assert c.get('/').status_code==200
    assert c.get('/web/app.js').status_code==200
    assert not c.get('/api/status').json()['ready']
    assert c.post('/api/generate',json={}).status_code==503
    assert c.get('/api/demos/missing.json.gz').status_code==404


def test_async_generation_cache_validation_and_recovery(monkeypatch):
    # 定数モデルはテスト専用。デモとして保存・表示しない。
    monkeypatch.setattr(server,'model',Constant())
    monkeypatch.setattr(server,'model_sha','b'*64)
    monkeypatch.setattr(server,'info',{'epoch':1})
    server.jobs.clear();server.cache.clear();server.cache_bytes=0
    c=TestClient(server.app)
    assert c.post('/api/generate',json={'steps':0}).status_code==422
    assert c.post('/api/generate',json={'count':100}).status_code==422
    r=c.post('/api/generate',json={'steps':5,'count':1,'seed':456}).json()
    for _ in range(100):
        j=c.get('/api/jobs/'+r['id']).json()
        if j['state']!='running':break
        time.sleep(.02)
    assert j['state']=='ready'
    d=c.get('/api/paths/'+r['id']).json();validate_payload(d)
    r2=c.post('/api/generate',json={'steps':5,'count':1,'seed':456}).json()
    assert r2['state']=='ready'
    server.cache.clear()
    assert c.get('/api/jobs/'+r['id']).status_code==410


def test_busy_and_exception(monkeypatch):
    monkeypatch.setattr(server,'model',Constant())
    server.jobs.clear();server.cache.clear()
    server.jobs['test']={'state':'running'}
    c=TestClient(server.app)
    assert c.post('/api/generate',json={}).status_code==429
    server.jobs.clear()
    class Broken(torch.nn.Module):
        def forward(self,x,t):raise RuntimeError('test error')
    monkeypatch.setattr(server,'model',Broken())
    r=c.post('/api/generate',json={'seed':999,'steps':5}).json()
    for _ in range(100):
        j=c.get('/api/jobs/'+r['id']).json()
        if j['state']!='running':break
        time.sleep(.01)
    assert j['state']=='error'
    assert 'test error' in j['error']
    server.jobs.clear()
