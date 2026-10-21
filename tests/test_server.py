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

def test_cache_eviction_byte_accounting_and_oversized_result(monkeypatch):
    from collections import OrderedDict
    monkeypatch.setattr(server,'model',Constant())
    monkeypatch.setattr(server,'model_sha','c'*64)
    monkeypatch.setattr(server,'info',{'epoch':1})
    monkeypatch.setattr(server,'cache',OrderedDict())
    monkeypatch.setattr(server,'jobs',OrderedDict())
    monkeypatch.setattr(server,'cache_bytes',0)
    monkeypatch.setattr(server,'MAX_CACHE_BYTES',110000)
    for i in range(8):
        server.work(str(i),server.Request(seed=i,count=1,steps=5))
        assert server.jobs[str(i)]['state']=='ready'
        assert server.cache_bytes==sum(len(x) for x in server.cache.values())
        assert server.cache_bytes<=server.MAX_CACHE_BYTES
    assert len(server.cache)<8
    server.work('7',server.Request(seed=7,count=1,steps=5))
    assert server.cache_bytes==sum(len(x) for x in server.cache.values())
    monkeypatch.setattr(server,'MAX_CACHE_BYTES',1)
    server.work('oversized',server.Request(seed=999,count=1,steps=5))
    assert server.jobs['oversized']['state']=='error'
    assert '上限' in server.jobs['oversized']['error']

def test_recorded_replay_does_not_require_torch(tmp_path):
    # 軽量再生を別プロセスで確認。fixtureはテスト用の応答で、展示デモとして配布しない。
    import os,subprocess,sys,textwrap
    demos=tmp_path/'demos';demos.mkdir()
    with gzip.open(demos/'test.json.gz','wt') as f:json.dump({'test_fixture':True},f)
    script=textwrap.dedent('''
        import builtins,sys
        original=builtins.__import__
        def guarded(name,*a,**k):
            if name.split('.')[0] in ('torch','torchvision','numpy'):
                raise AssertionError('Heavy AI dependency was imported: '+name)
            return original(name,*a,**k)
        builtins.__import__=guarded
        from pathlib import Path
        from fastapi.testclient import TestClient
        import server
        server.ROOT=Path(sys.argv[1])
        c=TestClient(server.app)
        assert c.get('/api/status').json()['ready'] is False
        assert c.get('/api/demos/test.json.gz').json()=={'test_fixture':True}
        assert c.post('/api/generate',json={}).status_code==503
    ''')
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path)],env={**os.environ,'FLOW_CHECKPOINT':str(tmp_path/'absent.pt')},capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
