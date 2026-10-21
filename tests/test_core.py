import importlib
import json
from pathlib import Path
import numpy as np
import torch
from ai.core import TinyUNet,sample_path
from ai.trajectory import generate_path,payload,validate_payload,load_model

torch.set_num_threads(2)

def test_original_core_preserved():
    n=json.loads(Path('notebooks/original.ipynb').read_text())
    source=Path('ai/core.py').read_text()
    assert ''.join(n['cells'][4]['source']).split('# 数学と形')[0] in source
    assert ''.join(n['cells'][6]['source']).split('model = TinyUNet')[0] in source

def test_sample_path_and_target():
    torch.manual_seed(1);x1=torch.rand(5,1,28,28)*2-1
    t,x,v=sample_path(x1);x0=x1-v
    torch.testing.assert_close(x,(1-t[:,None,None,None])*x0+t[:,None,None,None]*x1)
    assert t.shape==(5,) and ((t>=0)&(t<1)).all()

def test_model_backward_and_shape():
    m=TinyUNet(base=16);x=torch.rand(2,1,28,28)*2-1;t,xt,v=sample_path(x)
    pred=m(xt,t);assert pred.shape==x.shape
    loss=(pred-v).square().mean();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

class Constant(torch.nn.Module):
    def forward(self,x,t):return torch.full_like(x,7.)

def test_actual_euler_trajectory_and_no_clamp():
    x,v=generate_path(Constant(),seed=12,count=2,steps=10)
    torch.testing.assert_close(x[1:]-x[:-1],v/10)
    torch.testing.assert_close(x[-1],x[0]+7)
    assert x[-1].max()>1
    d=payload(x,v,seed=12,steps=10,checkpoint_sha256='a'*64,epoch=1)
    validate_payload(d)
    d['states'][1][0][0]+=1
    import pytest
    with pytest.raises(AssertionError):validate_payload(d)

def test_initial_noise_step_comparison_and_repeatability():
    a,_=generate_path(Constant(),42,4,10);b,_=generate_path(Constant(),42,4,20)
    torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
    c,_=generate_path(Constant(),42,4,10);assert torch.equal(a,c)

def test_checkpoint_roundtrip(tmp_path):
    p=tmp_path/'m.pt';m=TinyUNet();torch.save({'model':m.state_dict(),'model_config':{'base':32,'time_dim':64}},p)
    loaded,info=load_model(p)
    for a,b in zip(m.parameters(),loaded.parameters()):assert torch.equal(a,b)
