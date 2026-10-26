import importlib
import json
from pathlib import Path
import numpy as np
import torch
from ai.core import TinyUNet,sample_path
from ai.trajectory import generate_path,payload,validate_payload,load_model,initial_noise

torch.set_num_threads(2)

def test_original_core_preserved():
    n=json.loads(Path('notebooks/original.ipynb').read_text(encoding='utf-8'))
    source=Path('ai/core.py').read_text(encoding='utf-8')
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
    with pytest.raises(ValueError):validate_payload(d)

def test_initial_noise_step_comparison_and_repeatability():
    a,_=generate_path(Constant(),42,4,10);b,_=generate_path(Constant(),42,4,20)
    torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
    c,_=generate_path(Constant(),42,4,10);assert torch.equal(a,c)

def test_checkpoint_roundtrip(tmp_path):
    p=tmp_path/'m.pt';m=TinyUNet();torch.save({'model':m.state_dict(),'model_config':{'base':32,'time_dim':64}},p)
    loaded,info=load_model(p)
    for a,b in zip(m.parameters(),loaded.parameters()):assert torch.equal(a,b)


def test_loaded_hash_matches_loaded_weights_if_file_is_replaced(tmp_path):
    import hashlib
    from unittest.mock import patch
    p=tmp_path/'changing.pt';first=TinyUNet(base=16);second=TinyUNet(base=16)
    torch.save({'model':first.state_dict(),'model_config':{'base':16}},p)
    raw=p.read_bytes();real_load=torch.load
    def replacing_load(*args,**kwargs):
        torch.save({'model':second.state_dict(),'model_config':{'base':16}},p)
        return real_load(*args,**kwargs)
    with patch('ai.trajectory.torch.load',side_effect=replacing_load):loaded,info=load_model(p)
    assert info['_source_sha256']==hashlib.sha256(raw).hexdigest()
    assert info['_source_sha256']!=hashlib.sha256(p.read_bytes()).hexdigest()
    for a,b in zip(first.parameters(),loaded.parameters()):assert torch.equal(a,b)


def test_bf16_loss_and_gradients_remain_finite():
    torch.manual_seed(7);m=TinyUNet(base=16);x=torch.rand(2,1,28,28)*2-1
    t,xt,target=sample_path(x)
    fp32=(m(xt,t)-target).square().mean()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        loss=torch.nn.functional.mse_loss(m(xt,t),target)
    assert abs(loss.item()-fp32.item())/fp32.item()<.005
    loss.backward()
    assert all(p.dtype==torch.float32 and p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

def test_checkpoint_can_select_ema_or_raw_weights(tmp_path):
    p=tmp_path/'both.pt';raw=TinyUNet(base=16);ema=TinyUNet(base=16)
    torch.save({'model':raw.state_dict(),'ema':ema.state_dict(),'model_config':{'base':16}},p)
    a,info=load_model(p);assert info['_weights_source']=='ema'
    b,info_raw=load_model(p,use_ema=False);assert info_raw['_weights_source']=='raw'
    assert info['_source_sha256']==info_raw['_source_sha256']
    for x,y in zip(a.parameters(),ema.parameters()):assert torch.equal(x,y)
    for x,y in zip(b.parameters(),raw.parameters()):assert torch.equal(x,y)


def test_progress_callback_emits_real_euler_states():
    frames=[]
    states,velocities=generate_path(Constant(),seed=31,count=2,steps=7,
                                     on_frame=lambda step,x:frames.append((step,x.clone())))
    assert [k for k,_ in frames]==list(range(8))
    for k,live in frames:
        torch.testing.assert_close(live,states[k],rtol=0,atol=0)
    torch.testing.assert_close(states[1:]-states[:-1],velocities/7)


def test_preview_noise_is_identical_to_true_inference_initial_state():
    for seed,count in [(42,4),(777,1),(2147483647,4)]:
        preview=initial_noise(seed,count)
        states,_=generate_path(Constant(),seed=seed,count=count,steps=5)
        torch.testing.assert_close(preview,states[0],rtol=0,atol=0)
