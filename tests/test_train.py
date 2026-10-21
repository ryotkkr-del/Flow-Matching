import sys
import pytest
import torch
from torch.utils.data import TensorDataset
from ai.core import TinyUNet
from ai import train


@pytest.mark.parametrize('requested,expected',[(None,.003),(.0001,.0001)])
def test_resume_preserves_or_explicitly_overrides_learning_rate(monkeypatch,tmp_path,requested,expected):
    # 一時ディレクトリのテスト用モデル。展示用checkpointとしては配布しない。
    m=TinyUNet(base=16);opt=torch.optim.AdamW(m.parameters(),lr=.003)
    initial=tmp_path/'initial.pt';out=tmp_path/'smoke.pt'
    torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),
                'model_config':{'base':16,'time_dim':64},'epoch':0,'loss_history':[]},initial)
    fake=TensorDataset(torch.zeros(2,1,28,28),torch.zeros(2,dtype=torch.long))
    monkeypatch.setattr(train.datasets,'MNIST',lambda *a,**k:fake)
    args=['train','--resume',str(initial),'--out',str(out),'--epochs','1','--batch-size','2','--threads','2','--max-batches','1','--offline']
    if requested is not None:args+=['--lr',str(requested)]
    monkeypatch.setattr(sys,'argv',args);train.main()
    result=torch.load(out,map_location='cpu',weights_only=True)
    assert result['optimizer']['param_groups'][0]['lr']==expected
    assert result['smoke_only'] is True
    assert result['samples_per_epoch']==2

def test_ema_is_a_weighted_average_and_resume_keeps_both_models(monkeypatch,tmp_path):
    m=TinyUNet(base=16);opt=torch.optim.AdamW(m.parameters(),lr=.0002)
    initial=tmp_path/'initial.pt';out=tmp_path/'ema.pt'
    raw={k:v.clone() for k,v in m.state_dict().items()}
    torch.save({'model':raw,'optimizer':opt.state_dict(),
                'model_config':{'base':16,'time_dim':64},'epoch':0,'loss_history':[]},initial)
    monkeypatch.setattr(train.datasets,'MNIST',lambda *a,**k:TensorDataset(torch.zeros(2,1,28,28),torch.zeros(2,dtype=torch.long)))
    args=['train','--resume',str(initial),'--out',str(out),'--epochs','1','--batch-size','2','--threads','2','--max-batches','1','--offline','--ema-decay','.9']
    monkeypatch.setattr(sys,'argv',args);train.main()
    c=torch.load(out,map_location='cpu',weights_only=True)
    assert c['ema_updates']==1 and c['ema_start_epoch']==1 and c['ema_decay']==.9
    for k in raw:torch.testing.assert_close(c['ema'][k],raw[k]*.9+c['model'][k]*.1)
    # EMA指定を省略して再開しても、rawのoptimizerとEMAの両方を継続。
    args=['train','--resume',str(out),'--out',str(out),'--epochs','2','--batch-size','2','--threads','2','--max-batches','1','--offline']
    monkeypatch.setattr(sys,'argv',args);train.main()
    c2=torch.load(out,map_location='cpu',weights_only=True)
    assert c2['ema_updates']==2 and c2['ema_decay']==.9
    for k in raw:torch.testing.assert_close(c2['ema'][k],c['ema'][k]*.9+c2['model'][k]*.1)
