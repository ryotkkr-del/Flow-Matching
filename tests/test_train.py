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
