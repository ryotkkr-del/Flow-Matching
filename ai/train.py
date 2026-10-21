"""python -m ai.train --epochs 20 (初回のみMNISTをダウンロード)"""
import argparse
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .core import TinyUNet, sample_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--base', type=int, choices=[16,32,48,64], default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--bf16', action='store_true', help='任意の混合精度高速化。重み・損失はfloat32のまま')
    p.add_argument('--stop-file', help='このファイルが存在したら次のepoch保存後に停止')
    p.add_argument('--data', default='data')
    p.add_argument('--out', default='checkpoints/mnist.pt')
    p.add_argument('--resume')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--max-batches', type=int, default=0, help='動作確認専用。品質確認用学習には使わない')
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.lr <= 0 or a.threads < 1: p.error('正の値を指定してください')
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if a.bf16 and device == 'cuda' and not torch.cuda.is_bf16_supported():
        p.error('このGPUはbf16に対応していません。--bf16を外してください')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,),(.5,))])
    dataset = datasets.MNIST(a.data, train=True, download=not a.offline, transform=transform)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True, num_workers=0,
                        pin_memory=device=='cuda')
    config = {'base': a.base, 'time_dim':64}
    saved = torch.load(a.resume, map_location='cpu', weights_only=True) if a.resume else None
    if saved: config = saved.get('model_config', {'base':32,'time_dim':64})
    model = TinyUNet(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    history, start = [], 0
    if saved:
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        history = saved['loss_history']; start = saved['epoch']
        if 'rng' in saved: torch.set_rng_state(saved['rng'])
        if device=='cuda' and saved.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(saved['cuda_rng'])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    print(f'train: {start} -> {a.epochs} epochs / {device} / bf16={a.bf16}', flush=True)
    for epoch in range(start, a.epochs):
        epoch_started = time.perf_counter()
        model.train(); total, seen = 0., 0
        for index, (x1, _) in enumerate(loader):
            x1 = x1.to(device)
            t, xt, target = sample_path(x1)
            # 高速化は演算精度の指定だけ。確率経路・予測対象・MSEは同じ。
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=a.bf16):
                predicted = model(xt, t)
                loss = F.mse_loss(predicted, target)
            if not torch.isfinite(loss): raise ValueError('lossが非有限値になりました')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * len(x1); seen += len(x1)
            if (index + 1) % 100 == 0:
                print(f'epoch {epoch+1} batch {index+1}/{len(loader)} loss={total/seen:.6f}', flush=True)
            if a.max_batches and index + 1 >= a.max_batches: break
        history.append(total / seen)
        data = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                'model_config':config, 'epoch':epoch+1, 'loss_history':history,
                'rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None,
                'seed':a.seed, 'samples_per_epoch':seen, 'smoke_only':bool(a.max_batches),
                'normalization':'2*x-1', 'training_args':vars(a),
                'precision': 'bf16-autocast' if a.bf16 else 'float32',
                'epoch_seconds': time.perf_counter()-epoch_started}
        torch.save(data, a.out+'.tmp'); os.replace(a.out+'.tmp', a.out)
        Path(a.out+'.history.json').write_text(json.dumps(history))
        print(f'epoch {epoch+1}/{a.epochs} loss={history[-1]:.6f} samples={seen} device={device} seconds={data["epoch_seconds"]:.1f}', flush=True)
        if a.stop_file and Path(a.stop_file).exists():
            print('停止要求を確認。checkpoint保存済み。', flush=True)
            break

if __name__ == '__main__': main()
