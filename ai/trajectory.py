"""実際のEuler軌道。表示用clampを計算状態に戻さない。"""
import hashlib
import json
from pathlib import Path
import torch
from .core import TinyUNet


def load_model(path, device='cpu'):
    data = torch.load(path, map_location=device, weights_only=True)
    model = TinyUNet(**data.get('model_config', {})).to(device)
    model.load_state_dict(data['model'])
    model.eval()
    return model, data


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def generate_path(model, seed=42, count=4, steps=50, device='cpu'):
    # CPUで初期ノイズを作り、step数を変えても完全に同じx0を使う。
    rng = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(count, 1, 28, 28, generator=rng).to(device)
    states = [x.cpu().clone()]
    velocities = []
    dt = 1.0 / steps
    model.eval()
    for k in range(steps):
        t = torch.full((count,), k / steps, device=device)
        v = model(x, t)
        x = x + dt * v
        if not torch.isfinite(x).all():
            raise ValueError('非有限値が発生しました。チェックポイントを確認してください。')
        velocities.append(v.cpu().clone())
        states.append(x.cpu().clone())
    return torch.stack(states), torch.stack(velocities)


def payload(states, velocities, *, seed, steps, checkpoint_sha256, epoch, loss_history=None):
    # [time, image, 784]。float32の実計算値を保持（非量子化）。
    return {'format': 1, 'method': 'euler', 'seed': seed, 'steps': steps,
            'count': states.shape[1], 'times': [k / steps for k in range(steps + 1)],
            'checkpoint_sha256': checkpoint_sha256, 'epoch': epoch,
            'loss_history': loss_history or [],
            'states': states.flatten(2).tolist(),
            'velocities': velocities.flatten(2).tolist()}


def validate_payload(data):
    import numpy as np
    steps, count = data['steps'], data['count']
    if not (1 <= steps <= 200 and 1 <= count <= 8):
        raise ValueError('軌道のサイズが範囲外です')
    x = np.asarray(data['states'], dtype=np.float32)
    v = np.asarray(data['velocities'], dtype=np.float32)
    assert x.shape == (steps + 1, count, 784)
    assert v.shape == (steps, count, 784)
    assert np.isfinite(x).all() and np.isfinite(v).all()
    assert data['method'] == 'euler' and len(data['checkpoint_sha256']) == 64
    assert np.allclose(data['times'], np.linspace(0, 1, steps + 1))
    assert np.allclose(x[1:] - x[:-1], v / steps, atol=2e-6, rtol=2e-5)
    return data
