import math
import torch
from torch import nn
from torch.nn import functional as F

def sample_path(x1: torch.Tensor):
    # x1: (B,1,28,28)  →  tは各画像で1個だけ引く
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    t4 = t[:, None, None, None]  # (B,1,1,1) にして画像へbroadcast
    xt = (1 - t4) * x0 + t4 * x1
    target_velocity = x1 - x0
    return t, xt, target_velocity


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = (nn.Identity() if in_channels == out_channels
                     else nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x, temb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(temb)[:, :, None, None]
        h = self.norm2(self.conv2(F.silu(h)))
        return (self.skip(x) + h) / math.sqrt(2.0)


class TinyUNet(nn.Module):
    def __init__(self, base=32, time_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(32, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.enc1 = ResBlock(1, base, time_dim)          # 28 x 28
        self.enc2 = ResBlock(base, base * 2, time_dim)   # 14 x 14
        self.mid = ResBlock(base * 2, base * 4, time_dim)  # 7 x 7
        self.dec2 = ResBlock(base * 6, base * 2, time_dim) # 14 x 14
        self.dec1 = ResBlock(base * 3, base, time_dim)     # 28 x 28
        self.output = nn.Conv2d(base, 1, 3, padding=1)

    @staticmethod
    def time_features(t):
        # t : (B,)。連続時刻を32次元の特徴に変換する
        frequencies = torch.exp(torch.linspace(0, math.log(100), 16, device=t.device))
        phases = (2 * math.pi) * t[:, None] * frequencies[None, :]
        return torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)

    def forward(self, xt, t):
        temb = self.time_mlp(self.time_features(t))
        s1 = self.enc1(xt, temb)
        s2 = self.enc2(F.avg_pool2d(s1, 2), temb)
        h = self.mid(F.avg_pool2d(s2, 2), temb)
        h = F.interpolate(h, size=s2.shape[-2:], mode="nearest")
        h = self.dec2(torch.cat([h, s2], dim=1), temb)
        h = F.interpolate(h, size=s1.shape[-2:], mode="nearest")
        h = self.dec1(torch.cat([h, s1], dim=1), temb)
        return self.output(F.silu(h))

