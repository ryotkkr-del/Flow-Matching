"""展示用の実軌道と品質確認用の最終画像を保存。"""
import argparse
import gzip
import json
import os
from pathlib import Path
import torch
from .trajectory import load_model, generate_path, payload, fingerprint, validate_payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='checkpoints/mnist.pt')
    p.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    p.add_argument('--steps', type=int, nargs='+', default=[20,50,100])
    p.add_argument('--count', type=int, default=4)
    p.add_argument('--out', default='demos')
    a = p.parse_args()
    if not 1 <= a.count <= 8 or any(not 1 <= s <= 200 for s in a.steps): p.error('サイズが範囲外です')
    torch.set_num_threads(4)
    # 展示PCのCPUと同じ計算で再現・照合できるよう、書き出しはCPU。
    device = 'cpu'
    model, info = load_model(a.checkpoint, device)
    if info.get('smoke_only'): raise ValueError('動作確認のみのモデルは展示デモに書き出せません')
    folder=Path(a.out); folder.mkdir(parents=True, exist_ok=True)
    sha=fingerprint(a.checkpoint)
    for seed in a.seeds:
        for steps in a.steps:
            x,v=generate_path(model, seed, a.count, steps, device)
            d=payload(x,v,seed=seed,steps=steps,checkpoint_sha256=sha,
                      epoch=info.get('epoch',0),loss_history=info.get('loss_history'))
            validate_payload(d)
            path=folder/f'seed{seed}-steps{steps}.json.gz'
            temporary=path.with_suffix(path.suffix+'.tmp')
            with gzip.open(temporary,'wt',encoding='utf-8') as f: json.dump(d,f,separators=(',',':'),allow_nan=False)
            os.replace(temporary,path)
            from PIL import Image
            pixels=((x[-1].clamp(-1,1)+1)*127.5).to(torch.uint8).squeeze(1).numpy()
            import numpy as np
            Image.fromarray(np.concatenate(list(pixels),axis=1)).resize((a.count*224,224),Image.Resampling.NEAREST).save(folder/f'seed{seed}-steps{steps}.png')
            print(path,flush=True)
if __name__=='__main__': main()
