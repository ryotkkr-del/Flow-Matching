"""未学習のMNIST test画像でFM損失を比較し、固定ノイズの生成画像を並べる。
FM損失は生成画像の品質スコアではない。最終PNGも目視で確認する。
"""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from torchvision import datasets, transforms
from PIL import Image, ImageDraw
from .core import sample_path
from .trajectory import load_model, generate_path, fingerprint


def save_grid(images, path, columns=8, label=''):
    n=len(images);scale=6;gap=8;side=28*scale
    rows=(n+columns-1)//columns
    out=Image.new('RGB',(columns*(side+gap)+gap,rows*(side+gap)+40),(10,14,21))
    draw=ImageDraw.Draw(out);draw.text((gap,10),label,fill=(118,234,216))
    pixels=((images.detach().cpu().clamp(-1,1)+1)*127.5).to(torch.uint8).squeeze(1).numpy()
    for i,p in enumerate(pixels):
        im=Image.fromarray(p).resize((side,side),Image.Resampling.NEAREST).convert('RGB')
        out.paste(im,(gap+(i%columns)*(side+gap),32+(i//columns)*(side+gap)))
    out.save(path)


@torch.inference_mode()
def measure(path, test_images, *, seed, steps, out, name):
    m,info=load_model(path)
    # モデル間で、同じテスト画像・時刻・ノイズを使う。
    torch.manual_seed(seed)
    total,count=0.,0
    for x1 in test_images.split(128):
        t,xt,target=sample_path(x1)
        loss=F.mse_loss(m(xt,t),target)
        total+=loss.item()*len(x1);count+=len(x1)
    start=time.perf_counter()
    x,_=generate_path(m,seed=seed,count=64,steps=steps)
    seconds=time.perf_counter()-start
    save_grid(x[-1],out/f'{name}-64.png',label=f'{name} / epoch {info["epoch"]} / seed {seed} / Euler {steps}')
    return {'checkpoint_sha256':info['_source_sha256'],'epoch':info['epoch'],
            'heldout_fm_mse':total/count,'test_count':count,
            'sampling_seed':seed,'steps':steps,'generated_count':64,
            'sampling_seconds':round(seconds,3)},x[-1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--before',default='checkpoints/exhibit.pt')
    p.add_argument('--after',default='checkpoints/mnist20.pt')
    p.add_argument('--out',default='quality')
    p.add_argument('--data',default='data')
    p.add_argument('--test-count',type=int,default=1024)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--steps',type=int,default=50)
    a=p.parse_args()
    if not 1<=a.test_count<=10000 or not 1<=a.steps<=200:p.error('評価サイズが範囲外です')
    torch.set_num_threads(4)
    d=datasets.MNIST(a.data,train=False,download=False,
                     transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5,),(.5,))]))
    images=torch.stack([d[i][0] for i in range(a.test_count)])
    folder=Path(a.out);folder.mkdir(parents=True,exist_ok=True)
    before,x_before=measure(a.before,images,seed=a.seed,steps=a.steps,out=folder,name='before')
    after,x_after=measure(a.after,images,seed=a.seed,steps=a.steps,out=folder,name='after')
    # 左4epoch・右追加学習後。同じ初期ノイズの対応を各行で比較。
    pairs=torch.stack([x_before,x_after],dim=1).reshape(-1,1,28,28)
    save_grid(pairs[:32],folder/'comparison.png',columns=8,label=f'Pairs: epoch {before["epoch"]} | epoch {after["epoch"]} / same noise / seed {a.seed}')
    result={'before':before,'after':after,
            'relative_heldout_loss_change':after['heldout_fm_mse']/before['heldout_fm_mse']-1,
            'note':'Held-out FM MSE is not an image-quality metric; inspect the fixed-seed grids.'}
    (folder/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
