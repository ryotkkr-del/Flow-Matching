"""保存軌道が指定checkpointの実際の出力であるか再計算して照合。"""
import argparse,gzip,json
import numpy as np
import torch
from .trajectory import load_model,generate_path,fingerprint,validate_payload

def main():
    p=argparse.ArgumentParser();p.add_argument('trajectory');p.add_argument('--checkpoint',default='checkpoints/mnist.pt');a=p.parse_args()
    opener=gzip.open if a.trajectory.endswith('.gz') else open
    with opener(a.trajectory,'rt',encoding='utf-8') as f:d=validate_payload(json.load(f))
    torch.set_num_threads(4);model,info=load_model(a.checkpoint)
    if info['_source_sha256']!=d['checkpoint_sha256']:raise ValueError('checkpoint SHA256が一致しません')
    x,v=generate_path(model,d['seed'],d['count'],d['steps'])
    np.testing.assert_allclose(x.flatten(2).numpy(),d['states'],atol=1e-5,rtol=1e-4)
    np.testing.assert_allclose(v.flatten(2).numpy(),d['velocities'],atol=1e-5,rtol=1e-4)
    print('PASS: checkpoint hash / Euler updates / recomputed model trajectory')
if __name__=='__main__':main()
