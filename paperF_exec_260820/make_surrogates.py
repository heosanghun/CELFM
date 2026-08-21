"""S3: 음성 대조군 서로게이트 3종 x 시드. 학습구간 데이터만 사용."""
import argparse, os, numpy as np, pandas as pd
from common import *
ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); a=ap.parse_args()
cfg=load_cfg(a.config); root=cfg['project_root']
lr=pd.read_parquet(f"{root}/panel/logret.parquet"); tr=lr.loc[:cfg['panel']['train_end']].values
os.makedirs(f"{root}/surrogates",exist_ok=True)
def iaaft(x,rng,iters=100):
    out=np.empty_like(x)
    for j in range(x.shape[1]):
        s=x[:,j]; amp=np.abs(np.fft.rfft(s)); srt=np.sort(s); y=rng.permutation(s)
        for _ in range(iters):
            y=np.fft.irfft(amp*np.exp(1j*np.angle(np.fft.rfft(y))),n=len(s)); y=srt[np.argsort(np.argsort(y))]
        out[:,j]=y
    return out
for seed in cfg['seeds']:
    rng=np.random.default_rng(seed)
    S={'time_shuffle':tr[rng.permutation(len(tr))],
       'gaussian':rng.multivariate_normal(tr.mean(0),np.cov(tr.T),size=len(tr)),
       'iaaft':iaaft(tr,rng)}
    for k in cfg['surrogates']['kinds']:
        p=f"{root}/surrogates/{k}_seed{seed}.npz"; np.savez(p,X=S[k]); h=sha256_file(p)
        print(f"[S3] {k} seed={seed} shape={S[k].shape} sha256={h}")
        ledger_append(root,{'stage':'S3','kind':k,'seed':seed,'sha256':h})
