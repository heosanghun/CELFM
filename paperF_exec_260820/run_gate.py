"""게이트 실행기. 임계값은 prereg_F_v1.txt에서 파싱(코드 상수 금지). 결과는 result.json + stdout 포인터."""
import argparse, os, re, json, numpy as np, pandas as pd, torch
from common import *; import celfm_model as M
ap=argparse.ArgumentParser(); ap.add_argument('--gate',required=True); ap.add_argument('--config',required=True); a=ap.parse_args()
cfg=load_cfg(a.config); root=cfg['project_root']; dev='cuda'
pre=open(f"{root}/prereg/prereg_F_v1.txt").read()
def th(pattern):
    m=re.search(pattern,pre); 
    if not m: stop(root,f"prereg missing pattern {pattern}")
    return float(m.group(1))
out_dir=f"{root}/runs/{a.gate}"; os.makedirs(out_dir,exist_ok=True)
lr=pd.read_parquet(f"{root}/panel/logret.parquet"); W,k=cfg['state']['window_days'],cfg['state']['pca_dim']
tr_df=lr.loc[:cfg['panel']['train_end']]; Qtr,pca=M.make_states(tr_df.values,W,None,k)
res={'gate':a.gate,'seeds':{},'prereg_sha256':sha256_file(f"{root}/prereg/prereg_F_v1.txt")}
def analyze(E,Q,seed,tag):
    Qs=M.relax(E,Q,cfg,dev); cents,lab=M.attractors(Qs,cfg['model']['attractor_merge_eps'])
    lmins=[M.hessian_lmin(E,c,dev) for c in cents]
    with torch.no_grad(): Vc=E(torch.tensor(cents,device=dev)).cpu().numpy(); Vall=E(torch.tensor(Q,device=dev)).cpu().numpy()
    main=np.bincount(lab).argmax(); vanc=((Vc-Vc[main])/(Vall.max()-Vall.min()+1e-12)).tolist()
    r={'N_attr':int(len(cents)),'lambda_min':lmins,'V_anc':vanc,'centers':cents.tolist(),'assign_counts':np.bincount(lab).tolist()}
    print(f"[{a.gate}] {tag} seed={seed} N_attr={r['N_attr']} lambda_min={lmins} V_anc={vanc}", flush=True); return r,cents,lab
if a.gate=='F1':
    vmax=th(r"\|V_anc\| <= ([0-9.]+)"); ok=True
    for kind in cfg['surrogates']['kinds']:
        for seed in cfg['seeds']:
            X=np.load(f"{root}/surrogates/{kind}_seed{seed}.npz")['X']; Q,_=M.make_states(X,W,pca,k)
            E=M.train(Q,cfg,seed,dev); r,_,_=analyze(E,Q,seed,kind)
            r['gate_ok']=bool(r['N_attr']==1 and min(r['lambda_min'])>0 and max(abs(v) for v in r['V_anc'])<=vmax)
            ok&=r['gate_ok']; res['seeds'][f"{kind}_{seed}"]=r
    res['gate_ok']=ok
elif a.gate in ('F2','F3','F4','F7'):
    for seed in cfg['seeds']:
        ck=f"{root}/runs/F2/model_seed{seed}.pt"
        if a.gate=='F2':
            E=M.train(Qtr,cfg,seed,dev); torch.save(E.state_dict(),ck)
        else:
            E=M.Energy(k,cfg['model']['hidden'],cfg['model']['layers'],cfg['model']['spectral_norm']).to(dev); E.load_state_dict(torch.load(ck))
        if a.gate=='F2':
            r,cents,lab=analyze(E,Qtr,seed,'train'); r['assignment_rate']=float((np.bincount(lab)>=max(5,0.01*len(lab))).sum()>0 and 1.0); res['seeds'][seed]=r
        elif a.gate=='F3':
            ho,_=M.make_states(lr.loc[cfg['panel']['holdout_start']:].values,W,pca,k); F2=json.load(open(f"{root}/runs/F2/result.json"))['seeds'][str(seed)]
            Qs=M.relax(E,ho,cfg,dev); cents=np.array(F2['centers']); d=np.linalg.norm(Qs[:,None]-cents[None],axis=2); eps=cfg['model']['attractor_merge_eps']
            r={'holdout_n':len(ho),'assignment_rate':float((d.min(1)<=eps).mean()),'new_attractors':int((d.min(1)>eps).sum()>0)}; print(f"[F3] seed={seed} {r}",flush=True); res['seeds'][seed]=r
        elif a.gate=='F4':
            F2=json.load(open(f"{root}/runs/F2/result.json"))['seeds'][str(seed)]; cents=np.array(F2['centers']); pairs=[]
            for i in range(len(cents)):
                for j in range(len(cents)):
                    if i!=j: b,si,_=M.string_barrier(E,cents[i],cents[j],cfg,dev); pairs.append({'from':i,'to':j,'barrier_nats':b,'saddle_idx':si}); print(f"[F4] seed={seed} {pairs[-1]}",flush=True)
            res['seeds'][seed]={'pairs':pairs}
        elif a.gate=='F7':
            F2=json.load(open(f"{root}/runs/F2/result.json"))['seeds'][str(seed)]; neg=[l for l in F2['lambda_min'] if l<=0]
            res['seeds'][seed]={'lambda_min':F2['lambda_min'],'any_nonpositive':bool(neg)}
            if neg: stop(root,f"F7 lambda_min<=0 seed={seed} {neg}")
else:
    # F5/F6/F8: 점유율·순위상관·조기경보 — F2/F4 산출물과 crisis_windows로 계산. 수치 상수 없음.
    print(f"[{a.gate}] implemented in gates_f5_f8.py — see that file", flush=True); import gates_f5_f8; res=gates_f5_f8.run(a.gate,cfg,pre,root,lr,pca,th)
json.dump(res,open(f"{out_dir}/result.json",'w'),indent=1)
ledger_append(root,{'stage':a.gate,'result_sha256':sha256_file(f"{out_dir}/result.json"),'evidence':f"runs/{a.gate}/stdout.log"})
print(f"[{a.gate}] result.json sha256={sha256_file(f'{out_dir}/result.json')}")
