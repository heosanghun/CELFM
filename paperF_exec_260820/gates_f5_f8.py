"""F5 점유율 CI / F6 위기 순위상관(가능값 집합 검사) / F8 조기경보 플라시보. 임계값은 prereg 파싱값 사용."""
import numpy as np, pandas as pd, json, re, torch, itertools
from scipy import stats
import celfm_model as M
def run(gate,cfg,pre,root,lr,pca,th):
    W,k=cfg['state']['window_days'],cfg['state']['pca_dim']; dev='cuda'; res={'gate':gate,'seeds':{}}
    Q,_=M.make_states(lr.values,W,pca,k); dates=lr.index[W:]
    for seed in cfg['seeds']:
        E=M.Energy(k,cfg['model']['hidden'],cfg['model']['layers'],cfg['model']['spectral_norm']).to(dev); E.load_state_dict(torch.load(f"{root}/runs/F2/model_seed{seed}.pt"))
        F2=json.load(open(f"{root}/runs/F2/result.json"))['seeds'][str(seed)]; cents=np.array(F2['centers'])
        Qs=M.relax(E,Q,cfg,dev); lab=np.linalg.norm(Qs[:,None]-cents[None],axis=2).argmin(1)
        main=np.bincount(lab).argmax(); nonmain=(lab!=main)
        if gate=='F5':
            ci=th(r"binomial ([0-9.]+) CI"); base=nonmain.mean(); out={}
            for cw in cfg['crisis_windows']:
                m=(dates>=cw['start'])&(dates<=cw['end']); n=int(m.sum()); x=int(nonmain[m].sum())
                lo,hi=stats.binom.interval(ci,n,base); out[cw['name']]={'n':n,'x':x,'baseline':float(base),'ci':[int(lo),int(hi)],'in_ci':bool(lo<=x<=hi)}
            res['seeds'][seed]=out; print(f"[F5] seed={seed} {out}",flush=True)
        elif gate=='F6':
            with torch.no_grad(): V=E(torch.tensor(Q,device=dev)).cpu().numpy()
            px=np.exp(np.log1p(np.exp(lr.values)-1).cumsum(0)) if False else None
            depth=[];dd=[]
            for cw in cfg['crisis_windows']:
                m=(dates>=cw['start'])&(dates<=cw['end']); depth.append(float(V[m].max()-V[m].min()))
                seg=lr.loc[cw['start']:cw['end']].mean(1).cumsum(); dd.append(float((seg-seg.cummax()).min()))
            n=len(depth); rho=float(stats.spearmanr(depth,[-d for d in dd]).correlation)
            poss=sorted(set(round(1-6*sum((i-j)**2 for i,j in zip(range(n),p))/(n*(n*n-1)),6) for p in itertools.permutations(range(n))))
            ok_set=any(abs(rho-p)<1e-6 for p in poss)
            res['seeds'][seed]={'n':n,'depth':depth,'drawdown':dd,'rho':rho,'rho_in_admissible_set':ok_set}
            print(f"[F6] seed={seed} n={n} rho={rho} admissible={ok_set}",flush=True)
        elif gate=='F8':
            lead=th(r"lead time >= ([0-9.]+)"); npl=int(th(r"placebo windows=([0-9]+)")); rng=np.random.default_rng(seed)
            with torch.no_grad(): V=E(torch.tensor(Q,device=dev)).cpu().numpy()
            trend=pd.Series(V,index=dates).rolling(60).mean().diff(20)
            def lead_time(start):
                s=pd.Timestamp(start); pre_=trend.loc[:s].dropna(); thr=pre_.quantile(0.95)
                hits=pre_[pre_>thr]; return float((s-hits.index[-1]).days) if len(hits) else np.nan
            real=[lead_time(cw['start']) for cw in cfg['crisis_windows']]
            plac=[lead_time(dates[i]) for i in rng.integers(400,len(dates)-1,size=npl)]
            p=float(stats.mannwhitneyu(np.nan_to_num(real,nan=0),np.nan_to_num(plac,nan=0),alternative='less').pvalue)
            res['seeds'][seed]={'real_lead_days':real,'placebo_median':float(np.nanmedian(plac)),'p_one_sided':p,'threshold_days':lead}
            print(f"[F8] seed={seed} {res['seeds'][seed]}",flush=True)
    return res
