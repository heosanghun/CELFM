"""CELFM 최소 구현: 상태 q(PCA), 보존적 flow matching 에너지 V, relaxation, 어트랙터, string, 헤시안.
여기에는 어떤 임계값도 없다. 임계값은 run_gate.py가 prereg 파일에서 파싱한다."""
import numpy as np, torch, torch.nn as nn
from sklearn.decomposition import PCA
def make_states(logret, W, pca=None, k=8):
    X=(logret-logret.mean(0))/ (logret.std(0)+1e-8)
    win=np.stack([X[i-W:i].reshape(-1) for i in range(W,len(X))])
    if pca is None: pca=PCA(n_components=k,random_state=0).fit(win)
    return pca.transform(win).astype(np.float32), pca
class Energy(nn.Module):
    def __init__(s,k,h,L,sn):
        super().__init__(); lay=[]; d=k
        for _ in range(L):
            lin=nn.Linear(d,h); lin=nn.utils.parametrizations.spectral_norm(lin) if sn else lin
            lay+=[lin,nn.Softplus()]; d=h
        lay.append(nn.Linear(d,1)); s.net=nn.Sequential(*lay)
    def forward(s,q): return s.net(q).squeeze(-1)
    def grad(s,q):
        q=q.requires_grad_(True); V=s(q).sum(); return torch.autograd.grad(V,q,create_graph=True)[0]
def train(Q,cfg,seed,dev):
    torch.manual_seed(seed); np.random.seed(seed); m=cfg['model']
    q0=torch.tensor(Q[:-1],device=dev); q1=torch.tensor(Q[1:],device=dev)
    E=Energy(Q.shape[1],m['hidden'],m['layers'],m['spectral_norm']).to(dev)
    opt=torch.optim.Adam(E.parameters(),lr=m['lr']); n=len(q0)
    for ep in range(m['epochs']):
        perm=torch.randperm(n,device=dev); tot=0.0
        for i in range(0,n,m['batch']):
            idx=perm[i:i+m['batch']]; a,b=q0[idx],q1[idx]; t=torch.rand(len(idx),1,device=dev)
            qt=(1-t)*a+t*b; u_target=b-a            # conditional FM target
            u=-E.grad(qt); loss=((u-u_target)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()*len(idx)
        if ep%20==0 or ep==m['epochs']-1: print(f"  epoch {ep} fm_loss {tot/n:.6f}", flush=True)
    return E
@torch.no_grad()
def _noop(): pass
def relax(E,q,cfg,dev):
    m=cfg['model']; q=torch.tensor(q,device=dev).clone(); v=torch.zeros_like(q); dt,g=m['relax_dt'],m['relax_gamma']
    for _ in range(m['relax_steps']):
        with torch.enable_grad(): f=-E.grad(q).detach()
        v=(1-g*dt)*v+dt*f; q=(q+dt*v).detach()
    return q.cpu().numpy()
def attractors(Qstar,eps):
    cents=[]; lab=np.full(len(Qstar),-1)
    for i,q in enumerate(Qstar):
        for j,c in enumerate(cents):
            if np.linalg.norm(q-c)<=eps: lab[i]=j; break
        if lab[i]<0: cents.append(q.copy()); lab[i]=len(cents)-1
    return np.array(cents), lab
def hessian_lmin(E,qs,dev):
    q=torch.tensor(qs,device=dev)
    H=torch.autograd.functional.hessian(lambda x:E(x.unsqueeze(0)).squeeze(),q)
    return float(torch.linalg.eigvalsh(H).min())
def string_barrier(E,qa,qb,cfg,dev):
    m=cfg['model']; n=m['string_nodes']
    path=torch.tensor(np.linspace(qa,qb,n),device=dev)
    for _ in range(m['string_iters']):
        with torch.enable_grad(): g=E.grad(path).detach()
        path[1:-1]-=m['relax_dt']*g[1:-1]
        # reparametrize by arc length
        d=torch.cumsum(torch.cat([torch.zeros(1,device=dev),(path[1:]-path[:-1]).norm(dim=1)]),0); d=d/d[-1]
        tgt=torch.linspace(0,1,n,device=dev); new=path.clone()
        for k in range(path.shape[1]): new[:,k]=torch.tensor(np.interp(tgt.cpu(),d.cpu(),path[:,k].cpu()),device=dev)
        path=new
    with torch.no_grad(): Vp=E(path).cpu().numpy()
    return float(Vp.max()-Vp[0]), int(Vp.argmax()), path.cpu().numpy()
