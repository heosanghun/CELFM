"""S2: 사전등록 문안 생성. thresholds에 TBD 있으면 거부. 측정값 없음."""
import argparse, os, json
from common import *
ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); a=ap.parse_args()
cfg=load_cfg(a.config); assert_no_tbd_thresholds(cfg); root=cfg['project_root']; th=cfg['thresholds']
n=len(cfg['crisis_windows'])
# n개 순위의 스피어만 가능값 집합 (이산) — 보고값이 이 집합 밖이면 즉시 반려
import itertools
poss=sorted(set(round(1-6*sum((i-j)**2 for i,j in zip(range(n),p))/(n*(n*n-1)),6) for p in itertools.permutations(range(n))))
txt=f"""PREREG_F_v1 | generated {utc_now()} | panel_manifest={sha256_file(f'{root}/panel/panel_manifest.txt')}
DEFINITIONS
 q_t = PCA_{cfg['state']['pca_dim']}( standardized log-returns window W={cfg['state']['window_days']} ), PCA fit on train <= {cfg['panel']['train_end']}
 V = learned energy; attractor q* = relaxation fixed point (damped Verlet, gamma={cfg['model']['relax_gamma']}, dt={cfg['model']['relax_dt']}, steps={cfg['model']['relax_steps']}); merge eps={cfg['model']['attractor_merge_eps']}
 V_anc = (V(q*_i) - V(q*_0)) / (V_max_train - V_min_train) with q*_0 = attractor holding most train points
 barrier(i->j) = V(saddle on string i->j) - V(q*_i), string nodes={cfg['model']['string_nodes']}
 lambda_min = smallest eigenvalue of autograd Hessian of V at q*
 crisis windows (fixed): {json.dumps(cfg['crisis_windows'])}
GATES (seeds {cfg['seeds']}; all conditions must hold for every seed unless stated)
 F1 surrogates {cfg['surrogates']['kinds']}: N_attr == 1 AND lambda_min > 0 AND |V_anc| <= {th['F1_vanc_abs_max']}
 F2 real train: N_attr >= {th['F2_min_attractors']} AND assignment_rate >= {th['F2_min_assignment']} AND cross-seed attractor distance <= {th['F2_seed_consistency_eps']}
 F3 holdout >= {cfg['panel']['holdout_start']}: assignment_rate >= {th['F3_min_holdout_assignment']} AND new_attractors == 0
 F4 any attractor pair: barrier >= {th['F4_min_barrier_nats']} nats AND at least one crisis window contains a basin transition
 F5 regime occupancy in crisis windows within binomial {th['F5_binomial_ci']} CI vs baseline occupancy
 F6 Spearman rho over n={n} crisis windows (rank by barrier crossing depth vs. realized drawdown) >= {th['F6_min_rank_corr']}; admissible values={poss}
 F7 lambda_min(H*) > 0 at EVERY attractor (any negative -> GATE_NOT_MET, no exceptions)
 F8 barrier-trend early warning lead time >= {th['F8_min_leadtime_days']} days, placebo windows={th['F8_placebo_windows']}, one-sided p < 0.05
RULES: thresholds frozen at seal; no measured value may be written here; F1 failure halts all later gates.
"""
os.makedirs(f"{root}/prereg",exist_ok=True); p=f"{root}/prereg/prereg_F_v1.txt"
if os.path.exists(p): raise SystemExit("prereg_F_v1.txt already exists — do not overwrite; issue v2 with reason")
open(p,'w').write(txt); print(txt); print("SHA256", sha256_file(p))
ledger_append(root,{'stage':'S2','prereg_sha256':sha256_file(p)})
