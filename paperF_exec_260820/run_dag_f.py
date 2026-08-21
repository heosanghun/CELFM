"""24h 자율 오케스트레이터. 단계 순서 고정, 스크립트 봉인, stdout 원문 저장, 하드 스톱. 체인 쓰기 없음."""
import argparse, os, subprocess, sys, json, time
from common import *
ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); a=ap.parse_args()
cfg=load_cfg(a.config); root=cfg['project_root']; os.chdir(root)
STAGES=[('S1','build_panel.py',[]),('S2','make_prereg.py',[]),('S3','make_surrogates.py',[]),
        ('F1','run_gate.py',['--gate','F1']),('F2','run_gate.py',['--gate','F2']),('F3','run_gate.py',['--gate','F3']),
        ('F4','run_gate.py',['--gate','F4']),('F5','run_gate.py',['--gate','F5']),('F6','run_gate.py',['--gate','F6']),
        ('F7','run_gate.py',['--gate','F7']),('F8','run_gate.py',['--gate','F8']),('S9','make_report.py',[])]
HARD_STOP={'F1','F7'}
def status(msg):
    with open('status.log','a') as f: f.write(f"{utc_now()} | {msg}\n")
    print(msg,flush=True)
for stage,script,args in STAGES:
    if os.path.exists('STOP_REASON.txt'): status('STOP file present — halting'); sys.exit(2)
    d=f"runs/{stage}"; os.makedirs(d,exist_ok=True)
    if os.path.exists(f"{d}/result.json") or (stage in ('S1','S2','S3') and os.path.exists(f"{d}/DONE")):
        status(f"{stage} already done — skip"); continue
    if stage=='S3':  # prereg 승인 토큰 대기 (유일한 사람 개입)
        p='prereg/prereg_F_v1.txt'
        while True:
            if os.path.exists('prereg/APPROVAL.txt') and os.path.exists(p):
                tok=open('prereg/APPROVAL.txt').read().strip(); h=sha256_file(p)
                if tok==f"[PRINCIPAL-APPROVE PREREG_F_V1 {h}]": status(f"prereg approved sha256={h}"); break
                status(f"APPROVAL.txt present but token mismatch (file sha256={h}) — waiting")
            else: status('waiting for prereg/APPROVAL.txt')
            time.sleep(300)
    h=seal_script(d,script); status(f"{stage} start script_sha256={h}")
    with open(f"{d}/stdout.log",'a') as log:
        log.write(f"=== {utc_now()} {stage} {script} {' '.join(args)} sha256={h} ===\n"); log.flush()
        rc=subprocess.call([sys.executable,script,'--config',a.config]+args,stdout=log,stderr=subprocess.STDOUT)
    status(f"{stage} exit={rc} stdout_sha256={sha256_file(f'{d}/stdout.log')}")
    if rc!=0:
        ts=utc_now().replace(':','-'); os.rename(d,f"{d}_FAILED_{ts}"); os.makedirs(d,exist_ok=True)
        stop(root,f"{stage} exit code {rc} — see runs/{stage}_FAILED_{ts}/stdout.log")
    if stage in ('S1','S2','S3'): open(f"{d}/DONE",'w').write(utc_now())
    if stage in HARD_STOP:
        r=json.load(open(f"{d}/result.json"))
        if stage=='F1' and not r.get('gate_ok',False): stop(root,'F1 negative control GATE_NOT_MET — method falsified, no retries')
status('DAG complete')
