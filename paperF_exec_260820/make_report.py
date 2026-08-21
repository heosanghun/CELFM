"""S9: 보고서 자동 생성. 판정 문구 없음 — 수치·포인터·해시만. 체인 후보 state 기록."""
import argparse, os, json, glob
from common import *
ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); a=ap.parse_args()
cfg=load_cfg(a.config); root=cfg['project_root']; os.chdir(root)
L=[json.loads(l) for l in open('ledgers/results_ledger.jsonl')]
lines=[f"# REPORT_F {utc_now()}", "", "## 파일 해시", "```"]
for p in sorted(glob.glob('panel/*')+glob.glob('prereg/*')+glob.glob('surrogates/*')+glob.glob('runs/*/script.sha256')+glob.glob('runs/*/result.json')+glob.glob('runs/*/stdout.log')+['ledgers/results_ledger.jsonl','env_freeze.txt']):
    if os.path.isfile(p): lines.append(f"{sha256_file(p)}  {p}")
lines+=["```","","## 게이트 결과 (result.json 원문 인용)"]
for g in ['F1','F2','F3','F4','F5','F6','F7','F8']:
    p=f"runs/{g}/result.json"
    if os.path.exists(p): lines+= [f"### {g}", "```json", open(p).read(), "```", f"evidence: runs/{g}/stdout.log (sha256 {sha256_file(f'runs/{g}/stdout.log')})"]
    else: lines.append(f"### {g}: NOT RUN")
if os.path.exists('STOP_REASON.txt'): lines+=["","## STOP_REASON.txt","```",open('STOP_REASON.txt').read(),"```"]
state=sha256_file('ledgers/results_ledger.jsonl')
with open('chain_candidates.jsonl','a') as f: f.write(json.dumps({'ts':utc_now(),'event':'PAPER_F_DAG_RUN','state_candidate':state,'note':'Principal appends to chain; Antigravity must not'})+'\n')
lines+=["",f"## 체인 후보 state = SHA-256(results_ledger.jsonl) = {state}","","판정은 감사자(System 2)가 수행한다. 이 문서에는 판정 문구가 없다."]
out=f"REPORT_F_{utc_now()[:10]}.md"; open(out,'w').write('\n'.join(lines)); print(f"report written {out} sha256={sha256_file(out)}")
