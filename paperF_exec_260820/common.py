import hashlib, json, os, subprocess, sys, time, datetime
def sha256_file(p, chunk=1<<20):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(chunk), b''): h.update(b)
    return h.hexdigest()
def sha256_bytes(b): return hashlib.sha256(b).hexdigest()
def utc_now(): return subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%S.%6N+00:00']).decode().strip()
def seal_script(stage_dir, script_path):
    os.makedirs(stage_dir, exist_ok=True)
    h=sha256_file(script_path)
    with open(os.path.join(stage_dir,'script.sha256'),'a') as f: f.write(f"{h}  {script_path}  {utc_now()}\n")
    return h
def ledger_append(root, rec):
    rec=dict(rec); rec['ts']=utc_now()
    p=os.path.join(root,'ledgers','results_ledger.jsonl'); os.makedirs(os.path.dirname(p),exist_ok=True)
    with open(p,'a',encoding='utf-8') as f: f.write(json.dumps(rec,ensure_ascii=False)+'\n')
def stop(root, reason):
    with open(os.path.join(root,'STOP_REASON.txt'),'a') as f: f.write(f"{utc_now()} | {reason}\n")
    print(f"[STOP] {reason}", flush=True); sys.exit(2)
def load_cfg(path):
    import yaml; cfg=yaml.safe_load(open(path))
    def walk(x,path=''):
        if isinstance(x,dict):
            for k,v in x.items(): walk(v,path+'/'+k)
        elif x=='TBD': raise SystemExit(f"config TBD remains at {path} — Principal must fill it")
    walk(cfg); return cfg
def assert_no_tbd_thresholds(cfg):
    for k,v in cfg['thresholds'].items():
        if v=='TBD': raise SystemExit(f"threshold {k} is TBD")
