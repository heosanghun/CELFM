#!/bin/bash
# 30분마다 상태 1줄 append. 아무것도 수정하지 않는다.
cd "$(dirname "$0")"
while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
  stage=$(tail -n 1 status.log 2>/dev/null | cut -d'|' -f2-)
  gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader | tr '\n' ';')
  alive=$(pgrep -f run_dag_f.py >/dev/null && echo ALIVE || echo NOT_RUNNING)
  echo "$ts | WATCHDOG | orchestrator=$alive | last=$stage | gpu=$gpu" >> status.log
  [ -f STOP_REASON.txt ] && { echo "$ts | WATCHDOG | STOP detected" >> status.log; exit 0; }
  [ "$alive" = "NOT_RUNNING" ] && grep -q 'DAG complete' status.log && exit 0
  sleep 1800
done
