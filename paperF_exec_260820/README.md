# Paper F (CELFM) 실행 킷 — 2026-08-20
1. 이 code/ 폴더 전체를 `cts-server:/home/shoon/celfm_paperf/`에 복사 (scp -r).
2. Principal: `config.yaml`의 `gpu_index`와 `thresholds`의 TBD를 전부 채운다. TBD가 남으면 모든 스크립트가 실행을 거부한다.
3. Antigravity: S0 환경 설치 (01_DIRECTIVE §2 S0). `sha256sum *.py config.yaml` 출력 제출.
4. Principal 기동: 01_DIRECTIVE §3의 두 명령.
5. 오케스트레이터는 S1→S2 후 `prereg/APPROVAL.txt`를 기다린다. Principal이 prereg_F_v1.txt를 읽고
   `echo "[PRINCIPAL-APPROVE PREREG_F_V1 $(sha256sum prereg/prereg_F_v1.txt | cut -d' ' -f1)]" > prereg/APPROVAL.txt`
6. 이후 24h 무인. 확인: `tail status.log`, `cat STOP_REASON.txt`.
7. 종료 후 Antigravity는 REPORT_F_*.md와 runs/ 전체를 로컬 CELFM\paperF_exec_260820\으로 미러(해시 쌍)하고 01_DIRECTIVE §5 양식으로 보고.
주의: 이 코드에는 실험 임계값이 없다. 전부 prereg 파일에서 파싱된다. 코드 상수로 임계값을 넣는 수정은 금지.
