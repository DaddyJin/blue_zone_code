#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
source "${CONFIG:-./config.sh}"
export PYTHONPATH= PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
stage="${1:?Use plan|convert|verify|load|accept|status|all}"
case "$stage" in plan|convert|verify|load|accept|status|all) ;; *) echo "Unknown stage: $stage" >&2; exit 2;; esac
mkdir -p "$WORK/logs"
run_stage() {
 case "$1" in
 plan|convert)
  "$CONVERT_PY" "$CONVERTER/scripts/convert_repos.py" "$1" \
   --input "$INPUT" --output "$OUTPUT" --work "$WORK" \
   --workers "$WORKERS" --video-workers "$VIDEO_WORKERS" --decode-threads "$DECODE_THREADS" \
   --cpu-budget "$CPU_BUDGET" --video-mb "$VIDEO_MB"
  if [[ "$1" == plan ]]; then
   "$CONVERT_PY" - "$WORK/plan.json" "$OUTPUT" <<'PY'
import sys,json,shutil
p=json.load(open(sys.argv[1]));need=p['extra_disk_estimate_bytes'];free=shutil.disk_usage(sys.argv[2]).free
print(json.dumps({'jobs':p['valid_jobs'],'skipped':p['skipped'],'missing_repositories':p['missing_expected_repos'],'input_TiB':p['input_bytes']/2**40,'additional_required_TiB':need/2**40,'free_TiB':free/2**40,'space_check_passed':free>=need},ensure_ascii=False,indent=2))
PY
  fi
  ;;
 verify)
  "$CONVERT_PY" "$CONVERTER/scripts/verify.py" --plan "$WORK/plan.json" --output "$OUTPUT" --report "$WORK/verification.json"
  ;;
 load)
  "$LEROBOT_PY" -c 'import importlib.metadata as m; v=m.version("lerobot"); print("LeRobot",v); assert v=="0.5.0"'
  "$CONVERT_PY" "$LOADER/scripts/official_load.py" --output "$OUTPUT" --python "$LEROBOT_PY" --report "$WORK/official_load.json"
  ;;
 accept)
  "$CONVERT_PY" "$HERE/final_acceptance.py" --work "$WORK" --output "$OUTPUT"
  ;;
 status)
  "$CONVERT_PY" - "$WORK" <<'PY'
import sys,json,collections
from pathlib import Path
w=Path(sys.argv[1]);p=json.loads((w/'plan.json').read_text());f=w/'progress.json'
r=json.loads(f.read_text()) if f.exists() else {'results':[]}
print(json.dumps({'planned_jobs':p['valid_jobs'],'reported_jobs':len(r['results']),'status_counts':dict(collections.Counter(x['status'] for x in r['results'])),'plan_skipped':len(p['skipped']),'missing_repositories':len(p['missing_expected_repos']),'failed':[x for x in r['results'] if x['status']=='failed']},ensure_ascii=False,indent=2))
print('progress.json is rewritten for each invocation; reported_jobs is not persistent lifetime history.')
PY
  ;;
 esac
}
if [[ "$stage" == all ]]; then stages=(convert verify load accept); else stages=("$stage"); fi
for item in "${stages[@]}"; do
 log="$WORK/logs/${item}_$(date +%Y%m%d_%H%M%S)_$$.log"
 run_stage "$item" 2>&1 | tee "$log"
done
