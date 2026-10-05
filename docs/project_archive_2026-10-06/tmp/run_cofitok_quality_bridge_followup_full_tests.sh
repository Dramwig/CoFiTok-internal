#!/usr/bin/env bash
set -u

checkout=/tmp/cofitok-quality-bridge-followup-decision-9b02fa8
log=/tmp/cofitok-quality-bridge-followup-decision-9b02fa8-full-tests.log
status=/tmp/cofitok-quality-bridge-followup-decision-9b02fa8-full-tests.status
pid_file=/tmp/cofitok-quality-bridge-followup-decision-9b02fa8-full-tests.pid

cd "$checkout" || exit 70
if [[ -e "$status" ]]; then
  exit 71
fi
printf '%s\n' "$$" >"$pid_file"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=.:src
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

/root/autodl-tmp/conda/envs/pf-vlm/bin/python -m pytest \
  -q -p no:cacheprovider >"$log" 2>&1
rc=$?
printf '%s\n' "$rc" >"$status"
exit "$rc"
