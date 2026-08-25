#!/usr/bin/env bash
set -euo pipefail

MONITOR_CHECKOUT=/root/autodl-tmp/CoFiTok/checkouts/min-snr-pilot-health-monitor-4bbc880/CoFiTok-internal
TRAINING_CHECKOUT=/root/autodl-tmp/CoFiTok/checkouts/min-snr-matched-pilot-842a341/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/min_snr_gamma5_matched_50k_pilot_v1
MONITOR_ROOT="$OUTPUT_ROOT/reports/health_monitor_v1"
STATUS="$MONITOR_ROOT/status.json"
LOCK="$MONITOR_ROOT/monitor.lock"
LOG="$MONITOR_ROOT/monitor.log"
RECEIPT="$MONITOR_ROOT/deployment_receipt.json"
MONITOR="$MONITOR_CHECKOUT/scripts/monitor_generation_min_snr_pilot.py"
CONTROLLER_REPORT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_full_data_100k_base128_quality_bridge_v1/reports/min_snr_matched_50k_pilot_v1_20260825

[[ "$(git -C "$MONITOR_CHECKOUT" rev-parse HEAD)" == 4bbc880ad3928b607ee61bc1be41d10a60853e4d ]]
[[ "$(git -C "$MONITOR_CHECKOUT" rev-parse HEAD^{tree})" == 6404585661c297cdb57447e81cdfd3f84331d569 ]]
[[ "$(git -C "$MONITOR_CHECKOUT" branch --show-current)" == analysis/generation-min-snr-pilot-health-monitor-v1-20260825 ]]
[[ -z "$(git -C "$MONITOR_CHECKOUT" status --porcelain=v1 --untracked-files=all)" ]]
[[ "$(sha256sum "$MONITOR" | awk '{print $1}')" == 21d0c7c83ef662c3e791a4780d6a33ac08bdc2afcce885c45d3fc621272bd5ad ]]
[[ "$(git -C "$TRAINING_CHECKOUT" rev-parse HEAD)" == 842a34130e82f241330707118a05bf6ed01e263e ]]
[[ "$(git -C "$TRAINING_CHECKOUT" rev-parse HEAD^{tree})" == 3fd4c4538d15b85233b1f8b582dce0f185dedba2 ]]
[[ "$(git -C "$TRAINING_CHECKOUT" branch --show-current)" == scale/generation-min-snr-matched-pilot-v1-20260825 ]]
[[ -z "$(git -C "$TRAINING_CHECKOUT" status --porcelain=v1 --untracked-files=all)" ]]
[[ -d /proc/407472 ]]
[[ "$(awk '{print $22}' /proc/407472/stat)" == 1729557004 ]]
[[ ! -e "$STATUS" ]]
[[ ! -e "$LOCK" ]]
[[ ! -e "$RECEIPT" ]]
if pgrep -f '/scripts/[m]onitor_generation_min_snr_pilot.py' >/dev/null; then
  printf 'refusing duplicate Min-SNR pilot health monitor\n' >&2
  exit 73
fi

mkdir -p "$MONITOR_ROOT"
nohup bash -c '
while true; do
  read -r _ _ _ parent_pid _ < "/proc/$$/stat"
  [[ "$parent_pid" == 1 ]] && break
  sleep 0.1
done
exec "$@"
' cofitok-min-snr-monitor \
  env \
  CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  PYTHONPATH="$MONITOR_CHECKOUT/src:$MONITOR_CHECKOUT" \
  nice -n 10 \
  ionice -c 3 \
  /root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10 \
  "$MONITOR" \
  --output-root "$OUTPUT_ROOT" \
  --status-output "$STATUS" \
  --monitor-lock "$LOCK" \
  --training-checkout "$TRAINING_CHECKOUT" \
  --monitor-checkout "$MONITOR_CHECKOUT" \
  --controller-status "$OUTPUT_ROOT/controller_status.json" \
  --controller-runbook "$TRAINING_CHECKOUT/artifacts/runbooks/generation_min_snr_matched_50k_pilot_v1.sh" \
  --controller-launch-receipt "$CONTROLLER_REPORT_ROOT/controller_launch_receipt.2026-08-25_842a341.json" \
  --controller-process-tree-audit "$CONTROLLER_REPORT_ROOT/controller_process_tree_audit.2026-08-25_842a341.json" \
  --preparation "$CONTROLLER_REPORT_ROOT/preparation.json" \
  --execution-gate "$CONTROLLER_REPORT_ROOT/execution_gate.json" \
  --execution-lock /tmp/cofitok-min-snr-matched-50k-pilot-v1.lock \
  --expected-training-revision 842a34130e82f241330707118a05bf6ed01e263e \
  --expected-training-tree 3fd4c4538d15b85233b1f8b582dce0f185dedba2 \
  --expected-training-branch scale/generation-min-snr-matched-pilot-v1-20260825 \
  --expected-monitor-revision 4bbc880ad3928b607ee61bc1be41d10a60853e4d \
  --expected-monitor-tree 6404585661c297cdb57447e81cdfd3f84331d569 \
  --expected-monitor-branch analysis/generation-min-snr-pilot-health-monitor-v1-20260825 \
  --expected-controller-pid 407472 \
  --expected-controller-start-ticks 1729557004 \
  --expected-self-sha256 21d0c7c83ef662c3e791a4780d6a33ac08bdc2afcce885c45d3fc621272bd5ad \
  --expected-preparation-sha256 8fa3b768015c08c6f34627dc767c4b2fddd465a43913495120a496ef411c6830 \
  --expected-execution-gate-sha256 19cb1560cca1feba38bf7a553652abf922e59d635d5574994996e37fcc72c014 \
  --expected-controller-runbook-sha256 55cbf51e89c27da6c5c0fbf1e9a9e0c07b4b261b2342392c0e17c3bd846d6656 \
  --expected-controller-launch-receipt-sha256 6ce5a4a3f4d3225f68ece879cd07ceb01361d7a1315dc88d4beffab7b057f261 \
  --expected-controller-process-tree-audit-sha256 9be864709967606a9c0da7eb90b73d84a765d505bf73b8350eb3891e6439fcca \
  --poll-seconds 60 \
  --require-parent-pid-one \
  >>"$LOG" 2>&1 </dev/null &

printf '%s\n' "$!"
