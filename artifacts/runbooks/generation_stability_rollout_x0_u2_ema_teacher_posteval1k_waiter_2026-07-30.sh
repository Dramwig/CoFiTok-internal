#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-ema-teacher-10f2f6b
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29
PAIR_ROOT="$ROOT/pair1k_rollout_x0_u2_ema_teacher"
MONITOR="$ROOT/pair1k_rollout_x0_u2_ema_teacher_monitor.json"
POSTEVAL=/tmp/generation_stability_rollout_x0_u2_ema_teacher_posteval1k_2026-07-30.sh
EXPECTED_CODE_REVISION=10f2f6bd9977fb1a63de4b2939ca641107a0ccaa
EXPECTED_POSTEVAL_SHA256=dedef29410b6d76c91fc0a626e9f2e1f05e062cf6782d926e8082b8452a434a5
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-14400}
POLL_SECONDS=${POLL_SECONDS:-60}

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ "$(sha256sum "$POSTEVAL" | awk '{print $1}')" == "$EXPECTED_POSTEVAL_SHA256" ]]
[[ "$MAX_WAIT_SECONDS" =~ ^[0-9]+$ ]]
[[ "$POLL_SECONDS" =~ ^[0-9]+$ ]]
(( MAX_WAIT_SECONDS > 0 && POLL_SECONDS > 0 ))
export PYTHONPATH=src

deadline=$(( $(date +%s) + MAX_WAIT_SECONDS ))
while true; do
  if [[ -f "$MONITOR" ]]; then
    state="$("$PYTHON" - "$MONITOR" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"{report.get('status', 'missing')}:{report.get('stage', 'missing')}")
PY
)"
    case "$state" in
      pass:complete)
        break
        ;;
      failed:*|stalled:*)
        printf 'training monitor reached terminal failure: %s\n' "$state" >&2
        exit 20
        ;;
    esac
  fi
  if (( $(date +%s) >= deadline )); then
    printf 'timed out waiting for matched EMA-teacher 1K pair\n' >&2
    exit 21
  fi
  sleep "$POLL_SECONDS"
done

while true; do
  runbook_active=0
  training_active=0
  if pgrep -af '[g]eneration_stability_rollout_x0_u2_ema_teacher_probe1k_2026-07-30.sh' >/dev/null; then
    runbook_active=1
  fi
  if pgrep -af '[s]cripts/train_generation.py.*pair1k_rollout_x0_u2_ema_teacher' >/dev/null; then
    training_active=1
  fi
  if (( runbook_active == 0 && training_active == 0 )) \
    && [[ -f "$PAIR_ROOT/pair_summary.json" ]]; then
    break
  fi
  if (( $(date +%s) >= deadline )); then
    printf 'timed out waiting for pair finalization\n' >&2
    exit 22
  fi
  sleep "$POLL_SECONDS"
done

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing automatic post-evaluation while the GPU is busy\n' >&2
  exit 23
fi

mapfile -t identities < <(
  "$PYTHON" - \
    "$PAIR_ROOT/pair_summary.json" \
    "$PAIR_ROOT/cofitok_rgbtail3_rollout_x0_u2_ema_teacher/training_report.json" \
    "$PAIR_ROOT/dense_rollout_x0_u2_ema_teacher/training_report.json" \
    "$EXPECTED_CODE_REVISION" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


summary_path = Path(sys.argv[1])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected_revision = sys.argv[4]
if (
    summary["status"] != "completed"
    or summary["stage"] != "matched_1k_qualification_probe"
    or summary["git_revision"] != expected_revision
):
    raise SystemExit("pair summary identity is invalid")
checkpoint_shas = []
for method, path in zip(("cofitok", "dense"), map(Path, sys.argv[2:4])):
    report = json.loads(path.read_text(encoding="utf-8"))
    latest = report["latest_checkpoint"]
    if (
        report["training_complete"] is not True
        or report["completed_steps"] != 1000
        or report["target_steps"] != 1000
        or report["git"]["revision"] != expected_revision
        or report["git"]["dirty"]
        or latest["step"] != 1000
    ):
        raise SystemExit(f"{method} training report identity is invalid")
    value = str(latest["checkpoint_sha256"])
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SystemExit(f"{method} checkpoint SHA256 is invalid")
    checkpoint_shas.append(value)
print(sha256(summary_path))
print(*checkpoint_shas, sep="\n")
PY
)
if [[ "${#identities[@]}" != 3 ]]; then
  printf 'post-evaluation identities are incomplete\n' >&2
  exit 24
fi
for identity in "${identities[@]}"; do
  [[ "$identity" =~ ^[0-9a-f]{64}$ ]]
done

EXPECTED_PAIR_SUMMARY_SHA256="${identities[0]}" \
EXPECTED_COFITOK_CHECKPOINT_SHA256="${identities[1]}" \
EXPECTED_DENSE_CHECKPOINT_SHA256="${identities[2]}" \
bash "$POSTEVAL"
