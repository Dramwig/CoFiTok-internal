#!/usr/bin/env bash
set -euo pipefail

PROJECT=/tmp/cofitok-generation-stability-ema-teacher-10f2f6b
PYTHON=/root/autodl-tmp/conda/envs/pf-vlm/bin/python
ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation/stability_probe_2026-07-29
PAIR_ROOT="$ROOT/pair1k_rollout_x0_u2_ema_teacher"
SCREENING_SUMMARY="$ROOT/qualification1k_rollout_x0_u2_ema_teacher/screening_summary.json"
SCREENING_QUALIFICATION="$ROOT/qualification1k_rollout_x0_u2_ema_teacher/model/qualification_report.json"
N64_RUNBOOK=/tmp/generation_stability_rollout_x0_u2_ema_teacher_n64_posteval1k_2026-07-31.sh
EXPECTED_CODE_REVISION=10f2f6bd9977fb1a63de4b2939ca641107a0ccaa
EXPECTED_N64_RUNBOOK_SHA256=04ff2aa80762a0e4a25c0608520ef4740c54ce49192d64380321141ee4ddf919
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-21600}
POLL_SECONDS=${POLL_SECONDS:-60}

cd "$PROJECT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_CODE_REVISION" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -x "$PYTHON" ]]
[[ "$(sha256sum "$N64_RUNBOOK" | awk '{print $1}')" == "$EXPECTED_N64_RUNBOOK_SHA256" ]]
[[ "$MAX_WAIT_SECONDS" =~ ^[0-9]+$ ]]
[[ "$POLL_SECONDS" =~ ^[0-9]+$ ]]
(( MAX_WAIT_SECONDS > 0 && POLL_SECONDS > 0 ))
export PYTHONPATH=src

deadline=$(( $(date +%s) + MAX_WAIT_SECONDS ))
while [[ ! -f "$SCREENING_SUMMARY" ]]; do
  if (( $(date +%s) >= deadline )); then
    printf 'timed out waiting for EMA-teacher n8 screening\n' >&2
    exit 21
  fi
  sleep "$POLL_SECONDS"
done

screening_status="$("$PYTHON" - "$SCREENING_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    report.get("status") != "completed"
    or report.get("stage") != "matched_1k_n8_screening"
    or report.get("scaling_authorization_allowed") is not False
):
    raise SystemExit("n8 screening summary identity is invalid")
raw = report["authoritative_raw_model_qualification"]
print(raw["status"])
if raw["status"] != "pass":
    print(",".join(sorted(raw["failed_gates"])), file=sys.stderr)
PY
)"
if [[ "$screening_status" != pass ]]; then
  printf 'raw n8 screening did not pass; skipping n64: %s\n' "$screening_status" >&2
  exit 20
fi

while pgrep -af '[g]eneration_stability_rollout_x0_u2_ema_teacher_posteval1k_2026-07-30.sh' >/dev/null; do
  if (( $(date +%s) >= deadline )); then
    printf 'timed out waiting for n8 post-evaluation to exit\n' >&2
    exit 22
  fi
  sleep "$POLL_SECONDS"
done

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'refusing automatic raw n64 evaluation while the GPU is busy\n' >&2
  exit 23
fi

mapfile -t identities < <(
  "$PYTHON" - \
    "$PAIR_ROOT/pair_summary.json" \
    "$SCREENING_QUALIFICATION" \
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


pair_path = Path(sys.argv[1])
screening_path = Path(sys.argv[2])
expected_revision = sys.argv[5]
pair = json.loads(pair_path.read_text(encoding="utf-8"))
screening = json.loads(screening_path.read_text(encoding="utf-8"))
if (
    pair["status"] != "completed"
    or pair["stage"] != "matched_1k_qualification_probe"
    or pair["git_revision"] != expected_revision
    or screening["status"] != "pass"
    or screening["identity"]["git_revision"] != expected_revision
):
    raise SystemExit("pair or screening identity is invalid")
checkpoint_shas = []
for method, path in zip(("cofitok", "dense"), map(Path, sys.argv[3:5])):
    report = json.loads(path.read_text(encoding="utf-8"))
    latest = report["latest_checkpoint"]
    if (
        report["training_complete"] is not True
        or report["completed_steps"] != 1000
        or report["git"]["revision"] != expected_revision
        or report["git"]["dirty"]
        or latest["step"] != 1000
    ):
        raise SystemExit(f"{method} training report identity is invalid")
    value = str(latest["checkpoint_sha256"])
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise SystemExit(f"{method} checkpoint SHA256 is invalid")
    checkpoint_shas.append(value)
print(sha256(pair_path))
print(sha256(screening_path))
print(*checkpoint_shas, sep="\n")
PY
)
if [[ "${#identities[@]}" != 4 ]]; then
  printf 'raw n64 identities are incomplete\n' >&2
  exit 24
fi
for identity in "${identities[@]}"; do
  [[ "$identity" =~ ^[0-9a-f]{64}$ ]]
done

EXPECTED_PAIR_SUMMARY_SHA256="${identities[0]}" \
EXPECTED_SCREENING_QUALIFICATION_SHA256="${identities[1]}" \
EXPECTED_COFITOK_CHECKPOINT_SHA256="${identities[2]}" \
EXPECTED_DENSE_CHECKPOINT_SHA256="${identities[3]}" \
bash "$N64_RUNBOOK"
