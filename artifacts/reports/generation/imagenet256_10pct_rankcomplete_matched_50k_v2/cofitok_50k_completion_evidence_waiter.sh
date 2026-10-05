#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/CoFiTok/CoFiTok-internal
OUTPUT_ROOT=/root/autodl-tmp/CoFiTok/checkpoints/generation
RUN_DIR="$OUTPUT_ROOT/imagenet256_10pct_rankcomplete_cofitok_k8_50k_v2"
REPORT_ROOT="$PROJECT/artifacts/reports/generation/imagenet256_10pct_rankcomplete_matched_50k_v2"
TRAINING_REPORT="$RUN_DIR/training_report.json"
LATEST="$RUN_DIR/latest.json"
CHECKPOINT="$RUN_DIR/checkpoint_step_00050000.pt"
SIDECAR="$CHECKPOINT.integrity.json"
AUDIT="$REPORT_ROOT/cofitok_progress_step_00050000.json"
VERIFICATION="$REPORT_ROOT/cofitok_50k_completion_collector_verification.json"
POLL_SECONDS=240
MAX_POLLS=540

for (( poll=1; poll<=MAX_POLLS; poll++ )); do
  if [[ -f "$TRAINING_REPORT" && -f "$LATEST" && -f "$CHECKPOINT" && -f "$SIDECAR" ]]; then
    if /root/miniconda3/bin/python - "$TRAINING_REPORT" "$LATEST" <<'PY'
import json
import sys
from pathlib import Path

training = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
latest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if training.get("training_complete") is not True:
    raise SystemExit(1)
if int(training.get("completed_steps", -1)) != 50000:
    raise SystemExit(1)
if int(training.get("target_steps", -1)) != 50000:
    raise SystemExit(1)
if training.get("latest_checkpoint") != latest:
    raise SystemExit(1)
if int(latest.get("step", -1)) != 50000:
    raise SystemExit(1)
if latest.get("checkpoint") != "checkpoint_step_00050000.pt":
    raise SystemExit(1)
PY
    then
      break
    fi
  fi
  if (( poll == MAX_POLLS )); then
    printf 'timed out waiting for completed CoFiTok step-50000 evidence\n' >&2
    exit 124
  fi
  sleep "$POLL_SECONDS"
done

cd "$PROJECT"
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=

/root/miniconda3/bin/python scripts/audit_generation_training_progress.py \
  --run-dir "$RUN_DIR" --expected-steps 50000 \
  --checkpoint-interval 5000 --evaluation-interval 1000 \
  --required-checkpoint-steps 50000 --grad-clip-norm 1.0 \
  --integrity-policy required --output "$AUDIT"

/root/miniconda3/bin/python - "$AUDIT" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if audit.get("status") != "healthy":
    raise SystemExit("step-50000 progress audit is not healthy")
if audit.get("issues") or audit.get("warnings"):
    raise SystemExit("step-50000 progress audit has issues or warnings")
if int(audit.get("last_step", -1)) != 50000:
    raise SystemExit("step-50000 progress audit has the wrong final step")
validation = audit.get("validation", {})
if int(validation.get("event_count", -1)) != 50:
    raise SystemExit("step-50000 progress audit lacks 50 validation events")
if validation.get("logging_complete") is not True:
    raise SystemExit("step-50000 validation logging is incomplete")
checkpoint = audit.get("checkpoint", {})
if checkpoint.get("latest_integrity", {}).get("status") != "verified":
    raise SystemExit("step-50000 checkpoint integrity is not verified")
if 50000 not in checkpoint.get("steps", []):
    raise SystemExit("step-50000 checkpoint is absent from the audit")
PY

for pair in \
  "$TRAINING_REPORT:$REPORT_ROOT/cofitok_training_report_step_00050000.json" \
  "$LATEST:$REPORT_ROOT/cofitok_latest_step_00050000.json" \
  "$SIDECAR:$REPORT_ROOT/cofitok_checkpoint_step_00050000.pt.integrity.json"
do
  source_path="${pair%%:*}"
  destination_path="${pair#*:}"
  temporary_path="${destination_path}.tmp.$$"
  cp -- "$source_path" "$temporary_path"
  mv -- "$temporary_path" "$destination_path"
done

export RUN_CHECKPOINT="$CHECKPOINT"
/root/miniconda3/bin/python - \
  "$TRAINING_REPORT" "$REPORT_ROOT/cofitok_training_report_step_00050000.json" \
  "$LATEST" "$REPORT_ROOT/cofitok_latest_step_00050000.json" \
  "$SIDECAR" "$REPORT_ROOT/cofitok_checkpoint_step_00050000.pt.integrity.json" \
  "$AUDIT" "$VERIFICATION" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def metadata(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.resolve().as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


training_source = Path(sys.argv[1])
training_destination = Path(sys.argv[2])
latest_source = Path(sys.argv[3])
latest_destination = Path(sys.argv[4])
sidecar_source = Path(sys.argv[5])
sidecar_destination = Path(sys.argv[6])
audit_path = Path(sys.argv[7])
output_path = Path(sys.argv[8])
pairs = []
for role, source, destination in (
    ("training_report", training_source, training_destination),
    ("latest_checkpoint_binding", latest_source, latest_destination),
    ("checkpoint_integrity_sidecar", sidecar_source, sidecar_destination),
):
    source_metadata = metadata(source)
    destination_metadata = metadata(destination)
    pairs.append(
        {
            "role": role,
            "source": source_metadata,
            "destination": destination_metadata,
            "match": (
                source_metadata["bytes"] == destination_metadata["bytes"]
                and source_metadata["sha256"] == destination_metadata["sha256"]
            ),
        }
    )
audit = json.loads(audit_path.read_text(encoding="utf-8"))
sidecar = json.loads(sidecar_source.read_text(encoding="utf-8"))
report = {
    "schema_version": 1,
    "status": "pass" if all(pair["match"] for pair in pairs) else "fail",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "role": "cofitok_step50000_immediate_read_only_completion_evidence",
    "gpu_used": False,
    "process_control_used": False,
    "checkpoint_copied": False,
    "checkpoint_loaded": False,
    "checkpoint": {
        "path": Path(os.environ["RUN_CHECKPOINT"]).resolve().as_posix(),
        "bytes": int(sidecar["checkpoint_bytes"]),
        "sha256": sidecar["checkpoint_sha256"],
        "step": int(sidecar["step"]),
    },
    "audit": {
        **metadata(audit_path),
        "status": audit["status"],
        "last_step": int(audit["last_step"]),
        "validation_event_count": int(audit["validation_event_count"]),
        "integrity_status": audit["checkpoint"]["latest_integrity"]["status"],
        "issue_count": len(audit["issues"]),
        "warning_count": len(audit["warnings"]),
    },
    "copies": pairs,
}
temporary_path = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary_path, output_path)
if report["status"] != "pass":
    raise SystemExit("completion evidence copy verification failed")
print(json.dumps(report, indent=2, sort_keys=True))
PY
