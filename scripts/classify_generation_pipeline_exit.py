from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RETRYABLE_STAGES = {"posteval_10pct", "full_training", "full_posteval"}


def classify_pipeline_exit(
    pipeline_status: dict[str, Any], *, exit_code: int
) -> dict[str, Any]:
    stage = str(pipeline_status.get("stage", "unknown"))
    status = str(pipeline_status.get("status", "unknown"))
    if exit_code == 0 and status == "pass" and stage == "complete":
        decision = "pass"
        reason = "pipeline_completed"
    elif stage in RETRYABLE_STAGES and status in {"running", "failed"}:
        decision = "retry"
        reason = "recoverable_stage_exit"
    else:
        decision = "stop"
        reason = "nonretryable_or_scientific_stage"
    return {
        "schema_version": 1,
        "decision": decision,
        "reason": reason,
        "pipeline_stage": stage,
        "pipeline_status": status,
        "exit_code": int(exit_code),
        "retryable_stages": sorted(RETRYABLE_STAGES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a completion-pipeline exit for bounded supervision."
    )
    parser.add_argument("--status", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    path = Path(args.status)
    try:
        with path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        status = {"stage": "unknown", "status": "unknown"}
    report = classify_pipeline_exit(status, exit_code=args.exit_code)
    print(
        f"{report['decision']} {report['pipeline_stage']} "
        f"{report['pipeline_status']}"
    )


if __name__ == "__main__":
    main()
