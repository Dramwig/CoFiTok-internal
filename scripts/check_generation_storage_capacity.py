from __future__ import annotations

import argparse
import math
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.reporting import git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
KIB = 1024
INSUFFICIENT_CAPACITY_EXIT = 78


def build_storage_capacity_report(
    *,
    stage: str,
    path: Path,
    total_bytes: int,
    used_bytes: int,
    free_bytes: int,
    checkpoint_count: int,
    checkpoint_bytes: int,
    checkpoint_size_multiplier: float = 1.0,
    sample_count: int,
    estimated_sample_bytes: int,
    additional_bytes: int,
    safety_margin_bytes: int,
    git: dict[str, Any],
    hostname: str,
    checked_at: str,
) -> dict[str, Any]:
    counts = (checkpoint_count, checkpoint_bytes, sample_count, estimated_sample_bytes)
    if not stage or any(value < 0 for value in counts):
        raise ValueError("storage capacity plan has invalid non-negative inputs")
    if (
        not math.isfinite(checkpoint_size_multiplier)
        or checkpoint_size_multiplier < 1.0
    ):
        raise ValueError("checkpoint size multiplier must be finite and at least one")
    if additional_bytes < 0 or safety_margin_bytes < 0:
        raise ValueError("storage capacity reserves must be non-negative")
    if total_bytes < 1 or used_bytes < 0 or free_bytes < 0:
        raise ValueError("filesystem usage is invalid")
    if used_bytes + free_bytes > total_bytes:
        raise ValueError("filesystem usage exceeds total capacity")

    planned_checkpoint_bytes = math.ceil(
        checkpoint_bytes * checkpoint_size_multiplier
    )
    checkpoint_reserve = checkpoint_count * planned_checkpoint_bytes
    sample_reserve = sample_count * estimated_sample_bytes
    required = checkpoint_reserve + sample_reserve + additional_bytes + safety_margin_bytes
    passed = free_bytes >= required
    return {
        "schema_version": 2,
        "role": "generation_storage_capacity_preflight",
        "stage": stage,
        "status": "pass" if passed else "fail",
        "git": git,
        "hostname": hostname,
        "checked_at": checked_at,
        "filesystem": {
            "path": path.resolve().as_posix(),
            "total_bytes": total_bytes,
            "used_bytes": used_bytes,
            "free_bytes": free_bytes,
        },
        "plan": {
            "checkpoint_count": checkpoint_count,
            "reference_checkpoint_bytes_each": checkpoint_bytes,
            "checkpoint_size_multiplier": checkpoint_size_multiplier,
            "checkpoint_bytes_each": planned_checkpoint_bytes,
            "checkpoint_reserve_bytes": checkpoint_reserve,
            "sample_count": sample_count,
            "estimated_sample_bytes_each": estimated_sample_bytes,
            "sample_reserve_bytes": sample_reserve,
            "additional_bytes": additional_bytes,
            "safety_margin_bytes": safety_margin_bytes,
            "required_free_bytes": required,
        },
        "headroom_bytes": free_bytes - required,
    }


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail before a generation stage when planned artifacts exceed free storage."
    )
    parser.add_argument("--path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--reference-checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-count", type=int, default=0)
    parser.add_argument(
        "--checkpoint-size-multiplier",
        type=_non_negative_float,
        default=1.0,
        help=(
            "Multiply the largest measured reference checkpoint before reserving "
            "space; values below one are rejected."
        ),
    )
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--estimated-sample-kib", type=_non_negative_float, default=256.0)
    parser.add_argument("--additional-gib", type=_non_negative_float, default=16.0)
    parser.add_argument("--safety-margin-gib", type=_non_negative_float, default=64.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path).resolve()
    if not path.is_dir():
        raise SystemExit(f"storage capacity path is not a directory: {path}")
    if args.checkpoint_count < 0 or args.sample_count < 0:
        raise SystemExit("checkpoint and sample counts must be non-negative")
    if args.checkpoint_size_multiplier < 1.0:
        raise SystemExit("checkpoint size multiplier must be at least one")

    references = [Path(value).resolve() for value in args.reference_checkpoint]
    missing = [str(value) for value in references if not value.is_file()]
    if missing:
        raise SystemExit(f"reference checkpoints are missing: {missing}")
    checkpoint_bytes = max((value.stat().st_size for value in references), default=0)
    if args.checkpoint_count > 0 and checkpoint_bytes < 1:
        raise SystemExit("planned checkpoints require at least one reference checkpoint")

    usage = shutil.disk_usage(path)
    report = build_storage_capacity_report(
        stage=args.stage,
        path=path,
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        checkpoint_count=args.checkpoint_count,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_size_multiplier=args.checkpoint_size_multiplier,
        sample_count=args.sample_count,
        estimated_sample_bytes=int(round(args.estimated_sample_kib * KIB)),
        additional_bytes=int(round(args.additional_gib * GIB)),
        safety_margin_bytes=int(round(args.safety_margin_gib * GIB)),
        git=git_provenance(PROJECT_ROOT),
        hostname=socket.gethostname(),
        checked_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json_report(args.output, report)
    print(args.output)
    if report["status"] != "pass":
        print(
            "insufficient generation storage capacity: "
            f"free={usage.free} required={report['plan']['required_free_bytes']}",
            file=sys.stderr,
        )
        raise SystemExit(INSUFFICIENT_CAPACITY_EXIT)


if __name__ == "__main__":
    main()
