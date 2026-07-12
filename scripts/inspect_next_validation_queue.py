from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from scripts.make_next_validation_queue import DEFAULT_DATE_TAG, NEXT_VALIDATION_RUNS, ValidationRun
except ModuleNotFoundError:
    from make_next_validation_queue import DEFAULT_DATE_TAG, NEXT_VALIDATION_RUNS, ValidationRun


ORDERS = ("ordered", "random", "reverse")


@dataclass(frozen=True)
class QueueMarker:
    label: str
    report_type: str
    marker: Path
    required: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect completion markers for the queued CoFiTok validation batch.")
    parser.add_argument("--checkpoint-root", default="/root/autodl-tmp/CoFiTok/checkpoints")
    parser.add_argument("--date-tag", default=DEFAULT_DATE_TAG)
    parser.add_argument("--quality-images", type=int, default=1024)
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--fail-if-incomplete", action="store_true")
    return parser.parse_args()


def expected_markers(
    checkpoint_root: Path,
    date_tag: str = DEFAULT_DATE_TAG,
    quality_images: int = 1024,
    sample_count: int = 2048,
    sample_steps: int = 50,
    runs: list[ValidationRun] | None = None,
) -> list[QueueMarker]:
    markers: list[QueueMarker] = []
    for run in runs or NEXT_VALIDATION_RUNS:
        train_dir = checkpoint_root / f"{run.train_id}_{date_tag}"
        markers.append(QueueMarker(run.label, "train", train_dir / "report.json"))
        quality_id = f"quality_{run.label}_{quality_images}_t500_lpips_inception_{date_tag}"
        markers.append(QueueMarker(run.label, "quality", checkpoint_root / quality_id / "quality_report.json"))
        if run.order_eval:
            for order in ORDERS:
                order_id = f"order_{run.label}_{order}_{date_tag}"
                markers.append(QueueMarker(run.label, f"order_eval:{order}", checkpoint_root / order_id / "report.json"))
        if run.sample_eval:
            sample_id = f"generated_{run.label}_{sample_count}_ddim{sample_steps}_{date_tag}"
            markers.append(QueueMarker(run.label, "sampling", checkpoint_root / sample_id / "sample_report.json"))
            generated_quality_id = f"generated_quality_{run.label}_{sample_count}_ddim{sample_steps}_{date_tag}"
            markers.append(
                QueueMarker(
                    run.label,
                    "generated_quality",
                    checkpoint_root / generated_quality_id / "generated_quality_report.json",
                )
            )
    return markers


def _marker_record(marker: QueueMarker, checkpoint_root: Path) -> dict[str, Any]:
    exists = marker.marker.is_file()
    try:
        relative = marker.marker.relative_to(checkpoint_root)
    except ValueError:
        relative = marker.marker
    return {
        "label": marker.label,
        "report_type": marker.report_type,
        "required": marker.required,
        "exists": exists,
        "path": marker.marker.as_posix(),
        "relative_path": relative.as_posix(),
        "bytes": marker.marker.stat().st_size if exists else None,
    }


def inspect_queue(
    checkpoint_root: Path,
    date_tag: str = DEFAULT_DATE_TAG,
    quality_images: int = 1024,
    sample_count: int = 2048,
    sample_steps: int = 50,
) -> dict[str, Any]:
    checkpoint_root = checkpoint_root.resolve()
    markers = expected_markers(
        checkpoint_root=checkpoint_root,
        date_tag=date_tag,
        quality_images=quality_images,
        sample_count=sample_count,
        sample_steps=sample_steps,
    )
    records = [_marker_record(marker, checkpoint_root) for marker in markers]
    complete = [record for record in records if record["exists"]]
    missing = [record for record in records if record["required"] and not record["exists"]]
    by_type: dict[str, dict[str, int]] = {}
    for record in records:
        report_type = str(record["report_type"]).split(":", maxsplit=1)[0]
        bucket = by_type.setdefault(report_type, {"expected": 0, "complete": 0, "missing": 0})
        bucket["expected"] += 1
        if record["exists"]:
            bucket["complete"] += 1
        else:
            bucket["missing"] += 1

    runs: list[dict[str, Any]] = []
    for run in NEXT_VALIDATION_RUNS:
        run_records = [record for record in records if record["label"] == run.label]
        run_missing = [record for record in run_records if record["required"] and not record["exists"]]
        runs.append(
            {
                "label": run.label,
                "config": run.config,
                "train_id": run.train_id,
                "order_eval": run.order_eval,
                "sample_eval": run.sample_eval,
                "status": "complete" if not run_missing else "incomplete",
                "expected_count": len(run_records),
                "complete_count": len(run_records) - len(run_missing),
                "missing_count": len(run_missing),
                "markers": run_records,
            }
        )

    return {
        "status": "complete" if not missing else "incomplete",
        "checkpoint_root": checkpoint_root.as_posix(),
        "date_tag": date_tag,
        "quality_images": quality_images,
        "sample_count": sample_count,
        "sample_steps": sample_steps,
        "run_count": len(NEXT_VALIDATION_RUNS),
        "counts": {
            "expected": len(records),
            "complete": len(complete),
            "missing": len(missing),
        },
        "by_type": by_type,
        "runs": runs,
        "missing": missing,
        "queued_runs": [asdict(run) for run in NEXT_VALIDATION_RUNS],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = inspect_queue(
        checkpoint_root=Path(args.checkpoint_root),
        date_tag=args.date_tag,
        quality_images=args.quality_images,
        sample_count=args.sample_count,
        sample_steps=args.sample_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        _write_json(Path(args.output), result)
    if args.fail_if_incomplete and result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
