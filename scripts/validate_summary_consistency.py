from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


SUMMARY_KEYS = ("train", "order_eval", "quality", "sampling", "generated_quality", "official_fid")
CSV_FILENAMES = {
    "train": "train_summary.csv",
    "order_eval": "order_eval_summary.csv",
    "quality": "quality_summary.csv",
    "sampling": "sample_summary.csv",
    "generated_quality": "generated_quality_summary.csv",
    "official_fid": "official_fid_summary.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CoFiTok summary artifact consistency.")
    parser.add_argument(
        "--summary-dir",
        default="artifacts/reports/summary_2026-07-08",
        help="Directory containing experiment_summary.json and summary CSV files.",
    )
    parser.add_argument(
        "--doc",
        action="append",
        default=[],
        help="Markdown document whose summary counts should match experiment_summary.json. Can be repeated.",
    )
    parser.add_argument(
        "--require-report-dir",
        action="append",
        default=[],
        help="Report directory name that must appear in at least one summary row. Can be repeated.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV summary: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def _extract_doc_counts(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    counts: dict[str, int] = {}
    for key in SUMMARY_KEYS:
        match = re.search(rf"^\s*{re.escape(key)}:\s*(\d+)\s*$", text, flags=re.MULTILINE)
        if match:
            counts[key] = int(match.group(1))
    return counts


def _all_report_dirs(summary: dict[str, Any]) -> set[str]:
    report_dirs = set()
    for key in SUMMARY_KEYS:
        for row in summary.get(key, []):
            report_dir = row.get("report_dir")
            if isinstance(report_dir, str):
                report_dirs.add(report_dir)
    return report_dirs


def validate_summary(
    summary_dir: Path,
    docs: list[Path] | None = None,
    required_report_dirs: list[str] | None = None,
) -> dict[str, Any]:
    docs = docs or []
    required_report_dirs = required_report_dirs or []
    summary_path = summary_dir / "experiment_summary.json"
    summary = _read_json(summary_path)
    counts = summary.get("counts")
    if not isinstance(counts, dict):
        raise AssertionError("experiment_summary.json is missing a counts object")

    json_counts: dict[str, int] = {}
    csv_counts: dict[str, int] = {}
    for key in SUMMARY_KEYS:
        rows = summary.get(key)
        if not isinstance(rows, list):
            raise AssertionError(f"summary key {key!r} is missing or not a list")
        actual = len(rows)
        declared = counts.get(key)
        if actual != declared:
            raise AssertionError(f"counts.{key}={declared} but len({key})={actual}")
        json_counts[key] = actual

        csv_path = summary_dir / CSV_FILENAMES[key]
        csv_count = _csv_row_count(csv_path)
        if csv_count != actual:
            raise AssertionError(f"{CSV_FILENAMES[key]} has {csv_count} rows but summary has {actual}")
        csv_counts[key] = csv_count

    doc_counts: dict[str, dict[str, int]] = {}
    for doc in docs:
        counts_in_doc = _extract_doc_counts(doc)
        missing = [key for key in SUMMARY_KEYS if key not in counts_in_doc]
        if missing:
            raise AssertionError(f"{doc} is missing summary count lines for: {', '.join(missing)}")
        for key in SUMMARY_KEYS:
            if counts_in_doc[key] != json_counts[key]:
                raise AssertionError(f"{doc} has {key}: {counts_in_doc[key]} but summary has {json_counts[key]}")
        doc_counts[doc.as_posix()] = counts_in_doc

    report_dirs = _all_report_dirs(summary)
    missing_dirs = [name for name in required_report_dirs if name not in report_dirs]
    if missing_dirs:
        raise AssertionError(f"Required report_dir values missing from summary: {', '.join(missing_dirs)}")

    return {
        "summary": summary_path.as_posix(),
        "json_counts": json_counts,
        "csv_counts": csv_counts,
        "doc_counts": doc_counts,
        "required_report_dirs": required_report_dirs,
        "status": "ok",
    }


def main() -> None:
    args = parse_args()
    result = validate_summary(
        Path(args.summary_dir),
        docs=[Path(path) for path in args.doc],
        required_report_dirs=list(args.require_report_dir),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
