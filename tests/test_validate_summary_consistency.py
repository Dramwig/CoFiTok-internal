import csv
import json
from pathlib import Path

import pytest

from scripts.validate_summary_consistency import CSV_FILENAMES, SUMMARY_KEYS, validate_summary


def _write_summary_bundle(root: Path, counts: dict[str, int] | None = None) -> None:
    counts = counts or {key: 1 for key in SUMMARY_KEYS}
    payload = {"counts": counts}
    for key in SUMMARY_KEYS:
        payload[key] = [{"report_dir": f"{key}_report"} for _ in range(counts[key])]
    (root / "experiment_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    for key in SUMMARY_KEYS:
        path = root / CSV_FILENAMES[key]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["report_dir"])
            writer.writeheader()
            for row in payload[key]:
                writer.writerow(row)


def test_validate_summary_accepts_matching_json_csv_and_docs(tmp_path) -> None:
    _write_summary_bundle(tmp_path)
    doc = tmp_path / "report.md"
    doc.write_text(
        "\n".join(
            [
                "summary counts:",
                "  train: 1",
                "  order_eval: 1",
                "  quality: 1",
                "  sampling: 1",
                "  generated_quality: 1",
                "  official_fid: 1",
            ]
        ),
        encoding="utf-8",
    )

    result = validate_summary(
        tmp_path,
        docs=[doc],
        required_report_dirs=["quality_report"],
    )

    assert result["status"] == "ok"
    assert result["json_counts"]["quality"] == 1
    assert result["doc_counts"][doc.as_posix()]["generated_quality"] == 1


def test_validate_summary_rejects_stale_csv(tmp_path) -> None:
    _write_summary_bundle(tmp_path, counts={key: 1 for key in SUMMARY_KEYS})
    (tmp_path / CSV_FILENAMES["quality"]).write_text("report_dir\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="quality_summary.csv"):
        validate_summary(tmp_path)
