from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "artifacts" / "reports" / "aaai27_ablation_2026-07-11"
FIGURES = ROOT / "artifacts" / "figures" / "ablation_hyperparameters_2026-07-11"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_remote_suite_status_is_complete() -> None:
    status = read_json(PACKAGE / "suite_status.json")
    assert status["complete"] is True
    assert status["failed_count"] == 0
    assert status["entry_count"] == 34
    assert sum(row["stage"] == "train" and row["status"] == "completed" for row in status["results"]) == 22
    assert sum(row["stage"] == "eval" and row["status"] == "completed" for row in status["results"]) == 34


def test_ablation_report_uses_common_matched_protocol() -> None:
    report = read_json(PACKAGE / "ablation_report.json")
    assert report["summary"]["run_count"] == 34
    assert report["training_protocol"]["seeds"] == [103, 139]
    assert report["training_protocol"]["steps"] == 5000
    assert report["evaluation_protocol"]["image_count"] == 1024
    assert report["evaluation_protocol"]["timestep"] == 500
    assert report["evaluation_protocol"]["evaluation_progress_power"] == 1.5
    assert report["summary"]["restricted_zero_ratio_max"] == 0.0
    assert report["summary"]["deep_synthesis_zero_ratio_nonzero"] is True
    assert report["summary"]["full_vs_no_path_prefix"]["full_path_better_count"] == 2
    assert report["summary"]["full_vs_no_path_component"]["full_path_better_count"] == 2


def test_ablation_package_contains_raw_reports_tables_and_curves() -> None:
    assert len(list((PACKAGE / "raw" / "train").glob("*.json"))) == 34
    assert len(list((PACKAGE / "raw" / "quality").glob("*.json"))) == 34
    for name in (
        "component_ablation_rows.tex",
        "architecture_control_rows.tex",
        "hyperparameter_sweep_rows.tex",
        "hyperparameter_sweeps.csv",
    ):
        assert (PACKAGE / name).is_file()
    for name in (
        "aaai27_ablation_hyperparameters.pdf",
        "aaai27_ablation_hyperparameters.png",
        "figure_manifest.json",
    ):
        assert (FIGURES / name).is_file()
