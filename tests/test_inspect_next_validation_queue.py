from pathlib import Path

from scripts.inspect_next_validation_queue import expected_markers, inspect_queue


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")


def test_inspect_queue_reports_missing_markers(tmp_path: Path) -> None:
    result = inspect_queue(tmp_path)

    assert result["status"] == "incomplete"
    assert result["counts"]["expected"] == 56
    assert result["counts"]["complete"] == 0
    assert result["counts"]["missing"] == 56
    assert result["by_type"]["train"] == {"expected": 10, "complete": 0, "missing": 10}
    assert result["by_type"]["quality"] == {"expected": 10, "complete": 0, "missing": 10}
    assert result["by_type"]["order_eval"] == {"expected": 24, "complete": 0, "missing": 24}
    assert result["by_type"]["sampling"] == {"expected": 6, "complete": 0, "missing": 6}


def test_inspect_queue_reports_complete_markers(tmp_path: Path) -> None:
    for marker in expected_markers(tmp_path):
        _touch(marker.marker)

    result = inspect_queue(tmp_path)

    assert result["status"] == "complete"
    assert result["counts"] == {"expected": 56, "complete": 56, "missing": 0}
    assert all(run["status"] == "complete" for run in result["runs"])


def test_inspect_queue_custom_sample_dimensions(tmp_path: Path) -> None:
    for marker in expected_markers(tmp_path, quality_images=256, sample_count=128, sample_steps=20):
        _touch(marker.marker)

    result = inspect_queue(tmp_path, quality_images=256, sample_count=128, sample_steps=20)
    default_result = inspect_queue(tmp_path)

    assert result["status"] == "complete"
    assert default_result["status"] == "incomplete"
